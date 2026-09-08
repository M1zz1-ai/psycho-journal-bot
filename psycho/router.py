"""Telegram routing + bot wiring (n8n "Bot Router" N8N0000000000001).

The n8n router read the per-chat ``psycho:awaiting:<chat>`` flag from redis, ran a
Switch (start / analyze-button / awaiting-period / journal), and pushed journal
entries to ``psycho:session:<ts>``. Voice notes were transcribed via OpenAI
Whisper before being logged.

This module is the Python edge of that router:

* ``build_psycho_bot`` wires the shared core (``core.tg`` + ``core.openai_agent`` factory
  + redis session store) into a :class:`~psycho.bot.PsychoBot`.
* ``build_dispatcher`` maps aiogram messages onto the bot's handlers — the same
  switch, now expressed as aiogram filters + ``tools.classify``.

Provider swaps from the n8n flow (deliberate, see ``tools`` docstring):
  * Routing/state classification: n8n used redis + a Code-node switch; here it's
    ``tools.classify`` over the same redis ``awaiting`` flag.
  * Voice STT: n8n used OpenAI Whisper inline. Same here — the dispatcher
    downloads the voice OGG/Opus and transcribes it via ``core.stt`` (Whisper)
    before logging. A caption, when present, is used verbatim and skips STT.
    A voice note is always logged as a journal entry.
  * All LLM calls go through ``core.openai_agent`` (OpenAI). The brain moved from
    Anthropic to OpenAI when the direct Anthropic key ran out of credit;
    ``OPENAI_API_KEY`` (already used for Whisper STT) powers it now.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable

import openai
from aiogram import Dispatcher
from aiogram.filters import Command
from aiogram.types import Message

from core import config, state, stt, tg, tg_auth
from core import openai_agent as core_agent

from .bot import HEARTBEAT_KEY, PsychoBot
from .session_store import PsychoSessionStore

logger = logging.getLogger("psycho_bot")

REDIS_NAMESPACE = "psycho"

# Gates the approval-card sink. Off by default: see ``_emit_cards_enabled``.
PSYCHO_EMIT_CARDS_ENV = "PSYCHO_EMIT_CARDS"


def _agent_factory(client: openai.OpenAI):
    """Build the ``(system_prompt, **kw) -> OpenAIAgent`` factory PsychoBot needs.

    Each task (period parse / analysis / structure / therapist) gets a fresh
    agent with its own system prompt and per-task model (``ANALYSIS_MODEL`` /
    ``REPORT_MODEL``) passed via ``**kw``.
    """

    def make(system: str, **kw: object) -> core_agent.OpenAIAgent:
        return core_agent.OpenAIAgent(client, system=system, **kw)

    return make


def _make_card_sink(redis_state: state.RedisState):
    """Build the optional card sink: analysis -> action_request card -> redis.

    The analysis is turned into an ``action_request`` card and RPUSHed onto
    the local ``approval:psycho`` list for a separate review queue to drain.
    Reuses the ledger's own redis client. Degrades gracefully inside
    ``push_card`` — a redis blip never crashes analysis.

    Imports ``card`` lazily (rather than at module level) so importing this
    module never requires that submodule to be present — the sink is built
    only when :func:`_emit_cards_enabled` is true.
    """
    from . import card as psycho_card

    def sink(analysis: str, *, period_label: str, source_session: str | None = None) -> None:
        item = psycho_card.build_analysis_card(
            analysis, period_label=period_label, source_session=source_session
        )
        psycho_card.push_card(redis_state, item)

    return sink


def _emit_cards_enabled() -> bool:
    """Whether ``build_psycho_bot`` should ALSO wire the approval-card sink.

    Off by default, checked via ``os.getenv`` (same pattern as
    ``PSYCHO_INPROC_REPORT`` in ``__main__``) rather than through
    ``config.Config``, since this is a deploy-time feature flag, not a secret.

    History, so the next reader does not restore this blind: from 2026-07-24
    ``build_psycho_bot`` wired the sink UNCONDITIONALLY ("pure producer" mode)
    and ``psycho.bot`` sent every analysis and every weekly report to the
    card sink INSTEAD of Telegram. The sink's only consumer was retired that
    same day, and its successor drains a different queue and does not read
    ``approval:psycho`` at all. So for six weeks every psycho analysis and
    report was written to a redis list nobody drains, and no report reached
    its owner (diagnosed 2026-09-08: the list held exactly two entries, both
    from that first day). Telegram delivery in ``psycho.bot`` is
    unconditional now regardless of this flag; only set ``PSYCHO_EMIT_CARDS=1``
    once something drains ``approval:psycho`` again, to also keep a parallel
    evidence trail there.

    Operator note: this flag will NOT take effect from a dotenv-style config
    file. ``os.getenv`` reads the real process environment, so it must come
    from an actual ``Environment=``/``EnvironmentFile=`` line in the service
    manager's unit (or an exported shell variable before the process starts)
    — not from ``core.config.load``, which reads the master env file with
    ``dotenv_values`` and never touches ``os.environ``. Getting this wrong
    fails safe (the sink just stays off), so it looks like nothing is wrong
    until someone goes looking for the missing card.
    """
    return os.getenv(PSYCHO_EMIT_CARDS_ENV, "0") == "1"


def build_psycho_bot(cfg: config.Config) -> tuple[PsychoBot, tg.TelegramClient]:
    """Wire the shared core into a PsychoBot from a loaded config.

    Telegram is always the delivery path (see the ``psycho.bot`` module
    docstring). The approval-card sink is an optional, additional artifact,
    gated behind ``PSYCHO_EMIT_CARDS=1``: see ``_emit_cards_enabled`` for why
    it defaults off.
    """
    chat_id = tg.gather_chat_ids(cfg.require("TELEGRAM_CHAT_ID"))[0]
    telegram = tg.TelegramClient.from_token(cfg.require("TELEGRAM_BOT_TOKEN_PSYCHO"), chat_id)
    client = openai.OpenAI(api_key=cfg.require("OPENAI_API_KEY"))
    redis_url = cfg.get("REDIS_URL") or state.DEFAULT_URL
    redis_state = state.RedisState(redis_url, namespace=REDIS_NAMESPACE)
    store = PsychoSessionStore(redis_state)
    bot = PsychoBot(
        telegram=telegram,
        agent_factory=_agent_factory(client),
        state=store,
        owner_chat_id=chat_id,
        card_sink=_make_card_sink(redis_state) if _emit_cards_enabled() else None,
    )
    return bot, telegram


def read_report_heartbeat(cfg: config.Config) -> str | None:
    """Read the last-delivered-report timestamp without wiring the full bot.

    Lets ``python -m psycho --check`` answer "did the weekly report
    actually reach Telegram lately" as one command, instead of a manual
    redis-cli lookup. Builds a bare ``state.RedisState`` (no Telegram client,
    no OpenAI client), so ``--check`` stays cheap and side-effect free.

    Returns ``None`` if the heartbeat key is missing, expired, or redis is
    unreachable: ``RedisState.get_session`` already degrades to ``None`` on a
    redis failure, so both cases read identically to the caller.
    """
    redis_url = cfg.get("REDIS_URL") or state.DEFAULT_URL
    redis_state = state.RedisState(redis_url, namespace=REDIS_NAMESPACE)
    return redis_state.get_session(HEARTBEAT_KEY)


def build_dispatcher(bot: PsychoBot, *, allowed_chat_ids: Iterable[int]) -> Dispatcher:
    """Map aiogram messages onto the PsychoBot handlers (the n8n Main Router).

    ``allowed_chat_ids`` is REQUIRED and keyword-only. This is the most private
    surface in the fleet: anything sent here is journalled into Bogdan's psycho
    sessions and analysed, and until the allowlist landed (2026-08-14) a stranger
    could both write into that journal and pull an analysis out of it. The check
    sits at the one place every handler passes through; unauthorized updates are
    dropped silently and logged (:mod:`core.tg_auth`).
    """
    dp = Dispatcher()
    tg_auth.install_chat_allowlist(dp, allowed_chat_ids)

    @dp.message(Command("start", "help"))
    async def _on_start(message: Message) -> None:
        await bot.on_text(message.chat.id, "/start")

    @dp.message(lambda m: m.voice is not None or m.audio is not None)
    async def _on_voice(message: Message) -> None:
        # A caption, when present, is the transcript; otherwise download the voice
        # OGG/Opus and transcribe it via Whisper (core.stt). Bogdan dictates notes
        # without a caption, so STT is the primary path here.
        media = message.voice or message.audio
        duration = getattr(media, "duration", 0) or 0
        transcript = (message.caption or "").strip()
        if not transcript:
            try:
                f = await message.bot.get_file(media.file_id)
                buf = await message.bot.download_file(f.file_path)
                audio_bytes = buf.read() if hasattr(buf, "read") else bytes(buf)
                transcript = (await stt.transcribe(audio_bytes, language="ru")).strip()
            except Exception:  # noqa: BLE001 — one bad note must not kill the poll loop
                logger.exception("psycho voice transcription failed; note skipped")
                return
        if transcript:
            await bot.on_voice_transcript(message.chat.id, transcript, duration_sec=duration)

    @dp.message(lambda m: m.text is not None)
    async def _on_text(message: Message) -> None:
        await bot.on_text(message.chat.id, message.text or "")

    return dp
