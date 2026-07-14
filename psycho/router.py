"""Telegram routing + bot wiring (n8n "Bot Router" ).

The n8n router read the per-chat ``psycho:awaiting:<chat>`` flag from redis, ran a
Switch (start / analyze-button / awaiting-period / journal), and pushed journal
entries to ``psycho:session:<ts>``. Voice notes were transcribed via OpenAI
Whisper before being logged.

This module is the Python edge of that router:

* ``build_psycho_bot`` wires the shared core (``core.tg`` + ``core.agent`` factory
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
    Anthropic to OpenAI when the direct Anthropic key ran out of credits on the
    server; ``OPENAI_API_KEY`` (already used for Whisper STT) powers it now.
"""

from __future__ import annotations

import logging

import openai
from aiogram import Dispatcher
from aiogram.filters import Command
from aiogram.types import Message

from core import config, state, stt, tg
from core import openai_agent as core_agent

from .bot import PsychoBot
from .session_store import PsychoSessionStore

logger = logging.getLogger("psycho_bot")

REDIS_NAMESPACE = "psycho"


def _agent_factory(client: openai.OpenAI):
    """Build the ``(system_prompt, **kw) -> OpenAIAgent`` factory PsychoBot needs.

    Each task (period parse / analysis / structure / therapist) gets a fresh
    agent with its own system prompt and per-task model (``ANALYSIS_MODEL`` /
    ``REPORT_MODEL``) passed via ``**kw``.
    """

    def make(system: str, **kw: object) -> core_agent.OpenAIAgent:
        return core_agent.OpenAIAgent(client, system=system, **kw)

    return make


def build_psycho_bot(cfg: config.Config) -> tuple[PsychoBot, tg.TelegramClient]:
    """Wire the shared core into a PsychoBot from a loaded config.

    Delivery: the weekly report and the on-demand analysis are sent straight to
    the owner's Telegram chat.
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
    )
    return bot, telegram


def build_dispatcher(bot: PsychoBot) -> Dispatcher:
    """Map aiogram messages onto the PsychoBot handlers (the n8n Main Router)."""
    dp = Dispatcher()

    @dp.message(Command("start", "help"))
    async def _on_start(message: Message) -> None:
        await bot.on_text(message.chat.id, "/start")

    @dp.message(lambda m: m.voice is not None or m.audio is not None)
    async def _on_voice(message: Message) -> None:
        # A caption, when present, is the transcript; otherwise download the voice
        # OGG/Opus and transcribe it via Whisper (core.stt). the user dictates notes
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
