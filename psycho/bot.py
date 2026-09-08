"""Psycho-bot handlers: journal router, on-demand analysis, scheduled report.

Wires the shared core (``core.tg`` + ``core.state`` + ``core.agent``) into the
behaviour of the three n8n workflows this module replaces. Every model call runs
inside ``core.errors.run_resilient`` so an OpenAI/Telegram blip is logged and
optionally alerted, but never kills the long-poll process or the report loop.

The bot builds a fresh ``core.agent.Agent`` per task via an ``agent_factory`` so
each agent gets its own system prompt (router/analysis/structure/therapist) with
prompt caching applied by core. This also keeps the unification path open: the
analysis capability is exposed as a tool (see ``tools.as_analysis_tool``).

Delivery to Telegram is UNCONDITIONAL, for both ``run_analysis`` and
``run_report``. Restored 2026-09-08 after six weeks (2026-07-24 to 2026-09-08)
in which "producer mode" sent every analysis and every weekly report into an
approval-card list, ``approval:psycho``, instead of Telegram. That list's
consumer was retired the same day producer mode landed, so nothing ever read
the cards and no report reached its owner. The approval card is now an
OPTIONAL, ADDITIONAL artifact: see ``router.build_psycho_bot`` for the
``PSYCHO_EMIT_CARDS`` flag that gates it. A card-sink failure is caught and
logged; it never blocks the Telegram send.

Both send sites went dead for six weeks (see above), so this module never had
to survive a raw model reply under the bot-wide ``parse_mode="HTML"``. Now
that delivery runs every time, the model body goes through
``core.tgfmt.to_telegram_html`` and the interpolated period label through
``core.tgfmt.escape_plain`` before either reaches ``core.tg.send_text``: a
stray ``<`` or ``&`` in either would otherwise make Telegram reject the whole
message with ``can't parse entities``, the exact failure this restore exists
to fix, just arriving from the other direction.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from core.errors import run_resilient
from core.tgfmt import escape_plain, to_telegram_html

from . import tools
from .analysis import ANALYSIS_MODEL
from .report import REPORT_MODEL, build_week_tasks_section

logger = logging.getLogger(__name__)


def _menu_keyboard() -> Any:
    """Build the persistent reply keyboard with the analysis button.

    Imported lazily so the pure bot logic stays importable without aiogram in
    contexts that only need the handlers under test.
    """
    from core.tg import reply_keyboard

    return reply_keyboard([[tools.ANALYZE_BUTTON_LABEL]])

# Redis key prefix for the journal ledger (n8n: "psycho:session:"; core.state
# adds the "psycho" namespace, so the per-key prefix here is just "session:").
SESSION_PREFIX = "session:"

WELCOME = (
    "<b>🧩 Psycho journal</b>\n\n"
    "Пиши или наговаривай мысли — я их сохраняю. "
    "Когда захочешь холодный стоический разбор за период — "
    f"нажми «{tools.ANALYZE_BUTTON_LABEL}» и укажи период "
    "(например «за неделю», «май», «01.05-07.05»).\n\n"
    "Раз в неделю придёт автоматический отчёт."
)

JOURNAL_ACK = "Записал. 🧷"
ASK_PERIOD = "За какой период сделать разбор? (например «за неделю», «май», «01.05-07.05»)"
REPORT_EMPTY = "За период нет записей — отчёт пуст. 🫙"

# Redis key for the delivery heartbeat: set after a report actually reaches
# Telegram (not merely after it is built). Written through the state's generic
# single-key API, the same one the per-chat awaiting flag uses, so it never
# shows up in ``_all_sessions()``'s journal enumeration.
HEARTBEAT_KEY = "last_report_delivered"
# 30 days: several multiples of the 7-day report cadence. Letting the key
# expire if no report lands for that long is a deliberate staleness signal for
# ``--check``, not a bug to paper over with an infinite TTL.
HEARTBEAT_TTL = 30 * 24 * 3600


class _TgLike(Protocol):
    async def send_text(self, text: str, chat_id: int | None = ..., **kw: Any) -> Any: ...


class _StateLike(Protocol):
    def set_session(self, key: str, value: Any, *, ttl: int = ...) -> None: ...
    def get_session(self, key: str, default: Any = ...) -> Any: ...
    def clear_session(self, key: str) -> None: ...


# Factory signature: (system_prompt, **agent_kwargs) -> Agent
_AgentFactory = Callable[..., Any]


class PsychoBot:
    """Holds the wired core and implements the journal + analysis + report flow.

    Args:
        telegram: a ``core.tg.TelegramClient`` (or compatible).
        agent_factory: callable building a ``core.agent.Agent`` from a system
            prompt — e.g. ``lambda system, **kw: Agent(client, system=system, **kw)``.
        state: a ``core.state.RedisState`` for the session ledger + awaiting flag.
        owner_chat_id: default chat for the scheduled report and failure alerts.
        card_sink: optional ADDITIONAL sink. When given, an analysis/report is
            ALSO emitted as an approval CARD via this callable, gated behind
            ``PSYCHO_EMIT_CARDS=1`` (see ``router.build_psycho_bot``). It is
            never a substitute for the Telegram send, which is unconditional.
            Signature: ``sink(analysis: str, *, period_label: str,
            source_session=None)``. A sink failure is caught and logged, and
            does not affect delivery. ``None`` (default) skips the card.
    """

    def __init__(
        self,
        telegram: _TgLike,
        agent_factory: _AgentFactory,
        state: _StateLike,
        owner_chat_id: int,
        card_sink: Callable[..., Any] | None = None,
    ) -> None:
        self._tg = telegram
        self._agent_factory = agent_factory
        self._state = state
        self._owner = owner_chat_id
        self._card_sink = card_sink

    # ---- ledger helpers -------------------------------------------------

    def _awaiting_key(self, chat_id: int) -> str:
        return f"awaiting:{chat_id}"

    def _store_session(self, transcript: str, *, source: str, duration_sec: int = 0) -> bool:
        """Persist one journal entry. Returns False for empty transcripts."""
        if not (transcript or "").strip():
            return False
        key, value = tools.session_record(transcript, source=source, duration_sec=duration_sec)
        self._state.set_session(key, value, ttl=tools.SESSION_TTL)  # 7-day retention (n8n parity)
        return True

    def _all_sessions(self) -> list[dict[str, Any]]:
        """Enumerate stored journal entries.

        Prefers a ``list_sessions(prefix)`` helper (provided by the redis-backed
        store / test fake); returns ``[]`` if the store can't enumerate.
        """
        lister = getattr(self._state, "list_sessions", None)
        if callable(lister):
            return [s for s in lister(SESSION_PREFIX) if isinstance(s, dict) and s.get("transcript")]
        return []

    # ---- router handlers ------------------------------------------------

    async def on_text(self, chat_id: int, text: str) -> None:
        """Route an inbound text message (n8n Main Router switch)."""
        awaiting = self._state.get_session(self._awaiting_key(chat_id), None)
        route = tools.classify(text, awaiting=awaiting)

        if route == tools.ROUTE_START:
            await self._tg.send_text(WELCOME, chat_id=chat_id, reply_markup=_menu_keyboard())
        elif route == tools.ROUTE_ANALYZE:
            self._state.set_session(
                self._awaiting_key(chat_id), tools.AWAITING_VALUE, ttl=tools.AWAITING_TTL
            )
            await self._tg.send_text(ASK_PERIOD, chat_id=chat_id)
        elif route == tools.ROUTE_AWAITING_PERIOD:
            self._state.clear_session(self._awaiting_key(chat_id))
            await self.run_analysis(chat_id, text)
        else:  # ROUTE_JOURNAL
            if self._store_session(text, source="text"):
                await self._tg.send_text(JOURNAL_ACK, chat_id=chat_id)

    async def on_voice_transcript(self, chat_id: int, transcript: str, *, duration_sec: int = 0) -> None:
        """Log an already-transcribed voice note as a journal entry.

        Voice STT belongs to the n8n edge / ``core.stt``; this brain receives the
        text. A voice note is always a journal entry (the n8n router pushes voice
        straight to the ledger).
        """
        if self._store_session(transcript, source="voice", duration_sec=duration_sec):
            await self._tg.send_text(JOURNAL_ACK, chat_id=chat_id)

    # ---- on-demand analysis (OnDemand Analysis workflow) ----------------

    async def run_analysis(self, chat_id: int, raw_period: str) -> None:
        """Parse the period, assemble in-window sessions, run the stoic analysis.

        Wrapped in ``run_resilient`` so a model failure pings the owner and is
        logged, but never crashes the long-poll process.
        """

        async def _work() -> None:
            parser = self._agent_factory(
                tools.PERIOD_PARSER_SYSTEM, max_tokens=1500, model=ANALYSIS_MODEL
            )
            period = tools.parse_period(parser, raw_period)

            sessions = self._sessions_in_window(period)
            context = self._build_context(period, sessions)

            # Caps sized for reasoning models: the visible answer plus GPT-5
            # reasoning tokens are drawn from one budget, so keep generous
            # headroom or the analysis can come back empty.
            analyst = self._agent_factory(
                tools.ANALYSIS_SYSTEM, max_tokens=3000, model=ANALYSIS_MODEL
            )
            result = tools.analyze_period(analyst, context)
            if self._card_sink is not None:
                # Additional artifact only: a sink failure must never block
                # the Telegram send below.
                try:
                    self._card_sink(result, period_label=period["label"])
                except Exception:  # noqa: BLE001 -- card sink is best-effort
                    logger.exception(
                        "psycho analysis card sink failed; Telegram delivery continues"
                    )
            header = f"🧩 <b>Разбор за {escape_plain(period['label'])}</b>\n\n"
            await self._tg.send_text(header + to_telegram_html(result), chat_id=chat_id)

        await run_resilient(_work, alerter=self._tg, label="psycho.analysis")

    def _sessions_in_window(self, period: dict[str, str]) -> list[dict[str, Any]]:
        """Filter the ledger to sessions whose timestamp falls in [from, to].

        Period bounds are DD.MM.YYYY; compared against the ms timestamps in the
        ledger. A session with no timestamp is kept (defensive, as in n8n).
        """
        from_num = _date_to_num(period["date_from"])
        to_num = _date_to_num(period["date_to"])
        if not from_num or not to_num:
            return self._all_sessions()
        kept = []
        for s in self._all_sessions():
            ts = s.get("timestamp")
            if not ts:
                kept.append(s)
                continue
            day = _ms_to_num(ts)
            if from_num <= day <= to_num:
                kept.append(s)
        return kept

    def _build_context(self, period: dict[str, str], sessions: list[dict[str, Any]]) -> str:
        """Assemble the analysis context string from the in-window journal entries."""
        import json

        if not sessions:
            return (
                f"Период: {period['label']} ({period['date_from']}–{period['date_to']}).\n"
                "Записей за период нет. Скажи об этом прямо и проанализируй отсутствие данных."
            )
        body = json.dumps(sessions, ensure_ascii=False, indent=2)
        return (
            f"Период: {period['label']} ({period['date_from']}–{period['date_to']}).\n"
            f"Записи дневника:\n{body}"
        )

    # ---- scheduled report (Report Workflow) -----------------------------

    async def run_report(self, chat_id: int | None = None) -> None:
        """Build and send the periodic report (driven by ``core.scheduler``).

        Reads the ledger, structures it, runs the therapist analysis into a
        ``{report_md, summary}`` dict, and sends it. Resilient: a failed cycle is
        logged/alerted and the schedule keeps running.
        """
        target = chat_id if chat_id is not None else self._owner

        async def _work() -> None:
            sessions = self._all_sessions()
            if not sessions:
                # An empty period is still a real delivery: Bogdan must see
                # that the week produced nothing, not silence.
                await self._tg.send_text(REPORT_EMPTY, chat_id=target)
                self._mark_report_delivered()
                return

            label = tools.period_label_from_sessions(sessions) or "период"
            structurer = self._agent_factory(
                tools.STRUCTURE_SYSTEM, max_tokens=4000, model=REPORT_MODEL
            )
            # structuring is best-effort context-cleaning; result isn't required to send
            tools.structure_sessions(structurer, sessions, label)

            # Enrich with the past week's Notion tasks (never raises: degrades to
            # an "unavailable" note on a Notion failure).
            now = datetime.now(UTC)
            tasks_section = build_week_tasks_section(
                (now - timedelta(days=7)).strftime("%Y-%m-%d"),
                now.strftime("%Y-%m-%d"),
            )

            therapist = self._agent_factory(
                tools.REPORT_THERAPIST_SYSTEM, max_tokens=4000, model=REPORT_MODEL
            )
            report = tools.therapist_report(
                therapist, sessions, label, tasks_section=tasks_section
            )

            if self._card_sink is not None:
                # Additional artifact only: a sink failure must never block
                # the Telegram send below.
                try:
                    self._card_sink(report["report_md"], period_label=label)
                except Exception:  # noqa: BLE001 -- card sink is best-effort
                    logger.exception(
                        "psycho report card sink failed; Telegram delivery continues"
                    )
            header = f"🧩 <b>Еженедельный разбор: {escape_plain(label)}</b>\n\n"
            await self._tg.send_text(
                header + to_telegram_html(report["report_md"]), chat_id=target
            )
            self._mark_report_delivered()

        await run_resilient(_work, alerter=self._tg, label="psycho.report")

    def _mark_report_delivered(self) -> None:
        """Record that a report just reached Telegram (the empty-ledger notice counts).

        Written after the send, never before: the heartbeat means "delivered",
        not "attempted". Uses the state's generic single-key API (the same one
        the per-chat awaiting flag uses), keyed by ``HEARTBEAT_KEY`` with
        ``HEARTBEAT_TTL``. ``RedisState.set_session`` already degrades to a
        logged warning on a redis failure and never raises, so a heartbeat
        write can never take the report down with it.
        """
        self._state.set_session(HEARTBEAT_KEY, datetime.now(UTC).isoformat(), ttl=HEARTBEAT_TTL)


# ---- date helpers (n8n "Filter Files by Period" numeric comparison) -----


def _date_to_num(ddmmyyyy: str) -> int:
    """DD.MM.YYYY -> YYYYMMDD int (0 if unparseable)."""
    parts = (ddmmyyyy or "").split(".")
    if len(parts) != 3:
        return 0
    d, m, y = parts
    try:
        return int(f"{int(y):04d}{int(m):02d}{int(d):02d}")
    except ValueError:
        return 0


def _ms_to_num(ms: int) -> int:
    """Epoch-ms -> YYYYMMDD int (UTC), for window comparison."""
    dt = datetime.fromtimestamp(ms / 1000, tz=UTC)
    return int(dt.strftime("%Y%m%d"))
