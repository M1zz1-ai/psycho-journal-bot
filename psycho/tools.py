"""Psycho-pipeline capabilities: routing, the session ledger, period parsing,
on-demand analysis, and the scheduled structured report.

Each capability is a small, independently testable callable (also registerable as
an agent tool). This module holds the routing rules, the redis session-ledger
shape, the free-text period parser, the cold-stoic analysis, and the structured
weekly-report schema.

Model note: a cheap/fast OpenAI "mini" model runs the on-demand path (period parse
+ analysis) and OpenAI's flagship chat model runs the weekly report. The concrete
model ids live next to their surface — ``analysis.ANALYSIS_MODEL`` (mini) and
``report.REPORT_MODEL`` (flagship) — and are env-overridable. ``bot`` passes them
into the agent factory per task. All LLM calls go through ``core.openai_agent``;
voice STT uses OpenAI Whisper (``core.stt``), so ``OPENAI_API_KEY`` powers the
whole bot.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# ---- routing (lifted from the n8n "Main Router" switch) ----------------

ROUTE_START = "start"
ROUTE_ANALYZE = "analyze"
ROUTE_AWAITING_PERIOD = "awaiting_period"
ROUTE_JOURNAL = "journal"

# Exact label the n8n switch matches for the "request analysis" button.
ANALYZE_BUTTON_LABEL = "Получить анализ 🧩"
# Redis flag value the router sets while waiting for a period reply (5-min TTL).
AWAITING_VALUE = "analyze"
AWAITING_TTL = 300  # seconds, mirrors the n8n redis SET expiry

# Journal entries live for 7 days in redis (n8n "Redis Push" ttl: 604800). After
# that the report/analysis no longer sees them — same retention as the n8n flow.
SESSION_TTL = 7 * 24 * 3600  # 604800 seconds


def classify(text: str, *, awaiting: str | None) -> str:
    """Route an incoming text message, mirroring the n8n Main Router switch.

    Order matters (first match wins), exactly as in the workflow:
      1. ``/start``            -> show the menu
      2. analyze button label  -> ask for a period
      3. awaiting == "analyze" -> the text IS the period
      4. anything else         -> a journal entry to log

    ``awaiting`` is the per-chat redis flag (``None`` when not set / expired).
    """
    stripped = (text or "").strip()
    if stripped == "/start":
        return ROUTE_START
    if stripped == ANALYZE_BUTTON_LABEL:
        return ROUTE_ANALYZE
    if awaiting == AWAITING_VALUE:
        return ROUTE_AWAITING_PERIOD
    return ROUTE_JOURNAL


# ---- session ledger (lifted from "Build Text/Voice Payload") -----------


def session_record(transcript: str, *, source: str, duration_sec: int = 0) -> tuple[str, dict[str, Any]]:
    """Build a (redis_key, value) pair for one journal entry.

    Key is ``session:<ms-timestamp>`` (the n8n ``psycho:session:<ts>`` key, minus
    the namespace which ``core.state`` adds). Value carries the transcript,
    source ("text"/"voice"), an ISO-ish ``session_id``, and UTC timestamp.
    """
    now = datetime.now(UTC)
    timestamp = int(time.time() * 1000)
    session_id = now.strftime("%Y-%m-%d-%H%M")
    value = {
        "session_id": session_id,
        "timestamp": timestamp,
        "source": source,
        "duration_sec": duration_sec,
        "transcript": (transcript or "").strip(),
    }
    return f"session:{timestamp}", value


# ---- period parsing (lifted from "Period Parser" + "Parse Period") -----

PERIOD_PARSER_SYSTEM = (
    "Parse a Russian or English period description into specific dates. "
    "Output ONLY valid JSON, no wrapper:\n"
    '{"date_from": "DD.MM.YYYY", "date_to": "DD.MM.YYYY", '
    '"label": "human readable period name"}\n\n'
    "Examples:\n"
    '- "вчера" -> yesterday\n'
    '- "неделя" / "за неделю" -> last 7 days from today\n'
    '- "месяц" / "за месяц" -> last 30 days\n'
    '- "01.05-07.05" -> that range in current year\n'
    '- "май" -> entire May of current year\n'
    '- "last 3 days" -> last 3 days'
)

# Bare object schema: core.agent.Agent.structured_output wraps it into
# output_config={"format": {"type": "json_schema", "schema": ...}} itself.
# Pre-wrapping here would double-wrap and the API rejects it with HTTP 400.
PERIOD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "date_from": {"type": "string"},
        "date_to": {"type": "string"},
        "label": {"type": "string"},
    },
    "required": ["date_from", "date_to", "label"],
    "additionalProperties": False,
}


class _AgentLike(Protocol):
    def run(self, prompt: str) -> str: ...
    def structured_output(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]: ...


def _normalize_date(raw: str) -> str:
    """Accept DD.MM.YYYY or YYYY-MM-DD; return DD.MM.YYYY (n8n "Parse Period")."""
    if not raw or not isinstance(raw, str):
        return ""
    raw = raw.strip()
    if len(raw) == 10 and raw[4] == "-":  # ISO
        y, m, d = raw.split("-")
        return f"{d}.{m}.{y}"
    return raw


def parse_period(agent: _AgentLike, raw_period: str) -> dict[str, str]:
    """Parse a free-text period into ``{date_from, date_to, label}``.

    Uses ``agent.structured_output`` (the canonical structured-outputs path).
    On any malformed/empty result, falls back to a last-7-days window so the
    flow never dies on a bad parse — same defensive posture as the n8n node.
    """
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    prompt = f"Parse this period request into dates. Today is {today}\nRequest: {raw_period}"
    try:
        parsed = agent.structured_output(prompt, PERIOD_SCHEMA)
    except Exception:  # structured parse / network failure -> fall back
        logger.exception("period parse failed; using fallback window")
        parsed = {}

    date_from = _normalize_date(str(parsed.get("date_from", "")))
    date_to = _normalize_date(str(parsed.get("date_to", "")))
    label = str(parsed.get("label", "")).strip()

    if not date_from or not date_to:
        now = datetime.now(UTC)
        week_ago = datetime.fromtimestamp(now.timestamp() - 7 * 24 * 3600, tz=UTC)
        return {
            "date_from": week_ago.strftime("%d.%m.%Y"),
            "date_to": now.strftime("%d.%m.%Y"),
            "label": "last 7 days (fallback)",
        }
    return {"date_from": date_from, "date_to": date_to, "label": label or f"{date_from}-{date_to}"}


# ---- on-demand analysis (lifted from "Analysis Agent") -----------------

ANALYSIS_SYSTEM = (
    "You are a cold psychologist-analyst. No rose-colored glasses. Analyze the "
    "psychological state for the requested period based on the provided records. "
    "Rules:\n"
    "- Be specific about what the data shows\n"
    "- Note patterns, contradictions, changes\n"
    "- If data is sparse, say so and analyze what IS there\n"
    "- Output in Russian markdown, 200-500 words\n"
    "- Do NOT wrap in code blocks\n\n"
    "Stoic analytical frame (apply to every analysis):\n"
    "1. Dichotomy-of-control audit: classify each reported stressor as within or "
    "outside the person's control. Flag where emotional energy is misallocated to "
    "the uncontrollable.\n"
    "2. Stoic reframes: where distorted thinking appears, offer reframes grounded "
    "in Seneca, Epictetus, or Marcus Aurelius — cite the actual principle or text "
    '(e.g. "Epictetus, Enchiridion §1"), never generic platitudes.\n'
    "3. Premeditatio malorum: if anxiety or catastrophizing patterns appear, apply "
    "this practice explicitly — name the feared outcome, assess its actual "
    "probability and recoverability.\n"
    "4. Virtue-ethics lens: where stated values contradict observed behaviour, name "
    "the relevant virtue (courage, temperance, justice, or wisdom) and describe the "
    "gap precisely.\n"
    "5. One concrete Stoic exercise: end each analysis with exactly one actionable "
    "exercise suited to the dominant pattern found.\n\n"
    "When citing philosopher texts with quotation marks in Russian markdown, use "
    "«» guillemets instead of double quotes to avoid breaking JSON encoding."
)

# Fallback shown to the user if the model returns nothing (n8n "Return Output").
ANALYSIS_FALLBACK = "Анализ не удался."


def analyze_period(agent: _AgentLike, context: str) -> str:
    """Run the cold-stoic analysis over an assembled context string.

    The agent's system prompt is expected to be :data:`ANALYSIS_SYSTEM`.
    Returns Russian markdown; never returns empty (falls back to a notice).
    """
    out = agent.run(context).strip()
    return out or ANALYSIS_FALLBACK


# ---- report: structure pass (lifted from "Structure Agent") ------------

STRUCTURE_SYSTEM = (
    "You are a note-structuring assistant. Convert raw voice/text thought dumps "
    "into clean, readable markdown journal entries. Group by logical themes. Keep "
    "ALL content, don't omit anything. Output ONLY the markdown content, no wrapper."
)


def structure_sessions(agent: _AgentLike, sessions: list[dict[str, Any]], period_label: str) -> str:
    """Structure raw session dumps into clean markdown (report step 1)."""
    prompt = (
        f"Structure these raw thought sessions into clean markdown. "
        f"Period: {period_label}\n\nSessions:\n{json.dumps(sessions, ensure_ascii=False, indent=2)}"
    )
    return agent.run(prompt).strip()


# ---- report: therapist pass (lifted from "Therapist Agent") ------------

REPORT_THERAPIST_SYSTEM = (
    "You are a cold psychologist-analyst. No rose-colored glasses. Direct, "
    "clinical, precise. Rules:\n"
    "- Compare patterns, note contradictions between what person says vs implied "
    "behaviour.\n"
    "- Forbidden: generic praise, unsupported advice, emotional support for its own "
    "sake.\n"
    "- If data is insufficient for a conclusion, say so.\n\n"
    "Stoic analytical frame (apply to every report):\n"
    "1. Dichotomy-of-control audit: classify each reported stressor as within or "
    "outside the person's control. Flag where emotional energy is misallocated to "
    "the uncontrollable.\n"
    "2. Stoic reframes: where distorted thinking appears, offer reframes grounded "
    "in Seneca, Epictetus, or Marcus Aurelius — cite the actual principle or text "
    '(e.g. "Epictetus, Enchiridion §1"), never generic platitudes.\n'
    "3. Premeditatio malorum: if anxiety or catastrophizing patterns appear, apply "
    "this practice explicitly — name the feared outcome, assess its actual "
    "probability and recoverability.\n"
    "4. Virtue-ethics lens: where stated values contradict observed behaviour, name "
    "the relevant virtue (courage, temperance, justice, or wisdom) and describe the "
    "gap precisely.\n"
    "5. One concrete Stoic exercise: end each report with exactly one actionable "
    "exercise suited to the dominant pattern found (e.g. morning reflection, evening "
    "review, voluntary discomfort, journaling prompt).\n\n"
    "When a «Неделя в задачах» section is present, ground the analysis in those "
    "actual tasks — name the gap between stated priorities (what got done vs left "
    "not started) and the logged emotional state. Never invent tasks that are not "
    "listed; if the section says data is unavailable, ignore it and do not mention "
    "tasks.\n\n"
    "Output a Russian markdown report (300-600 words) plus a 2-sentence Russian "
    "summary. Return ONLY valid JSON matching the requested schema."
)

# Bare object schema (same reason as PERIOD_SCHEMA): structured_output wraps it.
REPORT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "report_md": {"type": "string"},
        "summary": {"type": "string"},
    },
    "required": ["report_md", "summary"],
    "additionalProperties": False,
}


def therapist_report(
    agent: _AgentLike,
    sessions: list[dict[str, Any]],
    period_label: str,
    *,
    tasks_section: str = "",
) -> dict[str, str]:
    """Run the therapist analysis -> ``{report_md, summary}`` (report step 2).

    Uses ``agent.structured_output`` with :data:`REPORT_SCHEMA`. On a malformed
    result, falls back to a minimal dict so the report still sends. An optional
    ``tasks_section`` (the "Неделя в задачах" summary from ``report``) is injected
    into the prompt context so the report grounds itself in the week's tasks.
    """
    prompt = f"Analyze the psychological state for period: {period_label}\n\n"
    if tasks_section:
        prompt += f"{tasks_section}\n\n"
    prompt += f"Raw sessions:\n{json.dumps(sessions, ensure_ascii=False, indent=2)}"
    try:
        parsed = agent.structured_output(prompt, REPORT_SCHEMA)
    except Exception:
        logger.exception("therapist report failed; using fallback")
        parsed = {}
    report_md = str(parsed.get("report_md", "")).strip()
    summary = str(parsed.get("summary", "")).strip()
    if not report_md:
        report_md = "Отчёт не удалось сформировать за этот период."
    return {"report_md": report_md, "summary": summary}


# ---- helpers ------------------------------------------------------------


def period_label_from_sessions(sessions: list[dict[str, Any]]) -> str:
    """Build a ``DD.MM.YYYY-DD.MM.YYYY`` label from session timestamps (n8n
    "Parse Sessions"). Empty string when there are no sessions."""
    timestamps = [s.get("timestamp", 0) for s in sessions if s.get("timestamp")]
    if not timestamps:
        return ""
    fmt = lambda ms: datetime.fromtimestamp(ms / 1000, tz=UTC).strftime("%d.%m.%Y")  # noqa: E731
    return f"{fmt(min(timestamps))}-{fmt(max(timestamps))}"


def as_analysis_tool(agent: _AgentLike) -> Any:
    """Expose on-demand analysis as a core.agent-compatible tool callable.

    Keeps the unification path open: a future unified Agent registers this so it
    can route "analyze my psycho journal for <period>" to this pipeline. The
    callable is synchronous (the Agent loop calls tools synchronously).
    """

    def analyze_psycho_period(context: str) -> str:
        """Analyze psychological state for a period from assembled journal context."""
        return analyze_period(agent, context)

    return analyze_psycho_period
