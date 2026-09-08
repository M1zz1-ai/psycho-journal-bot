"""Psycho pipeline: Telegram journal router + on-demand stoic analysis + weekly report.

A voice-journaling Telegram bot with an AI "cold-stoic therapist". Three surfaces,
all built on the shared ``core``:

  * Router   — routes each message: /start, the analysis button, a period reply,
    or a plain journal entry (text or transcribed voice) pushed to a redis ledger.
  * Analysis — on-demand, button-triggered: parse a free-text period, gather the
    in-window journal entries, run the cold-stoic analysis.
  * Report   — a weekly cold-stoic report over the last 7 days of entries,
    enriched with the past week's Notion tasks.

Delivery to Telegram is the primary, unconditional path (see ``bot``). An
analysis/report can ALSO be mirrored as an approval card for a separate
review queue, gated behind an opt-in flag (see ``router``); that sink is an
optional, additional artifact, not the delivery path.
"""

from .bot import PsychoBot
from .tools import (
    ANALYSIS_SYSTEM,
    ANALYZE_BUTTON_LABEL,
    PERIOD_SCHEMA,
    REPORT_SCHEMA,
    REPORT_THERAPIST_SYSTEM,
    ROUTE_ANALYZE,
    ROUTE_AWAITING_PERIOD,
    ROUTE_JOURNAL,
    ROUTE_START,
    STRUCTURE_SYSTEM,
    analyze_period,
    as_analysis_tool,
    classify,
    parse_period,
    period_label_from_sessions,
    session_record,
    structure_sessions,
    therapist_report,
)

__all__ = [
    "ANALYSIS_SYSTEM",
    "ANALYZE_BUTTON_LABEL",
    "PERIOD_SCHEMA",
    "PsychoBot",
    "REPORT_SCHEMA",
    "REPORT_THERAPIST_SYSTEM",
    "ROUTE_ANALYZE",
    "ROUTE_AWAITING_PERIOD",
    "ROUTE_JOURNAL",
    "ROUTE_START",
    "STRUCTURE_SYSTEM",
    "analyze_period",
    "as_analysis_tool",
    "classify",
    "parse_period",
    "period_label_from_sessions",
    "session_record",
    "structure_sessions",
    "therapist_report",
]
