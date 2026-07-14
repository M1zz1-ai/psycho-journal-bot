"""On-demand analysis surface.

The analysis logic itself lives in ``tools`` (period parse + cold-stoic analysis)
and is orchestrated by ``PsychoBot.run_analysis`` (period -> in-window sessions ->
context -> analysis -> Telegram). This module is the thin public facade: it
re-exports the building blocks and exposes the analysis as an agent-compatible
tool (:func:`as_analysis_tool`), so the capability can be registered on a larger
agent later.

Analysis operates purely on the redis session ledger (the journal entries stored
over the last 7 days); there is no external document store.
"""

from __future__ import annotations

import os
from typing import Any

from . import tools

# On-demand analysis runs on OpenAI's cheap/fast mini tier — the right tier for
# the button/command path (period parse + stoic analysis). The bot's LLM brain
# moved from Anthropic to OpenAI when the direct Anthropic key ran out of credits
#; ``OPENAI_API_KEY`` already powers Whisper STT. ``gpt-5.4-mini`` is
# the newest mini model, verified against the live /v1/models list. Override via
# ``PSYCHO_ANALYSIS_MODEL``.
ANALYSIS_MODEL = os.getenv("PSYCHO_ANALYSIS_MODEL", "gpt-5.4-mini")

# Public re-exports (stable import surface for callers / the unified agent).
ANALYSIS_SYSTEM = tools.ANALYSIS_SYSTEM
PERIOD_PARSER_SYSTEM = tools.PERIOD_PARSER_SYSTEM
PERIOD_SCHEMA = tools.PERIOD_SCHEMA
parse_period = tools.parse_period
analyze_period = tools.analyze_period


def as_analysis_tool(agent: Any) -> Any:
    """Return the on-demand analysis as a ``core.agent``-compatible callable.

    Thin pass-through to :func:`tools.as_analysis_tool` — kept here so the
    unification entry point lives next to the analysis facade. The returned
    callable takes an assembled journal context string and returns the analysis.
    """
    return tools.as_analysis_tool(agent)
