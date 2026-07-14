"""Unit tests for psycho.tools — routing, session records, period parsing,
analysis/report capabilities. the LLM client is faked; no network.

Mirrors the n8n trio (Bot Router , OnDemand Analysis
, Report Workflow ): the routing rules, the
psycho:session ledger shape, the period parser, and the structured report.
"""

from __future__ import annotations

import json
from typing import Any

from psycho import tools

# ---- fakes --------------------------------------------------------------


class _FakeAgent:
    """Stands in for core.agent.Agent. run() returns a fixed string;
    structured_output() returns a fixed dict; both record the prompt."""

    def __init__(self, *, run_reply: str = "", structured: dict[str, Any] | None = None) -> None:
        self.run_reply = run_reply
        self.structured = structured or {}
        self.run_prompts: list[str] = []
        self.structured_calls: list[dict[str, Any]] = []

    def run(self, prompt: str) -> str:
        self.run_prompts.append(prompt)
        return self.run_reply

    def structured_output(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        self.structured_calls.append({"prompt": prompt, "schema": schema})
        return self.structured


# ---- routing ------------------------------------------------------------


def test_classify_start():
    assert tools.classify("/start", awaiting=None) == tools.ROUTE_START


def test_classify_analyze_button():
    # The n8n switch matches the exact Russian button label.
    assert tools.classify(tools.ANALYZE_BUTTON_LABEL, awaiting=None) == tools.ROUTE_ANALYZE


def test_classify_awaiting_period_takes_priority_over_journal():
    # When awaiting == "analyze", a plain text message is the period, not a journal entry.
    assert tools.classify("за неделю", awaiting="analyze") == tools.ROUTE_AWAITING_PERIOD


def test_classify_plain_text_is_journal():
    assert tools.classify("сегодня было тяжело", awaiting=None) == tools.ROUTE_JOURNAL


def test_classify_empty_is_journal_when_not_awaiting():
    # Defensive: empty/whitespace falls through to journal (handler decides to ignore).
    assert tools.classify("   ", awaiting=None) == tools.ROUTE_JOURNAL


# ---- session ledger -----------------------------------------------------


def test_session_record_shape():
    key, value = tools.session_record("привет мир", source="text")
    assert key.startswith("session:")
    assert value["transcript"] == "привет мир"
    assert value["source"] == "text"
    assert isinstance(value["timestamp"], int)
    assert "session_id" in value


def test_session_record_voice_carries_duration():
    _, value = tools.session_record("трансткрипт", source="voice", duration_sec=12)
    assert value["source"] == "voice"
    assert value["duration_sec"] == 12


def test_session_record_unique_keys_by_timestamp(monkeypatch):
    ts = iter([1000, 2000])
    monkeypatch.setattr(tools.time, "time", lambda: next(ts) / 1000.0)
    k1, _ = tools.session_record("a", source="text")
    k2, _ = tools.session_record("b", source="text")
    assert k1 != k2


# ---- period parsing -----------------------------------------------------


def test_parse_period_structured_happy_path():
    agent = _FakeAgent(structured={"date_from": "01.05.2026", "date_to": "07.05.2026", "label": "май"})
    out = tools.parse_period(agent, "первая неделя мая")
    assert out["label"] == "май"
    assert out["date_from"] == "01.05.2026"
    # the parser must request the PERIOD schema
    assert agent.structured_calls[0]["schema"] == tools.PERIOD_SCHEMA


def test_parse_period_accepts_iso_dates():
    # the model sometimes returns ISO; normalize to DD.MM.YYYY.
    agent = _FakeAgent(structured={"date_from": "2026-05-01", "date_to": "2026-05-07", "label": "x"})
    out = tools.parse_period(agent, "x")
    assert out["date_from"] == "01.05.2026"
    assert out["date_to"] == "07.05.2026"


def test_parse_period_fallback_on_bad_output():
    # Missing fields -> fall back to a last-7-days window, never raise.
    agent = _FakeAgent(structured={})
    out = tools.parse_period(agent, "whatever")
    assert "fallback" in out["label"]
    assert out["date_from"] and out["date_to"]


# ---- analysis (ondemand) ------------------------------------------------


def test_analyze_period_returns_agent_text():
    agent = _FakeAgent(run_reply="Холодный анализ за период.")
    out = tools.analyze_period(agent, "контекст сессий")
    assert out == "Холодный анализ за период."
    assert "контекст сессий" in agent.run_prompts[0]


def test_analyze_period_fallback_when_empty():
    agent = _FakeAgent(run_reply="   ")
    out = tools.analyze_period(agent, "ctx")
    assert out  # never empty back to the user
    assert "не удал" in out.lower() or "no data" in out.lower()


# ---- report -------------------------------------------------------------


def test_structure_sessions_returns_markdown():
    agent = _FakeAgent(run_reply="## неделя\n- мысль")
    sessions = [{"transcript": "мысль", "timestamp": 1}]
    out = tools.structure_sessions(agent, sessions, "01.05-07.05")
    assert "неделя" in out
    # period label + sessions json must reach the model
    assert "01.05-07.05" in agent.run_prompts[0]


def test_therapist_report_returns_schema_dict():
    agent = _FakeAgent(structured={"report_md": "# отчёт", "summary": "две строки"})
    sessions = [{"transcript": "мысль", "timestamp": 1}]
    out = tools.therapist_report(agent, sessions, "01.05-07.05")
    assert out["report_md"] == "# отчёт"
    assert out["summary"] == "две строки"
    assert agent.structured_calls[0]["schema"] == tools.REPORT_SCHEMA


def test_therapist_report_fallback_on_missing_fields():
    agent = _FakeAgent(structured={})
    out = tools.therapist_report(agent, [{"transcript": "x", "timestamp": 1}], "x")
    assert "report_md" in out and "summary" in out


def test_report_schema_round_trips_a_conforming_dict():
    sample = {"report_md": "# r", "summary": "s"}
    # the schema's required keys must match what we produce
    required = set(tools.REPORT_SCHEMA["required"])
    assert required <= set(sample)
    json.dumps(sample)  # serializable


# ---- structured-output schema shape (regression: double-wrap -> HTTP 400) ----


def test_schemas_are_bare_not_double_wrapped():
    # core.agent.Agent.structured_output wraps the schema into
    # output_config={"format": {"type": "json_schema", "schema": schema}} itself.
    # If these schemas pre-wrap with their own "json_schema"/"schema" envelope,
    # the payload is double-wrapped and a strict JSON-schema API rejects it with HTTP 400
    # (silently swallowed by the fallback). They must be bare object schemas.
    for schema in (tools.PERIOD_SCHEMA, tools.REPORT_SCHEMA):
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "json_schema" not in schema  # no outer envelope key
        assert "schema" not in schema  # not nested under a "schema" wrapper


# ---- system prompts -----------------------------------------------------


def test_system_prompts_present_and_stoic():
    # The cold-psychologist + Stoic frame is the load-bearing content lifted from n8n.
    assert "cold psychologist" in tools.ANALYSIS_SYSTEM.lower()
    assert "dichotomy-of-control" in tools.ANALYSIS_SYSTEM.lower()
    assert "cold psychologist" in tools.REPORT_THERAPIST_SYSTEM.lower()
    assert "valid json" in tools.REPORT_THERAPIST_SYSTEM.lower()


# ---- unified-agent tool surface ----------------------------------------


def test_period_label_helper():
    sessions = [
        {"transcript": "a", "timestamp": 1714521600000},  # 2024-05-01
        {"transcript": "b", "timestamp": 1714694400000},  # 2024-05-03
    ]
    label = tools.period_label_from_sessions(sessions)
    assert "." in label  # DD.MM.YYYY-DD.MM.YYYY shape


def test_period_label_empty():
    assert tools.period_label_from_sessions([]) == ""
