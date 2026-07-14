"""Unit tests for psycho.report — weekly tasks enrichment + prompt injection.

Notion (core.notion.list_tasks) is faked; no notion-cli subprocess, no network.
Covers the pure formatting (with tasks / empty week / notion-error fallback),
the impure fetch's graceful degradation, and that the therapist report prompt
and the end-to-end run_report both carry the "Неделя в задачах" section.
"""

from __future__ import annotations

from typing import Any

import pytest

from core.errors import NotionError
from psycho import bot as psycho_bot
from psycho import report, tools

# ---- pure formatting ----------------------------------------------------


def test_group_week_tasks_buckets() -> None:
    tasks = [
        {"title": "A", "status": "Done"},
        {"title": "B", "status": "In progress"},
        {"title": "C", "status": "Not started"},
        {"title": "D"},  # missing status -> "не начато"
    ]
    groups = report.group_week_tasks(tasks)
    assert groups["сделано"] == ["A"]
    assert groups["в работе"] == ["B"]
    assert groups["не начато"] == ["C", "D"]


def test_format_week_tasks_with_tasks() -> None:
    section = report.format_week_tasks(
        [{"title": "A", "status": "Done"}, {"title": "B", "status": "In progress"}]
    )
    assert report.WEEK_TASKS_HEADER in section
    assert "сделано (1): A" in section
    assert "в работе (1): B" in section


def test_format_week_tasks_empty_week() -> None:
    section = report.format_week_tasks([])
    assert report.WEEK_TASKS_HEADER in section
    assert "нет задач" in section
    # no fabricated buckets when the week is empty
    assert "сделано" not in section


def test_format_week_tasks_none_is_unavailable() -> None:
    section = report.format_week_tasks(None)
    assert report.TASKS_UNAVAILABLE_NOTE in section


# ---- impure fetch (graceful degradation) --------------------------------


def test_fetch_week_tasks_degrades_on_notion_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(**kw: Any) -> Any:
        raise NotionError("notion-cli down")

    monkeypatch.setattr(report.notion, "list_tasks", _boom)
    assert report.fetch_week_tasks("2026-07-07", "2026-07-14") is None


def test_build_week_tasks_section_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        report.notion, "list_tasks", lambda **kw: [{"title": "A", "status": "Done"}]
    )
    section = report.build_week_tasks_section("2026-07-07", "2026-07-14")
    assert "сделано (1): A" in section


def test_build_week_tasks_section_notion_error_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(**kw: Any) -> Any:
        raise NotionError("down")

    monkeypatch.setattr(report.notion, "list_tasks", _boom)
    section = report.build_week_tasks_section("2026-07-07", "2026-07-14")
    assert report.TASKS_UNAVAILABLE_NOTE in section


# ---- prompt injection ---------------------------------------------------


class _CapAgent:
    """Captures the prompt passed to structured_output."""

    def __init__(self) -> None:
        self.prompt: str | None = None

    def run(self, prompt: str) -> str:
        return ""

    def structured_output(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        self.prompt = prompt
        return {"report_md": "# r", "summary": "s"}


def test_therapist_report_prompt_includes_tasks_section() -> None:
    agent = _CapAgent()
    tools.therapist_report(
        agent,
        [{"transcript": "мысль", "timestamp": 1}],
        "неделя",
        tasks_section="Неделя в задачах:\n- сделано (1): A",
    )
    assert agent.prompt is not None
    assert "Неделя в задачах" in agent.prompt
    assert "сделано (1): A" in agent.prompt


def test_therapist_report_prompt_omits_section_when_absent() -> None:
    agent = _CapAgent()
    tools.therapist_report(agent, [{"transcript": "x", "timestamp": 1}], "неделя")
    assert agent.prompt is not None
    assert "Неделя в задачах" not in agent.prompt


# ---- end-to-end run_report injects the section --------------------------


class _FakeTg:
    def __init__(self) -> None:
        self.texts: list[dict[str, Any]] = []

    async def send_text(self, text: str, chat_id: int | None = None, **kw: Any) -> list[Any]:
        self.texts.append({"text": text, "chat_id": chat_id, **kw})
        return []


class _FakeState:
    def __init__(self) -> None:
        self.sessions: dict[str, Any] = {}

    def set_session(self, key: str, value: Any, *, ttl: int = 3600) -> None:
        self.sessions[key] = value

    def get_session(self, key: str, default: Any = None) -> Any:
        return self.sessions.get(key, default)

    def clear_session(self, key: str) -> None:
        self.sessions.pop(key, None)

    def list_sessions(self, prefix: str) -> list[Any]:
        return [v for k, v in self.sessions.items() if k.startswith(prefix)]


@pytest.mark.asyncio
async def test_run_report_injects_tasks_section(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(psycho_bot, "build_week_tasks_section", lambda df, dt: "SENTINEL_TASKS")
    agent = _CapAgent()
    state = _FakeState()
    key, value = tools.session_record("мысль", source="text")
    state.set_session(key, value)
    bot = psycho_bot.PsychoBot(
        telegram=_FakeTg(),
        agent_factory=lambda system, **kw: agent,
        state=state,
        owner_chat_id=42,
    )
    await bot.run_report()
    assert agent.prompt is not None and "SENTINEL_TASKS" in agent.prompt
