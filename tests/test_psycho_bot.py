"""Unit tests for psycho.bot — router handlers + analysis/report orchestration.

Telegram, the LLM agent, and redis are all faked. No network, no real bot.
"""

from __future__ import annotations

from typing import Any

import pytest

from psycho import bot as psycho_bot
from psycho import tools

# ---- fakes --------------------------------------------------------------


class _FakeTg:
    def __init__(self) -> None:
        self.texts: list[dict[str, Any]] = []

    async def send_text(self, text: str, chat_id: int | None = None, **kw: Any) -> list[Any]:
        self.texts.append({"text": text, "chat_id": chat_id, **kw})
        return []


class _FakeAgent:
    def __init__(self, *, run_reply: str = "ok", structured: dict[str, Any] | None = None) -> None:
        self.run_reply = run_reply
        self.structured = structured or {}
        self.run_prompts: list[str] = []

    def run(self, prompt: str) -> str:
        self.run_prompts.append(prompt)
        return self.run_reply

    def structured_output(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        return self.structured


class _FakeState:
    """In-memory stand-in for core.state.RedisState."""

    def __init__(self) -> None:
        self.sessions: dict[str, Any] = {}

    def set_session(self, key: str, value: Any, *, ttl: int = 3600) -> None:
        self.sessions[key] = value

    def get_session(self, key: str, default: Any = None) -> Any:
        return self.sessions.get(key, default)

    def clear_session(self, key: str) -> None:
        self.sessions.pop(key, None)

    # report enumeration helper the bot relies on
    def list_sessions(self, prefix: str) -> list[Any]:
        return [v for k, v in self.sessions.items() if k.startswith(prefix)]


def _make_bot(*, agent: _FakeAgent | None = None, tg: _FakeTg | None = None,
              state: _FakeState | None = None) -> tuple[psycho_bot.PsychoBot, _FakeTg, _FakeState]:
    tg = tg or _FakeTg()
    state = state or _FakeState()
    agent = agent or _FakeAgent()
    bot = psycho_bot.PsychoBot(
        telegram=tg,
        agent_factory=lambda system, **kw: agent,
        state=state,
        owner_chat_id=42,
    )
    return bot, tg, state


# ---- routing handlers ---------------------------------------------------


@pytest.mark.asyncio
async def test_on_text_start_shows_menu():
    bot, tg, _ = _make_bot()
    await bot.on_text(42, "/start")
    assert tg.texts
    # the menu mentions the analyze button label
    assert any(tools.ANALYZE_BUTTON_LABEL in t["text"] for t in tg.texts)


@pytest.mark.asyncio
async def test_on_text_start_shows_reply_keyboard_with_analysis_button():
    bot, tg, _ = _make_bot()
    await bot.on_text(42, "/start")
    markup = tg.texts[-1].get("reply_markup")
    assert markup is not None, "/start must attach the reply keyboard"
    labels = [btn.text for row in markup.keyboard for btn in row]
    assert tools.ANALYZE_BUTTON_LABEL in labels


@pytest.mark.asyncio
async def test_on_text_analyze_button_sets_awaiting():
    bot, tg, state = _make_bot()
    await bot.on_text(42, tools.ANALYZE_BUTTON_LABEL)
    assert state.get_session("awaiting:42") == tools.AWAITING_VALUE
    assert tg.texts  # asked for a period


@pytest.mark.asyncio
async def test_on_text_journal_stores_session_and_acks():
    bot, tg, state = _make_bot()
    await bot.on_text(42, "сегодня было тяжело")
    stored = [k for k in state.sessions if k.startswith("session:")]
    assert stored
    assert state.sessions[stored[0]]["transcript"] == "сегодня было тяжело"
    assert tg.texts  # acknowledged


@pytest.mark.asyncio
async def test_on_text_journal_ignores_empty():
    bot, tg, state = _make_bot()
    await bot.on_text(42, "    ")
    assert not [k for k in state.sessions if k.startswith("session:")]


@pytest.mark.asyncio
async def test_on_text_awaiting_period_triggers_analysis_and_clears_flag():
    agent = _FakeAgent(
        run_reply="Холодный анализ.",
        structured={"date_from": "01.05.2026", "date_to": "07.05.2026", "label": "неделя"},
    )
    bot, tg, state = _make_bot(agent=agent)
    state.set_session("awaiting:42", tools.AWAITING_VALUE)
    await bot.on_text(42, "за неделю")
    # flag cleared, analysis sent
    assert state.get_session("awaiting:42") is None
    assert any("Холодный анализ" in t["text"] for t in tg.texts)


@pytest.mark.asyncio
async def test_on_voice_transcript_stores_session():
    bot, tg, state = _make_bot()
    await bot.on_voice_transcript(42, "проговорил мысль", duration_sec=8)
    stored = [v for k, v in state.sessions.items() if k.startswith("session:")]
    assert stored and stored[0]["source"] == "voice"
    assert stored[0]["duration_sec"] == 8


# ---- ondemand analysis --------------------------------------------------


@pytest.mark.asyncio
async def test_run_analysis_sends_result():
    agent = _FakeAgent(
        run_reply="Анализ за период.",
        structured={"date_from": "01.05.2026", "date_to": "07.05.2026", "label": "x"},
    )
    bot, tg, state = _make_bot(agent=agent)
    # seed a session inside the window
    key, value = tools.session_record("мысль", source="text")
    state.set_session(key, value)
    await bot.run_analysis(42, "за неделю")
    assert any("Анализ за период" in t["text"] for t in tg.texts)


@pytest.mark.asyncio
async def test_run_analysis_renders_raw_html_metacharacters_safely():
    """A model reply with raw ``<``/``&``/a ``<script>`` tag must never reach
    Telegram unescaped: the bot sends with bot-wide ``parse_mode="HTML"``, so
    an unescaped metacharacter makes the Bot API reject the WHOLE message with
    ``can't parse entities`` (see the module docstring). The period label is
    interpolated straight into the ``<b>...</b>`` header, so a raw ``<`` there
    is just as dangerous and must be escaped too.
    """
    agent = _FakeAgent(
        run_reply="Опасно: <script>alert(1)</script> & <b>fake</b>",
        structured={"date_from": "01.05.2026", "date_to": "07.05.2026", "label": "1 < 2 & 3"},
    )
    bot, tg, state = _make_bot(agent=agent)
    key, value = tools.session_record("мысль", source="text")
    state.set_session(key, value)

    await bot.run_analysis(42, "за неделю")

    assert len(tg.texts) == 1
    sent = tg.texts[0]["text"]
    # Body: escaped, not interpreted as HTML.
    assert "<script>" not in sent
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in sent
    assert "&amp;" in sent
    assert "<b>fake</b>" not in sent
    # Header: the interpolated label is escaped too, so the header's own
    # <b>...</b> pair stays the only real tag in that line.
    assert "1 &lt; 2 &amp; 3" in sent
    assert "<b>Разбор за 1 < 2" not in sent


# ---- scheduled report ---------------------------------------------------


@pytest.mark.asyncio
async def test_run_report_empty_ledger_notifies_empty():
    bot, tg, state = _make_bot()
    await bot.run_report()
    assert tg.texts
    assert any("нет" in t["text"].lower() or "empt" in t["text"].lower() for t in tg.texts)


@pytest.mark.asyncio
async def test_run_report_with_sessions_sends_report():
    agent = _FakeAgent(
        run_reply="## structured",
        structured={"report_md": "# Отчёт недели", "summary": "две строки"},
    )
    bot, tg, state = _make_bot(agent=agent)
    key, value = tools.session_record("мысль одна", source="text")
    state.set_session(key, value)
    await bot.run_report()
    assert any("Отчёт недели" in t["text"] for t in tg.texts)


@pytest.mark.asyncio
async def test_run_report_renders_raw_html_metacharacters_safely(monkeypatch):
    """Same failure mode as the analysis path (see
    ``test_run_analysis_renders_raw_html_metacharacters_safely``), for the
    weekly report: ``report["report_md"]`` and the period label both reach
    Telegram through the header/body path in ``run_report``.
    """
    monkeypatch.setattr(
        psycho_bot.tools, "period_label_from_sessions", lambda sessions: "week < 1 & done"
    )
    agent = _FakeAgent(
        run_reply="## structured",
        structured={
            "report_md": "# Отчёт\n\nРиск: <script>alert(1)</script> & <b>fake</b>",
            "summary": "две строки",
        },
    )
    bot, tg, state = _make_bot(agent=agent)
    key, value = tools.session_record("мысль одна", source="text")
    state.set_session(key, value)

    await bot.run_report()

    assert len(tg.texts) == 1
    sent = tg.texts[0]["text"]
    assert "<script>" not in sent
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in sent
    assert "&amp;" in sent
    assert "<b>fake</b>" not in sent
    assert "week &lt; 1 &amp; done" in sent
    assert "<b>Еженедельный разбор: week < 1" not in sent


@pytest.mark.asyncio
async def test_run_report_defaults_chat_to_owner():
    agent = _FakeAgent(structured={"report_md": "# r", "summary": "s"})
    bot, tg, state = _make_bot(agent=agent)
    key, value = tools.session_record("x", source="text")
    state.set_session(key, value)
    await bot.run_report()  # no chat_id -> owner (42)
    assert all(t["chat_id"] in (42, None) for t in tg.texts)


# ---- model routing (cheap split: mini on-demand, flagship weekly) -------


@pytest.mark.asyncio
async def test_run_analysis_uses_analysis_model():
    """On-demand analysis agents (period parse + stoic analysis) run on the mini model."""
    from psycho import analysis

    agent = _FakeAgent(
        run_reply="Анализ.",
        structured={"date_from": "01.05.2026", "date_to": "07.05.2026", "label": "x"},
    )
    models: list[Any] = []
    bot = psycho_bot.PsychoBot(
        telegram=_FakeTg(),
        agent_factory=lambda system, **kw: (models.append(kw.get("model")), agent)[1],
        state=_FakeState(),
        owner_chat_id=42,
    )
    await bot.run_analysis(42, "за неделю")
    assert models  # at least one agent built
    assert all(m == analysis.ANALYSIS_MODEL == "gpt-5.4-mini" for m in models)


@pytest.mark.asyncio
async def test_run_report_uses_report_model():
    """Weekly report agents (structure + therapist) run on the flagship model."""
    from psycho import report

    agent = _FakeAgent(structured={"report_md": "# r", "summary": "s"})
    state = _FakeState()
    key, value = tools.session_record("мысль", source="text")
    state.set_session(key, value)
    models: list[Any] = []
    bot = psycho_bot.PsychoBot(
        telegram=_FakeTg(),
        agent_factory=lambda system, **kw: (models.append(kw.get("model")), agent)[1],
        state=state,
        owner_chat_id=42,
    )
    await bot.run_report()
    assert models
    assert all(m == report.REPORT_MODEL == "gpt-5.5" for m in models)


# ---- resilience ---------------------------------------------------------


@pytest.mark.asyncio
async def test_analysis_failure_does_not_crash():
    class _BoomAgent(_FakeAgent):
        def run(self, prompt: str) -> str:
            raise RuntimeError("anthropic down")

    agent = _BoomAgent(structured={"date_from": "01.05.2026", "date_to": "07.05.2026", "label": "x"})
    bot, tg, state = _make_bot(agent=agent)
    key, value = tools.session_record("x", source="text")
    state.set_session(key, value)
    # run_resilient swallows the error; the call returns without raising
    await bot.run_analysis(42, "за неделю")


# ---- card sink is an ADDITIONAL artifact, never a substitute for Telegram --
#
# Regression coverage for the 2026-07-24 to 2026-09-08 outage: "producer mode"
# wired ``card_sink`` unconditionally and both delivery paths returned early
# once it was set, so nothing reached Telegram for six weeks. These tests pin
# the fixed contract: Telegram delivery is unconditional; the card, when a
# sink is given, is emitted in addition and its failure must not cancel the
# Telegram send.


class _RecordingSink:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, analysis: str, *, period_label: str,
                 source_session: str | None = None) -> None:
        self.calls.append(
            {"analysis": analysis, "period_label": period_label,
             "source_session": source_session}
        )


class _BoomSink:
    """A card sink that always fails, to prove Telegram delivery survives it."""

    def __call__(self, analysis: str, *, period_label: str,
                 source_session: str | None = None) -> None:
        raise RuntimeError("redis unreachable")


def _make_bot_with_sink(*, agent: _FakeAgent | None = None, sink: Any = None):
    tg = _FakeTg()
    state = _FakeState()
    agent = agent or _FakeAgent()
    sink = sink if sink is not None else _RecordingSink()
    bot = psycho_bot.PsychoBot(
        telegram=tg,
        agent_factory=lambda system, **kw: agent,
        state=state,
        owner_chat_id=42,
        card_sink=sink,
    )
    return bot, tg, state, sink


@pytest.mark.asyncio
async def test_run_analysis_sends_to_telegram_even_with_card_sink_set():
    agent = _FakeAgent(
        run_reply="Разбор с sink.",
        structured={"date_from": "01.05.2026", "date_to": "07.05.2026", "label": "неделя"},
    )
    bot, tg, state, sink = _make_bot_with_sink(agent=agent)
    key, value = tools.session_record("мысль", source="text")
    state.set_session(key, value)
    await bot.run_analysis(42, "за неделю")
    # the card is additional, not instead: both the card AND Telegram fire
    assert any("Разбор с sink" in t["text"] for t in tg.texts), (
        "card_sink set must not suppress the Telegram send"
    )
    assert len(sink.calls) == 1
    assert "Разбор с sink" in sink.calls[0]["analysis"]
    assert sink.calls[0]["period_label"] == "неделя"


@pytest.mark.asyncio
async def test_run_analysis_card_sink_failure_does_not_block_telegram():
    agent = _FakeAgent(
        run_reply="Разбор несмотря на sink.",
        structured={"date_from": "01.05.2026", "date_to": "07.05.2026", "label": "неделя"},
    )
    bot, tg, state, _sink = _make_bot_with_sink(agent=agent, sink=_BoomSink())
    key, value = tools.session_record("мысль", source="text")
    state.set_session(key, value)
    await bot.run_analysis(42, "за неделю")
    assert any("Разбор несмотря на sink" in t["text"] for t in tg.texts)


@pytest.mark.asyncio
async def test_run_report_sends_to_telegram_even_with_card_sink_set():
    agent = _FakeAgent(
        run_reply="## structured",
        structured={"report_md": "# Отчёт с sink", "summary": "s"},
    )
    bot, tg, state, sink = _make_bot_with_sink(agent=agent)
    key, value = tools.session_record("мысль одна", source="text")
    state.set_session(key, value)
    await bot.run_report()
    assert any("Отчёт с sink" in t["text"] for t in tg.texts), (
        "card_sink set must not suppress the Telegram send"
    )
    assert len(sink.calls) == 1
    assert "Отчёт с sink" in sink.calls[0]["analysis"]


@pytest.mark.asyncio
async def test_run_report_card_sink_failure_does_not_block_telegram():
    agent = _FakeAgent(structured={"report_md": "# Отчёт выжил", "summary": "s"})
    bot, tg, state, _sink = _make_bot_with_sink(agent=agent, sink=_BoomSink())
    key, value = tools.session_record("мысль одна", source="text")
    state.set_session(key, value)
    await bot.run_report()
    assert any("Отчёт выжил" in t["text"] for t in tg.texts)


@pytest.mark.asyncio
async def test_run_report_empty_ledger_notifies_even_with_card_sink_set():
    bot, tg, state, sink = _make_bot_with_sink()
    await bot.run_report()
    # empty period: still a Telegram notice, and nothing to put on a card
    assert any("нет" in t["text"].lower() or "empt" in t["text"].lower() for t in tg.texts)
    assert not sink.calls


# ---- delivery heartbeat ---------------------------------------------------


@pytest.mark.asyncio
async def test_run_report_marks_heartbeat_after_delivery():
    agent = _FakeAgent(structured={"report_md": "# r", "summary": "s"})
    bot, tg, state = _make_bot(agent=agent)
    key, value = tools.session_record("x", source="text")
    state.set_session(key, value)
    assert state.get_session(psycho_bot.HEARTBEAT_KEY) is None
    await bot.run_report()
    assert state.get_session(psycho_bot.HEARTBEAT_KEY) is not None


@pytest.mark.asyncio
async def test_run_report_marks_heartbeat_on_empty_ledger_notice_too():
    bot, tg, state = _make_bot()
    await bot.run_report()
    # an empty week is still a real delivery: the notice reached Telegram
    assert state.get_session(psycho_bot.HEARTBEAT_KEY) is not None


@pytest.mark.asyncio
async def test_run_report_does_not_mark_heartbeat_when_delivery_fails():
    """A build-time model failure alone does not prove the point: ``tools.therapist_report``
    already catches it and falls back to a sendable stub. Only a failure in the actual
    Telegram send means nothing was delivered, so that is what must skip the heartbeat.
    """

    class _BoomTg:
        async def send_text(self, text: str, chat_id: int | None = None, **kw: Any) -> Any:
            raise RuntimeError("telegram down")

    agent = _FakeAgent(structured={"report_md": "# r", "summary": "s"})
    state = _FakeState()
    key, value = tools.session_record("x", source="text")
    state.set_session(key, value)
    bot = psycho_bot.PsychoBot(
        telegram=_BoomTg(),
        agent_factory=lambda system, **kw: agent,
        state=state,
        owner_chat_id=42,
    )
    # run_resilient swallows the failure; nothing was actually delivered
    await bot.run_report()
    assert state.get_session(psycho_bot.HEARTBEAT_KEY) is None


@pytest.mark.asyncio
async def test_run_analysis_does_not_touch_report_heartbeat():
    """The heartbeat is specifically about the weekly report, not on-demand analysis."""
    agent = _FakeAgent(
        run_reply="Анализ.",
        structured={"date_from": "01.05.2026", "date_to": "07.05.2026", "label": "x"},
    )
    bot, tg, state = _make_bot(agent=agent)
    key, value = tools.session_record("мысль", source="text")
    state.set_session(key, value)
    await bot.run_analysis(42, "за неделю")
    # the analysis itself must still reach Telegram (see the regression tests above)
    assert tg.texts
    assert state.get_session(psycho_bot.HEARTBEAT_KEY) is None
