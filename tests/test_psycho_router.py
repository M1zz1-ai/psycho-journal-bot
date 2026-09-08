"""Unit tests for psycho.router — bot wiring + aiogram dispatcher routing.

openai, aiogram networking, and redis are all faked. No real bot, no network.

The dispatcher tests invoke the registered handler callbacks directly (rather than
booting aiogram's filter engine) so we test OUR routing logic, not aiogram's
internals: the text handler normalizes to on_text, the voice handler logs a
captioned note verbatim and transcribes an uncaptioned one via Whisper.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from aiogram import Dispatcher

from core import config
from psycho import router
from psycho.bot import HEARTBEAT_KEY, PsychoBot

# ---- build_psycho_bot ---------------------------------------------------


def _write_env(tmp_path: Path, **keys: str) -> Path:
    env = tmp_path / ".env"
    env.write_text("\n".join(f"{k}={v}" for k, v in keys.items()), encoding="utf-8")
    return env


def _load_test_cfg(tmp_path: Path) -> config.Config:
    env = _write_env(
        tmp_path,
        TELEGRAM_BOT_TOKEN_PSYCHO="123:abc",
        OPENAI_API_KEY="sk-test",
        TELEGRAM_CHAT_ID="42",
    )
    return config.load(
        ["TELEGRAM_BOT_TOKEN_PSYCHO", "OPENAI_API_KEY", "TELEGRAM_CHAT_ID"],
        env_path=env,
    )


def test_build_psycho_bot_wires_core(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _load_test_cfg(tmp_path)
    # Don't construct a real openai client.
    monkeypatch.setattr(router.openai, "OpenAI", lambda **kw: object())

    bot, telegram = router.build_psycho_bot(cfg)
    assert isinstance(bot, PsychoBot)
    assert bot._owner == 42
    assert callable(bot._agent_factory)
    # the session store exposes the enumeration the report needs
    assert hasattr(bot._state, "list_sessions")
    assert telegram is not None


def test_build_psycho_bot_leaves_card_sink_unset_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No consumer drains ``approval:psycho`` today (drainer retired 2026-07-24).

    Regression for the 2026-07-24 to 2026-09-08 outage: wiring this sink
    unconditionally meant every analysis/report was written to a list nobody
    reads and never sent to Telegram. Default must be off.
    """
    cfg = _load_test_cfg(tmp_path)
    monkeypatch.setattr(router.openai, "OpenAI", lambda **kw: object())
    monkeypatch.delenv(router.PSYCHO_EMIT_CARDS_ENV, raising=False)

    bot, _telegram = router.build_psycho_bot(cfg)
    assert bot._card_sink is None


def test_build_psycho_bot_wires_card_sink_when_flag_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the flag on, ``_make_card_sink`` runs and needs ``card`` importable —
    an internal-only module, not part of the public showcase export (see
    ``showcase/export-psycho-bot.sh``): skip rather than fail where it is absent.
    """
    pytest.importorskip("psycho.card")
    cfg = _load_test_cfg(tmp_path)
    monkeypatch.setattr(router.openai, "OpenAI", lambda **kw: object())
    monkeypatch.setenv(router.PSYCHO_EMIT_CARDS_ENV, "1")

    bot, _telegram = router.build_psycho_bot(cfg)
    assert callable(bot._card_sink)


def test_build_psycho_bot_flag_0_also_leaves_sink_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _load_test_cfg(tmp_path)
    monkeypatch.setattr(router.openai, "OpenAI", lambda **kw: object())
    monkeypatch.setenv(router.PSYCHO_EMIT_CARDS_ENV, "0")

    bot, _telegram = router.build_psycho_bot(cfg)
    assert bot._card_sink is None


# ---- read_report_heartbeat -----------------------------------------------


def _fake_redis_state_factory(*, to_return: str | None, captured: dict[str, object]):
    """Build a ``state.RedisState`` stand-in so the heartbeat read never
    touches a real redis connection; records the constructor args it saw."""

    class _Fake:
        def __init__(self, url: str, *, namespace: str) -> None:
            captured["url"] = url
            captured["namespace"] = namespace

        def get_session(self, key: str, default: object | None = None) -> object | None:
            assert key == HEARTBEAT_KEY
            return to_return if to_return is not None else default

    return _Fake


def test_read_report_heartbeat_returns_stored_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _load_test_cfg(tmp_path)
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        router.state,
        "RedisState",
        _fake_redis_state_factory(to_return="2026-09-08T00:00:00+00:00", captured=captured),
    )

    result = router.read_report_heartbeat(cfg)
    assert result == "2026-09-08T00:00:00+00:00"
    assert captured["namespace"] == router.REDIS_NAMESPACE


def test_read_report_heartbeat_none_when_never_delivered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _load_test_cfg(tmp_path)
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        router.state, "RedisState", _fake_redis_state_factory(to_return=None, captured=captured)
    )

    assert router.read_report_heartbeat(cfg) is None


def test_card_sink_emits_valid_action_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """The wired sink builds an action_request card and pushes it to redis.

    ``card`` is an internal-only module, not part of the public showcase
    export (see ``showcase/export-psycho-bot.sh``): skip rather than fail
    where it is absent.
    """
    psycho_card = pytest.importorskip("psycho.card")

    pushed: list[tuple[object, dict]] = []
    monkeypatch.setattr(
        psycho_card, "push_card", lambda st, item: pushed.append((st, item)) or True
    )
    sentinel_state = object()
    sink = router._make_card_sink(sentinel_state)  # type: ignore[arg-type]
    sink("Разбор.", period_label="неделя", source_session="2026-05-01-1200")
    assert len(pushed) == 1
    st, item = pushed[0]
    assert st is sentinel_state
    assert item["type"] == "action_request"
    assert item["details"]["source_session"] == "2026-05-01-1200"


def test_agent_factory_builds_agent_with_system_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeAgent:
        def __init__(self, client: object, *, system: str = "", **kw: object) -> None:
            captured["system"] = system
            captured["kw"] = kw

    monkeypatch.setattr(router.core_agent, "OpenAIAgent", _FakeAgent)
    factory = router._agent_factory(object())
    factory("SYS PROMPT", max_tokens=300, model="gpt-5.4-mini")
    assert captured["system"] == "SYS PROMPT"
    assert captured["kw"] == {"max_tokens": 300, "model": "gpt-5.4-mini"}


# ---- dispatcher wiring --------------------------------------------------


def test_build_dispatcher_registers_three_handlers() -> None:
    dp = router.build_dispatcher(_FakeBot(), allowed_chat_ids=[7])  # type: ignore[arg-type]
    assert isinstance(dp, Dispatcher)
    # /start command + voice/audio + plain text = three message handlers
    assert len(dp.message.handlers) == 3


# ---- handler behavior (callbacks invoked directly) ----------------------


class _FakeBot:
    def __init__(self) -> None:
        self.text_calls: list[tuple[int, str]] = []
        self.voice_calls: list[tuple[int, str, int]] = []

    async def on_text(self, chat_id: int, text: str) -> None:
        self.text_calls.append((chat_id, text))

    async def on_voice_transcript(
        self, chat_id: int, transcript: str, *, duration_sec: int = 0
    ) -> None:
        self.voice_calls.append((chat_id, transcript, duration_sec))


class _StubChat:
    def __init__(self, cid: int) -> None:
        self.id = cid


class _StubVoice:
    def __init__(self, duration: int = 0, file_id: str = "file-1") -> None:
        self.duration = duration
        self.file_id = file_id


class _StubFile:
    def __init__(self, file_path: str = "voice/file-1.ogg") -> None:
        self.file_path = file_path


class _StubBot:
    """Fake aiogram Bot exposing just get_file + download_file for STT."""

    def __init__(self, audio: bytes = b"OGG-BYTES") -> None:
        self._audio = audio
        self.downloaded: list[str] = []

    async def get_file(self, file_id: str) -> _StubFile:
        return _StubFile(file_path=f"voice/{file_id}.ogg")

    async def download_file(self, file_path: str) -> io.BytesIO:
        self.downloaded.append(file_path)
        return io.BytesIO(self._audio)


class _StubMessage:
    def __init__(
        self,
        *,
        chat_id: int,
        text: str | None = None,
        voice: _StubVoice | None = None,
        audio: _StubVoice | None = None,
        caption: str | None = None,
        bot: _StubBot | None = None,
    ) -> None:
        self.chat = _StubChat(chat_id)
        self.text = text
        self.voice = voice
        self.audio = audio
        self.caption = caption
        self.bot = bot


def _callback_for(dp: Dispatcher, *, kind: str):
    """Return the handler callback whose own (non-Command) filter matches ``kind``.

    The text handler's filter accepts a text-only message; the voice handler's
    accepts a voice-only message. The /start handler uses aiogram's Command
    filter (needs a bot) and is exercised via the text path instead.
    """
    text_msg = _StubMessage(chat_id=0, text="x")
    voice_msg = _StubMessage(chat_id=0, voice=_StubVoice())
    for handler in dp.message.handlers:
        lambdas = [f for f in (handler.filters or []) if _is_plain_lambda(f.callback)]
        if not lambdas:
            continue
        flt = lambdas[0].callback
        if kind == "text" and flt(text_msg) and not flt(voice_msg):
            return handler.callback
        if kind == "voice" and flt(voice_msg) and not flt(text_msg):
            return handler.callback
    raise AssertionError(f"no {kind} handler found")


def _is_plain_lambda(fn: object) -> bool:
    return getattr(fn, "__name__", "") == "<lambda>"


@pytest.mark.asyncio
async def test_text_handler_normalizes_to_on_text() -> None:
    fake = _FakeBot()
    dp = router.build_dispatcher(fake, allowed_chat_ids=[7])  # type: ignore[arg-type]
    cb = _callback_for(dp, kind="text")
    await cb(_StubMessage(chat_id=7, text="сегодня тяжело"))
    assert fake.text_calls == [(7, "сегодня тяжело")]


@pytest.mark.asyncio
async def test_voice_handler_logs_captioned_note() -> None:
    fake = _FakeBot()
    dp = router.build_dispatcher(fake, allowed_chat_ids=[7])  # type: ignore[arg-type]
    cb = _callback_for(dp, kind="voice")
    await cb(_StubMessage(chat_id=7, voice=_StubVoice(duration=9), caption="наговорил мысль"))
    assert fake.voice_calls == [(7, "наговорил мысль", 9)]


@pytest.mark.asyncio
async def test_voice_handler_transcribes_uncaptioned_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No caption -> download + Whisper -> transcript goes to the ledger path."""
    fake = _FakeBot()
    stub_bot = _StubBot(audio=b"OGG-OPUS")
    seen: dict[str, object] = {}

    async def _fake_transcribe(audio, *, language=None, **kw):  # noqa: ANN001
        seen["audio"] = audio
        seen["language"] = language
        return "  распознанная мысль  "

    monkeypatch.setattr(router.stt, "transcribe", _fake_transcribe)

    dp = router.build_dispatcher(fake, allowed_chat_ids=[7])  # type: ignore[arg-type]
    cb = _callback_for(dp, kind="voice")
    await cb(_StubMessage(chat_id=7, voice=_StubVoice(duration=9), bot=stub_bot))

    assert seen["audio"] == b"OGG-OPUS"
    assert seen["language"] == "ru"
    assert stub_bot.downloaded == ["voice/file-1.ogg"]
    # same path as a captioned/text note: logged to the ledger, transcript stripped
    assert fake.voice_calls == [(7, "распознанная мысль", 9)]


@pytest.mark.asyncio
async def test_voice_handler_survives_stt_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Whisper/download failure is swallowed so the long-poll loop stays alive."""
    fake = _FakeBot()
    stub_bot = _StubBot()

    async def _boom(audio, *, language=None, **kw):  # noqa: ANN001
        raise router.stt.SttError("whisper down")

    monkeypatch.setattr(router.stt, "transcribe", _boom)

    dp = router.build_dispatcher(fake, allowed_chat_ids=[7])  # type: ignore[arg-type]
    cb = _callback_for(dp, kind="voice")
    # must not raise
    await cb(_StubMessage(chat_id=7, voice=_StubVoice(duration=9), bot=stub_bot))
    assert fake.voice_calls == []  # nothing logged, but no crash
