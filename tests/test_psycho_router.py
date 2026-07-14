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
from psycho.bot import PsychoBot

# ---- build_psycho_bot ---------------------------------------------------


def _write_env(tmp_path: Path, **keys: str) -> Path:
    env = tmp_path / ".env"
    env.write_text("\n".join(f"{k}={v}" for k, v in keys.items()), encoding="utf-8")
    return env


def test_build_psycho_bot_wires_core(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = _write_env(
        tmp_path,
        TELEGRAM_BOT_TOKEN_PSYCHO="123:abc",
        OPENAI_API_KEY="test-key",
        TELEGRAM_CHAT_ID="42",
    )
    cfg = config.load(
        ["TELEGRAM_BOT_TOKEN_PSYCHO", "OPENAI_API_KEY", "TELEGRAM_CHAT_ID"],
        env_path=env,
    )
    # Don't construct a real openai client.
    monkeypatch.setattr(router.openai, "OpenAI", lambda **kw: object())

    bot, telegram = router.build_psycho_bot(cfg)
    assert isinstance(bot, PsychoBot)
    assert bot._owner == 42
    assert callable(bot._agent_factory)
    # the session store exposes the enumeration the report needs
    assert hasattr(bot._state, "list_sessions")
    assert telegram is not None
    # Delivery: report + analysis go straight to the owner's Telegram chat.


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
    dp = router.build_dispatcher(_FakeBot())  # type: ignore[arg-type]
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
    dp = router.build_dispatcher(fake)  # type: ignore[arg-type]
    cb = _callback_for(dp, kind="text")
    await cb(_StubMessage(chat_id=7, text="сегодня тяжело"))
    assert fake.text_calls == [(7, "сегодня тяжело")]


@pytest.mark.asyncio
async def test_voice_handler_logs_captioned_note() -> None:
    fake = _FakeBot()
    dp = router.build_dispatcher(fake)  # type: ignore[arg-type]
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

    dp = router.build_dispatcher(fake)  # type: ignore[arg-type]
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

    dp = router.build_dispatcher(fake)  # type: ignore[arg-type]
    cb = _callback_for(dp, kind="voice")
    # must not raise
    await cb(_StubMessage(chat_id=7, voice=_StubVoice(duration=9), bot=stub_bot))
    assert fake.voice_calls == []  # nothing logged, but no crash
