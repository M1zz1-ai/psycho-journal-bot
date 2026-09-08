"""The chat allowlist: a bot with write tools answers ONE chat and no other.

Every bot in the fleet takes Telegram text straight into an LLM that holds
tools. Without an authorization layer the only thing standing between a
stranger's DM and Bogdan's Notion is the bot's username being unguessed, which
is not a control. These tests pin the two halves of the guard that matter:

* an unauthorized update never reaches the handler (so the bot never acts), and
* nothing is sent back (a reply — even a refusal — confirms the bot exists and
  that the token is live).
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from aiogram import Dispatcher

from core import tg_auth

OWNER = 111
STRANGER = 999


def _message(chat_id: int, text: str = "list my tasks") -> SimpleNamespace:
    return SimpleNamespace(chat=SimpleNamespace(id=chat_id), text=text)


class _Recorder:
    """Stands in for the handler chain: records what reached the bot."""

    def __init__(self) -> None:
        self.seen: list[int] = []

    async def __call__(self, event, data):  # noqa: ANN001 - aiogram handler shape
        self.seen.append(event.chat.id)
        return "handled"


@pytest.mark.asyncio
async def test_owner_message_reaches_the_handler() -> None:
    handler = _Recorder()
    mw = tg_auth.ChatAllowlistMiddleware([OWNER])
    result = await mw(handler, _message(OWNER), {})
    assert handler.seen == [OWNER]
    assert result == "handled"


@pytest.mark.asyncio
async def test_stranger_never_reaches_the_handler() -> None:
    handler = _Recorder()
    mw = tg_auth.ChatAllowlistMiddleware([OWNER])
    result = await mw(handler, _message(STRANGER), {})
    assert handler.seen == []
    assert result is None


@pytest.mark.asyncio
async def test_drop_is_silent_and_logged_with_the_chat_id(caplog) -> None:
    """No reply goes back — but the attempt is not invisible to the operator."""
    mw = tg_auth.ChatAllowlistMiddleware([OWNER])
    with caplog.at_level(logging.WARNING, logger="core.tg_auth"):
        await mw(_Recorder(), _message(STRANGER), {})
    assert str(STRANGER) in caplog.text
    assert any(r.levelno == logging.WARNING for r in caplog.records)


@pytest.mark.asyncio
async def test_callback_query_chat_is_resolved_through_its_message() -> None:
    handler = _Recorder()
    mw = tg_auth.ChatAllowlistMiddleware([OWNER])
    callback = SimpleNamespace(message=_message(STRANGER), data="x:y:z")
    assert await mw(handler, callback, {}) is None
    assert handler.seen == []


@pytest.mark.asyncio
async def test_event_with_no_resolvable_chat_is_dropped() -> None:
    """Fail closed: an event shape we cannot place is not an authorized one."""
    handler = _Recorder()
    mw = tg_auth.ChatAllowlistMiddleware([OWNER])
    assert await mw(handler, SimpleNamespace(data="orphan"), {}) is None
    assert handler.seen == []


def test_empty_allowlist_is_refused_at_construction() -> None:
    """An empty allowlist would silently drop everything, bot included."""
    with pytest.raises(ValueError):
        tg_auth.ChatAllowlistMiddleware([])


def test_install_registers_outer_middleware_on_the_dispatcher() -> None:
    """OUTER, so the check runs before filters — not once per matching handler."""
    dp = Dispatcher()
    mw = tg_auth.install_chat_allowlist(dp, [OWNER])
    assert mw in list(dp.message.outer_middleware)
    assert mw in list(dp.callback_query.outer_middleware)
