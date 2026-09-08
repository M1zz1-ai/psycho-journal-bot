"""Chat authorization for aiogram bots — the allowlist, as one middleware.

Every bot in this monorepo hands Telegram text to an LLM that holds tools, so
the chat an update came from is the whole authorization story: a bot username
is discoverable, and `/start` from a stranger otherwise buys the same tool
surface Bogdan has. The check therefore belongs BEFORE the handlers rather than
inside each one — a filter added per handler is a filter someone forgets on the
next handler.

Two deliberate choices:

* **Outer middleware.** ``outer_middleware`` runs before the handler filters, so
  an unauthorized update costs one set lookup and never touches a bot object,
  a Redis call, or an OpenAI call. Inner middleware would run per matching
  handler and after filtering — the same answer, later and repeatedly.
* **A silent drop.** No reply, not even a refusal. Any answer confirms that the
  username resolves to a live bot with a valid token, which is exactly the fact
  an unauthorized prober is testing for. The operator still sees it: one WARNING
  per dropped update, carrying the chat id, which is what makes an attempt
  visible in the process logs without making it visible to its author.

Wire it with :func:`install_chat_allowlist` right after the Dispatcher is built.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Iterable
from typing import Any

from aiogram import Dispatcher
from aiogram.dispatcher.middlewares.base import BaseMiddleware
from aiogram.types import TelegramObject

logger = logging.getLogger(__name__)


def _chat_id_of(event: Any) -> int | None:
    """Best-effort chat id for a Message, an edited Message or a CallbackQuery.

    Returns None when no chat can be resolved — which the caller treats as
    unauthorized, because an event we cannot place is not one we can vouch for.
    """
    chat = getattr(event, "chat", None)
    if chat is None:
        chat = getattr(getattr(event, "message", None), "chat", None)
    chat_id = getattr(chat, "id", None)
    return chat_id if isinstance(chat_id, int) else None


class ChatAllowlistMiddleware(BaseMiddleware):
    """Pass updates from allowed chats; drop everything else silently.

    Args:
        allowed: The chat ids permitted to reach the handlers — in practice the
            parsed ``TELEGRAM_CHAT_ID`` of the unit (see ``core.tg.gather_chat_ids``).

    Raises:
        ValueError: If ``allowed`` is empty. An empty allowlist drops every
            update including the owner's, which reads in production as "the bot
            went mute" — a misconfiguration must fail at boot, not at 07:00.
    """

    def __init__(self, allowed: Iterable[int]) -> None:
        self._allowed = frozenset(allowed)
        if not self._allowed:
            raise ValueError("chat allowlist is empty — refusing to drop every update")

    @property
    def allowed(self) -> frozenset[int]:
        """The chat ids this middleware lets through (for logging and tests)."""
        return self._allowed

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        chat_id = _chat_id_of(event)
        if chat_id is None or chat_id not in self._allowed:
            logger.warning(
                "dropped update from unauthorized chat_id=%s (%s)",
                chat_id,
                type(event).__name__,
            )
            return None
        return await handler(event, data)


def install_chat_allowlist(dp: Dispatcher, allowed: Iterable[int]) -> ChatAllowlistMiddleware:
    """Attach one allowlist to every user-originated observer of ``dp``.

    Returns the middleware so a caller can log or assert what was installed.
    """
    middleware = ChatAllowlistMiddleware(allowed)
    dp.message.outer_middleware(middleware)
    dp.edited_message.outer_middleware(middleware)
    dp.callback_query.outer_middleware(middleware)
    return middleware
