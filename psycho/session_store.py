"""Journal session store: ``core.state.RedisState`` plus a ``list_sessions``
enumeration the report/analysis flows need.

``core.state.RedisState`` (which I must not modify) offers get/set/clear of a
single session key, but the psycho report (n8n "Redis Keys Sessions" ->
``keys psycho:session:*``) needs to enumerate ALL stored entries. This thin
wrapper adds that one capability via a redis ``SCAN`` over the namespaced
``session:<prefix>*`` keyspace, and otherwise delegates to the wrapped state.

It is the object ``PsychoBot`` expects: ``set_session`` / ``get_session`` /
``clear_session`` (delegated) + ``list_sessions(prefix)`` (added here). Like the
rest of core, it degrades gracefully: if redis is unreachable, enumeration
returns ``[]`` rather than crashing the report cycle.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import redis

logger = logging.getLogger(__name__)


class PsychoSessionStore:
    """Wrap a ``core.state.RedisState`` and add ``list_sessions`` enumeration.

    Args:
        state: a ``core.state.RedisState`` (or test fake) providing the single-key
            session API. Its ``namespace`` + redis client are reused for SCAN.
    """

    def __init__(self, state: Any) -> None:
        self._state = state

    # ---- delegated single-key API --------------------------------------

    def set_session(self, key: str, value: Any, *, ttl: int = 3600) -> None:
        self._state.set_session(key, value, ttl=ttl)

    def get_session(self, key: str, default: Any = None) -> Any:
        return self._state.get_session(key, default)

    def clear_session(self, key: str) -> None:
        self._state.clear_session(key)

    # ---- added enumeration ---------------------------------------------

    def list_sessions(self, prefix: str) -> list[dict[str, Any]]:
        """Enumerate stored journal entries whose key starts with ``prefix``.

        Mirrors the n8n "Redis Keys Sessions" scan (``psycho:session:*``). The
        full redis key is ``<namespace>:session:<prefix><...>`` — core.state's
        ``set_session`` stores under ``<namespace>:session:<key>``. Returns the
        parsed JSON values (dicts); skips unparseable entries. Returns ``[]`` if
        redis can't be scanned (degraded, never raises).
        """
        client = getattr(self._state, "_client", None)
        namespace = getattr(self._state, "namespace", "app")
        if client is None:
            return []
        match = f"{namespace}:session:{prefix}*"
        out: list[dict[str, Any]] = []
        try:
            for raw_key in client.scan_iter(match=match):
                raw_value = client.get(raw_key)
                if not raw_value:
                    continue
                try:
                    parsed = json.loads(raw_value)
                except (json.JSONDecodeError, TypeError):
                    continue
                if isinstance(parsed, dict):
                    out.append(parsed)
        except redis.RedisError as exc:
            logger.warning("session scan failed, returning empty: %s", exc)
            return []
        return out
