"""Unit tests for psycho.session_store — list_sessions SCAN + redis-down fallback.

A fake redis client backs the wrapped core.state.RedisState; no real redis.
"""

from __future__ import annotations

from typing import Any

import redis

from core import state as core_state
from psycho.session_store import PsychoSessionStore


class _FakeRedis:
    """Minimal redis stand-in: set/get/delete + scan_iter over a dict."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.store[key] = value

    def get(self, key: str) -> str | None:
        return self.store.get(key)

    def delete(self, key: str) -> None:
        self.store.pop(key, None)

    def exists(self, key: str) -> int:
        return 1 if key in self.store else 0

    def scan_iter(self, match: str = "*"):
        # only the trailing "*" wildcard is used by the store
        prefix = match.rstrip("*")
        for k in list(self.store):
            if k.startswith(prefix):
                yield k


class _BoomRedis(_FakeRedis):
    def scan_iter(self, match: str = "*"):
        raise redis.RedisError("scan down")
        yield  # pragma: no cover


def _store(client: Any) -> PsychoSessionStore:
    rs = core_state.RedisState(namespace="psycho", client=client)
    return PsychoSessionStore(rs)


def test_list_sessions_enumerates_namespaced_keys() -> None:
    client = _FakeRedis()
    store = _store(client)
    store.set_session("session:1000", {"transcript": "a", "timestamp": 1000})
    store.set_session("session:2000", {"transcript": "b", "timestamp": 2000})
    # an unrelated session key in the same namespace must not match the prefix
    store.set_session("awaiting:42", "analyze")

    out = store.list_sessions("session:")
    transcripts = sorted(s["transcript"] for s in out)
    assert transcripts == ["a", "b"]


def test_list_sessions_skips_unparseable_values() -> None:
    client = _FakeRedis()
    store = _store(client)
    store.set_session("session:1", {"transcript": "ok", "timestamp": 1})
    # inject a malformed raw value directly under the namespaced key
    client.store["psycho:session:bad"] = "{not json"

    out = store.list_sessions("session:")
    assert [s["transcript"] for s in out] == ["ok"]


def test_list_sessions_returns_empty_on_redis_error() -> None:
    store = _store(_BoomRedis())
    assert store.list_sessions("session:") == []


def test_delegated_get_set_clear_roundtrip() -> None:
    client = _FakeRedis()
    store = _store(client)
    store.set_session("awaiting:42", "analyze")
    assert store.get_session("awaiting:42") == "analyze"
    store.clear_session("awaiting:42")
    assert store.get_session("awaiting:42") is None
