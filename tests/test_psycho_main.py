"""Unit tests for psycho.__main__ --check + live_smoke gating.

--check must fail loud (exit 1) naming missing keys, and pass (exit 0) when all
required keys are present. live_smoke must skip cleanly (exit 0) without creds.
No network in any path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core import config
from psycho import __main__ as psycho_main
from psycho import live_smoke


def _write_env(tmp_path: Path, **keys: str) -> Path:
    env = tmp_path / ".env"
    env.write_text("\n".join(f"{k}={v}" for k, v in keys.items()), encoding="utf-8")
    return env


# ---- --check ------------------------------------------------------------


def _patch_env(monkeypatch: pytest.MonkeyPatch, env_path: Path) -> None:
    """Force psycho_main's config.load to read ``env_path``.

    ``config.load``'s ``env_path`` default is bound to MASTER_ENV_PATH at
    def-time, so patching the module constant doesn't reroute it; wrap load to
    inject the path explicitly — the honest seam for testing main()'s flow.
    """
    real_load = config.load
    monkeypatch.setattr(
        psycho_main.config,
        "load",
        lambda required, **kw: real_load(required, env_path=env_path),
    )


def test_check_fails_loud_on_missing_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_env(monkeypatch, _write_env(tmp_path))
    monkeypatch.setattr(psycho_main.sys, "argv", ["psycho", "--check"])

    rc = psycho_main.main()
    assert rc == 1
    err = capsys.readouterr().err
    # names at least one of the required keys
    assert any(k in err for k in psycho_main.REQUIRED_KEYS)


def test_check_passes_when_keys_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    env = _write_env(
        tmp_path,
        TELEGRAM_BOT_TOKEN_PSYCHO="123:abc",
        TELEGRAM_CHAT_ID="42",
        OPENAI_API_KEY="test-openai-key",
    )
    _patch_env(monkeypatch, env)
    monkeypatch.setattr(psycho_main.sys, "argv", ["psycho", "--check"])

    rc = psycho_main.main()
    assert rc == 0
    assert "Config OK" in capsys.readouterr().out


def test_inproc_report_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PSYCHO_INPROC_REPORT", raising=False)
    assert psycho_main._inproc_report_enabled() is False


def test_inproc_report_enabled_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PSYCHO_INPROC_REPORT", "1")
    assert psycho_main._inproc_report_enabled() is True


def test_required_keys_match_spec() -> None:
    assert psycho_main.REQUIRED_KEYS == [
        "TELEGRAM_BOT_TOKEN_PSYCHO",
        "TELEGRAM_CHAT_ID",
        "OPENAI_API_KEY",
    ]


# ---- live_smoke gating --------------------------------------------------


def test_live_smoke_gate_skips_without_creds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    empty = _write_env(tmp_path)
    monkeypatch.setattr(config, "MASTER_ENV_PATH", empty)
    monkeypatch.setattr(live_smoke.config, "MASTER_ENV_PATH", empty)

    assert live_smoke._gate() is None
    assert "SKIP" in capsys.readouterr().out


def test_live_smoke_main_exits_zero_when_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty = _write_env(tmp_path)
    monkeypatch.setattr(config, "MASTER_ENV_PATH", empty)
    monkeypatch.setattr(live_smoke.config, "MASTER_ENV_PATH", empty)

    assert live_smoke.main() == 0
