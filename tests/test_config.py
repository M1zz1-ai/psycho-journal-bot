"""Config loading: requested-only keys, defaults, fail-loud on missing."""

import pytest

from core.config import Config, load
from core.errors import ConfigError

ENV = """\
TELEGRAM_BOT_TOKEN_PSYCHO=tok
TELEGRAM_CHAT_ID=111222333
OPENAI_API_KEY=test-openai-key
"""


def _write(tmp_path, content=ENV):
    p = tmp_path / ".env"
    p.write_text(content)
    return p


def test_load_only_requested_keys(tmp_path):
    cfg = load(["TELEGRAM_BOT_TOKEN_PSYCHO", "OPENAI_API_KEY"], env_path=_write(tmp_path))
    assert isinstance(cfg, Config)
    assert cfg.require("TELEGRAM_BOT_TOKEN_PSYCHO") == "tok"
    assert cfg.require("OPENAI_API_KEY") == "test-openai-key"
    # A key that was not requested is simply absent.
    assert cfg.get("TELEGRAM_CHAT_ID") is None


def test_missing_key_raises_named_error(tmp_path):
    with pytest.raises(ConfigError) as exc:
        load(["TELEGRAM_BOT_TOKEN_PSYCHO", "MISSING_TOKEN"], env_path=_write(tmp_path))
    assert "MISSING_TOKEN" in str(exc.value)


def test_empty_value_treated_as_missing(tmp_path):
    env = ENV.replace("OPENAI_API_KEY=test-openai-key", "OPENAI_API_KEY=")
    with pytest.raises(ConfigError) as exc:
        load(["OPENAI_API_KEY"], env_path=_write(tmp_path, env))
    assert "OPENAI_API_KEY" in str(exc.value)


def test_redis_url_default_when_absent(tmp_path):
    cfg = load(["REDIS_URL"], env_path=_write(tmp_path))
    assert cfg.require("REDIS_URL") == "redis://localhost:6379"


def test_redis_url_override_from_env(tmp_path):
    env = ENV + "REDIS_URL=redis://otherhost:6380\n"
    cfg = load(["REDIS_URL"], env_path=_write(tmp_path, env))
    assert cfg.require("REDIS_URL") == "redis://otherhost:6380"


def test_attribute_access(tmp_path):
    cfg = load(["OPENAI_API_KEY"], env_path=_write(tmp_path))
    assert cfg.openai_api_key == "test-openai-key"
    with pytest.raises(AttributeError):
        _ = cfg.nonexistent_key


def test_requested_keys_are_independent(tmp_path):
    p = _write(tmp_path)
    a = load(["TELEGRAM_BOT_TOKEN_PSYCHO"], env_path=p)
    b = load(["OPENAI_API_KEY"], env_path=p)
    assert a.get("OPENAI_API_KEY") is None
    assert b.get("TELEGRAM_BOT_TOKEN_PSYCHO") is None
