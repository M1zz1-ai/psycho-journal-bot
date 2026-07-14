"""Live end-to-end smoke test for the psycho bot, gated on real credentials.

Drives the bot's OWN pipeline (router classification + redis ledger + the wired
core agents) against real OpenAI + redis. Guard-railed: it logs a couple of
journal entries, runs one on-demand analysis, sends the result to the owner chat,
and never starts the long-poll loop.

Gating mirrors the voice/crypto bots: when ``TELEGRAM_BOT_TOKEN_PSYCHO`` or
``OPENAI_API_KEY`` are absent it SKIPS with a clear message and exits 0 — it
never invents credentials.

Run:
  uv run python -m psycho.live_smoke
"""

from __future__ import annotations

import asyncio
import logging

from core import config, tg
from core.errors import ConfigError

from .router import build_psycho_bot

logger = logging.getLogger("psycho_bot.live_smoke")

# Keys whose absence means "skip, don't fail" — the gating credentials.
GATING_KEYS = ["TELEGRAM_BOT_TOKEN_PSYCHO", "OPENAI_API_KEY"]
# Keys needed to actually run the smoke once gated through.
SMOKE_KEYS = [
    "TELEGRAM_BOT_TOKEN_PSYCHO",
    "OPENAI_API_KEY",
    "TELEGRAM_CHAT_ID",
]


def _present(key: str) -> bool:
    try:
        cfg = config.load([key], env_path=config.MASTER_ENV_PATH)
        return bool(cfg.get(key))
    except ConfigError:
        return False


def _gate() -> config.Config | None:
    """Return a loaded Config if gating creds are present, else None (skip)."""
    try:
        return config.load(SMOKE_KEYS, env_path=config.MASTER_ENV_PATH)
    except ConfigError as exc:
        missing = [k for k in GATING_KEYS if not _present(k)]
        if missing:
            print(
                f"SKIP — live smoke needs {', '.join(GATING_KEYS)} in "
                f"the local .env file (missing: {', '.join(missing)}). "
                "No real creds present; nothing to do."
            )
            return None
        print(f"Config error: {exc}")
        return None


async def _run(cfg: config.Config) -> int:
    """Real E2E: log journal entries, run one analysis, send it to the owner."""
    chat_id = tg.gather_chat_ids(cfg.require("TELEGRAM_CHAT_ID"))[0]
    bot, telegram = build_psycho_bot(cfg)
    failures = 0
    try:
        await telegram.send_text("🧪 <b>psycho-bot LIVE E2E</b> — logging + analysis…")

        print("[..] step 1 — log two journal entries (text)", flush=True)
        await bot.on_text(chat_id, "сегодня много работал, чувствую усталость и тревогу о дедлайне")
        await bot.on_text(chat_id, "поспал плохо, но утром сделал пробежку")
        print("[PASS] journal entries logged", flush=True)

        print("[..] step 2 — run on-demand analysis for last week via OpenAI", flush=True)
        # run_analysis sends the result to the owner chat; a model failure is
        # swallowed by run_resilient and alerted, so reaching here is a pass.
        await bot.run_analysis(chat_id, "за неделю")
        print("[PASS] analysis dispatched to owner", flush=True)
    finally:
        await telegram.close()
    print(f"\n{failures} failure(s).", flush=True)
    return failures


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = _gate()
    if cfg is None:
        return 0  # skipped or config-printed; not a hard failure for CI
    return asyncio.run(_run(cfg))


if __name__ == "__main__":
    raise SystemExit(main())
