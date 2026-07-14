"""Entrypoint: one asyncio process running the psycho journal bot.

Combines the three n8n workflows this module replaces into a single long-lived
process: the Bot Router (long-poll), the OnDemand Analysis (triggered from chat),
and the weekly Report Workflow (in-process scheduler, the crypto-news pattern).

Run modes:
  python -m psycho           # run: long-poll router + weekly report scheduler
  python -m psycho --once    # build and send one report, then exit (cron-style)
  python -m psycho --check   # validate config loading, then exit

Config keys (declared required):
  TELEGRAM_BOT_TOKEN_PSYCHO, TELEGRAM_CHAT_ID, OPENAI_API_KEY
  OPENAI_API_KEY powers BOTH the LLM brain (analysis + weekly report, via
  core.openai_agent) and voice-note transcription (core.stt / Whisper); without
  it uncaptioned voice notes — the user's primary input — cannot be logged, and no
  analysis runs. ANTHROPIC_API_KEY is no longer used by psycho (brain moved to
  OpenAI after the direct Anthropic key ran out of credits).
  REDIS_URL is optional (defaults to redis://localhost:6379 via core.config).

A missing key fails loud naming the key (core.config.ConfigError) — the process
never silently runs without credentials.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from core import config
from core.errors import ConfigError

from .report import run_report_scheduler
from .router import build_dispatcher, build_psycho_bot

logger = logging.getLogger("psycho_bot")


def _inproc_report_enabled() -> bool:
    """Whether to run the weekly report loop inside the long-poll process.

    Off by default: the weekly report is driven by a Sunday-aligned systemd
    timer (``psycho-report.timer`` -> ``--once``), so the long-poll
    service keeps only routing/journaling. Set ``PSYCHO_INPROC_REPORT=1`` to
    restore the old in-process scheduler (e.g. a host without systemd timers).
    """
    return os.getenv("PSYCHO_INPROC_REPORT", "0") == "1"

REQUIRED_KEYS = [
    "TELEGRAM_BOT_TOKEN_PSYCHO",
    "TELEGRAM_CHAT_ID",
    "OPENAI_API_KEY",
]


async def run_once(cfg: config.Config) -> None:
    """Build and send a single weekly report, then exit (cron-style one-shot)."""
    bot, telegram = build_psycho_bot(cfg)
    try:
        await bot.run_report()
    finally:
        await telegram.close()


async def run(cfg: config.Config) -> None:
    """Run the router long-poll; the weekly report is a systemd timer by default.

    The in-process report scheduler is opt-in (``PSYCHO_INPROC_REPORT=1``); the
    default deployment fires ``--once`` from a Sunday-aligned systemd timer, so
    this process keeps only routing/journaling.
    """
    bot, telegram = build_psycho_bot(cfg)
    dp = build_dispatcher(bot)
    scheduler: asyncio.Task[None] | None = None
    if _inproc_report_enabled():
        logger.info("psycho-bot started; long-poll router + in-process weekly report")
        scheduler = asyncio.create_task(
            run_report_scheduler(bot, alerter=telegram, run_immediately=False)
        )
    else:
        logger.info("psycho-bot started; long-poll router only (weekly report via systemd timer)")
    try:
        await dp.start_polling(telegram.bot, handle_signals=False)
    finally:
        if scheduler is not None:
            scheduler.cancel()
        await telegram.close()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(prog="psycho")
    parser.add_argument("--check", action="store_true", help="Validate config and exit.")
    parser.add_argument("--once", action="store_true", help="Send one report and exit.")
    args = parser.parse_args()

    try:
        cfg = config.load(REQUIRED_KEYS)
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 1

    if args.check:
        print(f"Config OK — all {len(REQUIRED_KEYS)} required keys present.")
        return 0

    try:
        asyncio.run(run_once(cfg) if args.once else run(cfg))
    except KeyboardInterrupt:
        logger.info("shutting down")
    return 0


if __name__ == "__main__":
    sys.exit(main())
