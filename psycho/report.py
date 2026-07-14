"""Scheduled weekly-report surface.

Builds a weekly cold-stoic report from the redis journal and sends it to the
owner. The report logic lives in ``tools`` (structure pass + therapist structured
output) and is orchestrated by ``PsychoBot.run_report``; this module provides the
schedule constant and the ``core.scheduler``-driven loop entry point so
``__main__`` can optionally run the report in-process.

In production the report is instead fired by a Sunday-aligned systemd timer
(``python -m psycho --once``) — see ``deploy/`` — because the in-process
scheduler cannot align to a specific weekday (see the design note in the README).
The report is built + sent from the redis ledger only; there is no external store.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from core import notion
from core.errors import Alerter, NotionError
from core.scheduler import run_interval

logger = logging.getLogger(__name__)

# The weekly report runs on OpenAI's flagship chat model — the deeper analysis
# pass (structure + therapist) warrants a stronger tier than the on-demand mini
# path. ``gpt-5.5`` (snapshot 2026-04-23) is verified against the live /v1/models
# list. Override via ``PSYCHO_REPORT_MODEL``. (Brain moved from Anthropic to
# OpenAI when the direct Anthropic key ran out of credits.)
REPORT_MODEL = os.getenv("PSYCHO_REPORT_MODEL", "gpt-5.5")

# Weekly cadence. The n8n cron was "0 1 * * 0" (Sunday 01:00). The in-process
# scheduler can't express day-of-week alignment, so it fires every 7 days from
# process start; the cron alignment is documented in the systemd unit. A failed
# cycle is logged/alerted and the loop keeps running (core.scheduler resilience).
REPORT_INTERVAL_SECONDS = 7 * 24 * 3600  # 604800


# ---- weekly tasks enrichment (Notion "Daily Task Tracker") --------------
#
# The weekly report is enriched with the user's past-week tasks pulled from Notion
# (via ``core.notion`` -> ``notion-cli``). The data-merge/formatting below are
# PURE functions (testable without Notion); ``build_week_tasks_section`` is the
# single impure entry point and NEVER raises — a Notion failure degrades to an
# "unavailable" note so it can't kill the report.

WEEK_TASKS_HEADER = "Неделя в задачах"
TASKS_UNAVAILABLE_NOTE = "данные задач недоступны"

# notion-cli status -> Russian bucket label, in display order. Unknown/missing
# statuses fall into "не начато" so no task is silently dropped (digest parity).
_TASK_BUCKETS: tuple[tuple[str, str], ...] = (
    ("Done", "сделано"),
    ("In progress", "в работе"),
    ("Not started", "не начато"),
)


def group_week_tasks(tasks: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Group task dicts into Russian status buckets -> list of titles (pure)."""
    valid = {status for status, _ in _TASK_BUCKETS}
    label_for = {status: label for status, label in _TASK_BUCKETS}
    groups: dict[str, list[str]] = {label: [] for _, label in _TASK_BUCKETS}
    for task in tasks:
        status = task.get("status") or "Not started"
        if status not in valid:
            status = "Not started"
        title = (task.get("title") or "Untitled").strip()
        groups[label_for[status]].append(title)
    return groups


def format_week_tasks(tasks: list[dict[str, Any]] | None) -> str:
    """Render the compact "Неделя в задачах" context section (pure).

    ``None`` (Notion unavailable) -> an explicit "unavailable" note; an empty
    list -> an explicit "no tasks" note; otherwise a compact per-bucket summary.
    """
    if tasks is None:
        return f"{WEEK_TASKS_HEADER}: {TASKS_UNAVAILABLE_NOTE}."
    if not tasks:
        return f"{WEEK_TASKS_HEADER}: нет задач за неделю."
    grouped = group_week_tasks(tasks)
    lines = [f"{WEEK_TASKS_HEADER}:"]
    for _, label in _TASK_BUCKETS:
        items = grouped.get(label) or []
        if items:
            lines.append(f"- {label} ({len(items)}): " + ", ".join(items))
    return "\n".join(lines)


def fetch_week_tasks(date_from: str, date_to: str) -> list[dict[str, Any]] | None:
    """Pull the week's tasks (all statuses) via ``core.notion``; ``None`` on failure.

    ``date_from``/``date_to`` are YYYY-MM-DD. A Notion failure is logged as a
    warning and returns ``None`` (caller degrades gracefully) — never raised.
    """
    try:
        return notion.list_tasks(date_from=date_from, date_to=date_to) or []
    except NotionError:
        logger.warning("weekly report: notion tasks unavailable, degrading", exc_info=True)
        return None


def build_week_tasks_section(date_from: str, date_to: str) -> str:
    """Fetch + format the week's tasks into a context section. Never raises."""
    return format_week_tasks(fetch_week_tasks(date_from, date_to))


async def run_report_scheduler(
    bot: Any,
    *,
    alerter: Alerter | None = None,
    run_immediately: bool = False,
    max_cycles: int | None = None,
) -> None:
    """Drive ``bot.run_report`` on the weekly interval via ``core.scheduler``.

    Args:
        bot: a :class:`~psycho.bot.PsychoBot` (run_report sends to its owner).
        alerter: optional Telegram client pinged on a failed cycle.
        run_immediately: send a report on entry before the first 7-day sleep.
        max_cycles: stop after N cycles (tests only); ``None`` = forever.
    """
    await run_interval(
        bot.run_report,
        REPORT_INTERVAL_SECONDS,
        alerter=alerter,
        label="psycho weekly report",
        run_immediately=run_immediately,
        max_cycles=max_cycles,
    )
