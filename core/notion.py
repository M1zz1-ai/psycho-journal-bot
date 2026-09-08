"""Thin Python wrapper over the existing ``notion-cli`` tool via subprocess.

notion-cli (at ~/bin/notion-cli) already owns Notion auth and the Daily Task
Tracker schema — health is GREEN, creds configured. We do NOT reimplement
Notion auth here; we shell out and parse its ``--json`` output, so a future
unified agent can register these as tools.

Each function maps to ``notion-cli tasks <subcommand>`` and returns the parsed
JSON (dict/list). A non-zero exit raises NotionError with stderr.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from packaging.version import InvalidVersion, Version

from .errors import NotionError

logger = logging.getLogger(__name__)

DEFAULT_CLI = str(Path.home() / "bin" / "notion-cli")

# Lowest notion-cli this module's argv is known to work against. It lives HERE,
# next to the flags it guards, and not in config: the floor is a property of the
# code (0.2.1 is where --project/--dod, ISO datetimes and the corrected
# type/effort enums landed), so it must move in the same commit as the call that
# needs it. A config key would let an operator "fix" a mismatch by lowering the
# floor, which turns a loud deploy gap back into a silent runtime failure.
#
# 0.2.2 is not about a new flag: it is where the CLI began FORCING a timed --date
# into the configured zone instead of forwarding whatever offset it was handed.
# This module deliberately keeps no second copy of that logic — one enforcement
# point covers every writer, including the ones that never come through here (the
# growth-weekly routine, ad-hoc shell calls) — so the bot's correctness now
# depends on the CLI version, and the floor is what makes an un-updated host
# announce itself at boot instead of quietly writing UTC+3 times.
# 0.2.3 adds the backlog pair: ``tasks list --no-date`` and ``tasks update
# --clear-date``. Neither has a fallback here — an undated row cannot be listed
# by any date filter, and a date cannot be removed by sending any value — so a
# host left on 0.2.2 must announce itself at boot rather than fail a backlog
# command at the moment Bogdan uses it.
#
# 0.3.1 adds ``--date-end``: a task that spans days is ONE row with a Notion date
# range. No fallback is possible either — an older CLI would reject the flag, and
# quietly writing the start alone would turn a five-day task into a one-day one,
# which is the exact mistake the flag exists to stop.
MIN_CLI_VERSION = "0.3.1"

# `notion-cli --version` prints "notion-cli 0.2.1" (click's version_option with a
# custom message); the trailing-token match also covers click's default
# "notion-cli, version 0.2.1" if that message is ever restored.
_VERSION_RE = re.compile(r"(\d+(?:\.\d+)*(?:[.\-+_a-zA-Z0-9]*))\s*$")

FIX_HINT = "update notion-cli on the host, then re-run `notion-cli --version`"

# Daily Task Tracker "Type" multi-select. Canonical copy — mirrors
# notion_cli.config.VALID_TYPES (live-verified 2026-07-25); bots/notion/tools.py
# re-exports this one rather than keeping a second list to drift from.
TYPE_VALUES = ("Notion", "IT", "Franpos", "Sport", "Productivity", "English", "Trading")


def _type_flags(types: str | list[str] | None) -> list[str]:
    """Coerce the Type argument to a value list and validate it against the enum.

    A BARE STRING is one value, never an iterable of characters. Left alone,
    ``for t in "English"`` emitted ``--type E --type n --type g …`` — seven
    invalid flags from one correct value (live, 2026-07-25: the task was not
    created and the bot asked Bogdan to drop the type). Nothing here ever wants a
    string treated as a sequence, so the coercion is unambiguous and belongs at
    this boundary, where the shape stops being Python and becomes argv.

    Validation happens HERE rather than being left to notion-cli's ``click.Choice``
    for one reason: order. The split ran first, so the CLI's perfectly good error
    arrived about the letter 'E'. Checking the values before they reach argv means
    a wrong type fails as a wrong type, naming the ones that exist.

    Raises:
        NotionError: if any value is not in :data:`TYPE_VALUES`.
    """
    if types is None:
        return []
    values = [types] if isinstance(types, str) else list(types)
    unknown = [v for v in values if v not in TYPE_VALUES]
    if unknown:
        raise NotionError(
            f"invalid task type(s) {', '.join(repr(v) for v in unknown)} — "
            f"valid types are: {', '.join(TYPE_VALUES)}"
        )
    return [flag for value in values for flag in ("--type", value)]


def _positional(value: str, *, field: str) -> str:
    """Guard a value that becomes a BARE argv word rather than an option's value.

    click stops treating a token as a positional the moment it starts with ``-``:
    ``notion-cli projects add "--parent"`` is parsed as the ``--parent`` OPTION,
    swallowing whatever the next word is. Titles reach here from plan files and,
    in production, from automated task creation, so a leading dash is
    attacker-shaped input at an argv boundary — the same boundary ``_type_flags``
    already guards for the multi-select.

    Nothing is escaped: there is no shell (``subprocess`` gets a list), and a
    quoted dash is still a dash to click. The only correct answer is to refuse,
    here, naming the field — rather than to let click report a missing TITLE or
    an invalid ``--status`` choice for a value nobody passed. No real Notion
    title starts with a dash; if one ever must, it needs a notion-cli that reads
    it after a ``--`` terminator, not a caller that hopes.
    """
    if value.startswith("-"):
        raise NotionError(
            f"{field} may not start with '-' ({value!r}): notion-cli would read it "
            f"as a command-line option, not as a value"
        )
    return value


# ── transient-failure retry ───────────────────────────────────────────────────
# Measured in production 2026-08-22/23: about 1% of polls came back
# "Notion unreachable: ConnectError(…[Errno 104] Connection reset by peer)" while
# the network was demonstrably fine — the next call answered in 0.4s. Every blip
# raised NotionError, and for the slot pinger that meant a "⚠️ slot pinger
# failed" in Telegram (6 errors / 3 alerts a day). A dropped TCP connection is
# not news; it is what a retry is for.
#
# The retry lives HERE, in the one wrapper every caller goes through (pinger,
# digest, brief, core.project_plan), rather than in any one of them — one blip
# class, one place that absorbs it.
#
# It is deliberately narrow. Only a failure whose *text* names a transport
# problem is retried; "project 'X' not found" is an answer, and repeating a
# question Notion already answered only delays the message that tells Bogdan
# what to fix. Same for a missing binary and for output that is not JSON — those
# do not heal in five seconds.
RETRY_ATTEMPTS = 3

# Waits BETWEEN attempts, so len() is RETRY_ATTEMPTS - 1. Worst case a caller
# blocks ~7s longer than today. That is affordable: every call site is either a
# 5-minute poll or a chat turn that is already waiting on an HTTP round trip.
RETRY_BACKOFF_SECONDS = (2.0, 5.0)

# Matched case-insensitively as substrings of the failure reason.
_TRANSIENT_MARKERS = (
    "notion unreachable",
    "connecterror",
    "connecttimeout",
    "readerror",
    "readtimeout",
    "connectionreset",
    "connection reset",
    "connection aborted",
    "errno 104",
    "timed out",
    "timeout",
    "temporarily unavailable",
)


# Verbs that only READ. The retry applies to these and to nothing else.
#
# notion-cli has no idempotency key, and _is_transient decides on the failure
# TEXT — which cannot tell "the request never left" from "the reply never came
# back". httpx's timeout is 30s, so a ReadTimeout means the POST was delivered
# and the answer was lost: retrying a write there creates the row a second and a
# third time, silently, while the caller sees success. Reads have no such cost,
# and reads are where the measured 1% blip lives.
_READ_ONLY_VERBS = frozenset({"list", "get", "today", "stats"})

# notion-cli's created-but-incomplete sentence, e.g.
# "Task created (abc123) but the DoD could not be written: ReadTimeout …".
# The CLI names the page id precisely so a caller will NOT re-run
# (notion_cli/commands/tasks.py: "The task exists; say so, or the retry creates
# a duplicate") — but the same sentence carries the words "timeout"/"unreachable"
# from the wrapped exception, so the text classifier reads it as a blip. A reason
# that names a page id is an answer, whatever verb produced it.
_CREATED_BUT_INCOMPLETE = re.compile(r"created\s*\([^)]+\)\s*but", re.IGNORECASE)


def _sleep(seconds: float) -> None:
    """Indirection so tests can watch the backoff without living through it."""
    time.sleep(seconds)


def _is_transient(reason: str) -> bool:
    """True when a failure reason reads like a transport blip rather than an answer."""
    lowered = reason.lower()
    return any(marker in lowered for marker in _TRANSIENT_MARKERS)


def _is_retryable(reason: str, verb: str) -> bool:
    """True when this failure, from THIS verb, may safely be attempted again.

    Two gates, and both must pass:

    1. The verb only reads (:data:`_READ_ONLY_VERBS`). A mutating call gets one
       attempt and the failure goes up, as it did before the retry existed.
    2. The reason reads like transport, and is not notion-cli reporting a row it
       already created.
    """
    if verb not in _READ_ONLY_VERBS:
        return False
    if _CREATED_BUT_INCOMPLETE.search(reason):
        return False
    return _is_transient(reason)


def _run(cli: str, args: list[str], *, group: str = "tasks") -> Any:
    """Run ``<cli> <group> <args> --json`` and return parsed JSON.

    ``group`` selects the notion-cli command group ("tasks" or "habits").

    A failure that names a transport problem is retried up to
    :data:`RETRY_ATTEMPTS` times with :data:`RETRY_BACKOFF_SECONDS` between
    attempts **only when the verb reads** (:data:`_READ_ONLY_VERBS`); a mutating
    verb gets exactly one attempt. Anything else is raised on the first attempt.

    Raises:
        NotionError: on non-zero exit or unparseable output.
    """
    cmd = [cli, group, *args, "--json"]
    label = " ".join(args)
    verb = args[0] if args else ""
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        except FileNotFoundError as exc:
            raise NotionError(f"notion-cli not found at {cli}") from exc
        if proc.returncode == 0:
            break
        reason = _failure_reason(proc)
        last = attempt == RETRY_ATTEMPTS
        if not _is_retryable(reason, verb) or last:
            raise NotionError(f"notion-cli {label} failed ({proc.returncode}): {reason}")
        wait = RETRY_BACKOFF_SECONDS[attempt - 1]
        logger.warning(
            "notion-cli %s: transient failure on attempt %d/%d (%s) — retrying in %.0fs",
            label,
            attempt,
            RETRY_ATTEMPTS,
            reason,
            wait,
        )
        _sleep(wait)
    out = proc.stdout.strip()
    if not out:
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError as exc:
        raise NotionError(f"notion-cli {' '.join(args)} returned non-JSON: {exc}") from exc


def _failure_reason(proc: subprocess.CompletedProcess[str]) -> str:
    """Best available explanation for a non-zero notion-cli exit.

    In ``--json`` mode notion-cli reports failures as ``{"ok": false, "error":
    ...}`` on STDOUT and leaves stderr EMPTY (see its ``_fail``/``emit``). Since
    every call here passes ``--json``, reading stderr alone yields "failed (1): "
    with no reason — which is exactly what a caller sees when a project name does
    not resolve, the one error a user can actually fix. So stdout's error field
    is preferred, with stderr as the fallback for failures that never reach the
    JSON path (bad flags, a traceback).
    """
    try:
        payload = json.loads(proc.stdout.strip())
    except ValueError:  # JSONDecodeError subclasses ValueError
        payload = None
    if isinstance(payload, dict) and payload.get("error"):
        return str(payload["error"])
    return proc.stderr.strip() or "no error message"


@dataclass(frozen=True)
class VersionCheck:
    """Outcome of comparing the notion-cli on disk against :data:`MIN_CLI_VERSION`.

    Attributes:
        ok: True only when a version was read AND it satisfies the floor.
        found: The version string read from the CLI, or None if none could be.
        required: The minimum that was demanded.
        reason: One of ``ok``/``too_old``/``missing``/``unavailable``/``unparseable`` —
            the machine-readable half; each maps to a different operator fix.
        message: One operator-facing line, ready for a log record or an alert.
    """

    ok: bool
    found: str | None
    required: str
    reason: str
    message: str


def cli_version(cli: str = DEFAULT_CLI) -> str:
    """Return the version string reported by ``<cli> --version``.

    Raises:
        NotionError: if the binary is missing, exits non-zero, or prints
            something no version can be read out of.
    """
    try:
        proc = subprocess.run([cli, "--version"], capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise NotionError(f"notion-cli not found at {cli}") from exc
    if proc.returncode != 0:
        raise NotionError(f"notion-cli --version failed ({proc.returncode}): {proc.stderr.strip()}")
    first_line = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""
    match = _VERSION_RE.search(first_line)
    if not match:
        raise NotionError(f"notion-cli --version output not parseable: {first_line!r}")
    return match.group(1)


def check_cli_version(cli: str = DEFAULT_CLI, minimum: str = MIN_CLI_VERSION) -> VersionCheck:
    """Compare the installed notion-cli against ``minimum``. Never raises.

    Comparison goes through ``packaging.version.Version``, not string ordering —
    a lexical compare puts 0.10.0 *below* 0.2.1 and would wave through exactly
    the stale host this check exists to catch.
    """
    try:
        found = cli_version(cli)
    except NotionError as exc:
        text = str(exc)
        if "not found" in text:
            reason = "missing"
        elif "not parseable" in text:
            reason = "unparseable"
        else:
            reason = "unavailable"
        return VersionCheck(
            ok=False,
            found=None,
            required=minimum,
            reason=reason,
            message=f"{text} (requires >= {minimum}) — {FIX_HINT}",
        )

    try:
        satisfied = Version(found) >= Version(minimum)
    except InvalidVersion:
        return VersionCheck(
            ok=False,
            found=None,
            required=minimum,
            reason="unparseable",
            message=(
                f"notion-cli --version output not parseable: {found!r} "
                f"(requires >= {minimum}) — {FIX_HINT}"
            ),
        )

    if satisfied:
        return VersionCheck(
            ok=True,
            found=found,
            required=minimum,
            reason="ok",
            message=f"notion-cli {found} satisfies the required >= {minimum}",
        )
    return VersionCheck(
        ok=False,
        found=found,
        required=minimum,
        reason="too_old",
        message=(
            f"notion-cli too old: found {found} at {cli}, requires >= {minimum} — "
            f"tasks using --project/--dod, ISO datetimes or the corrected "
            f"type/effort enums WILL fail; {FIX_HINT}"
        ),
    )


def add_task(
    title: str,
    *,
    date: str | None = None,
    date_end: str | None = None,
    types: str | list[str] | None = None,
    effort: str | None = None,
    status: str | None = None,
    project: str | None = None,
    dod: str | None = None,
    cli: str = DEFAULT_CLI,
) -> Any:
    """Add a task. ``types`` maps to repeatable ``--type`` flags.

    ``types`` is a LIST of :data:`TYPE_VALUES` (Type is a multi-select). A single
    value may be passed as a bare string — it is one value, not seven letters —
    and an unknown value raises before anything is spawned. Same argument, same
    name and same shape as :func:`update_task`.

    ``project`` is an exact Project Tracker name (notion-cli resolves it to the
    relation); ``dod`` becomes a paragraph in the page body — there is no DoD
    property in the schema, so the body is the only place it can live.

    ``date_end`` makes the row MULTI-DAY: the Notion Date becomes an inclusive
    range. It needs a ``date`` beside it (the CLI exits 2 otherwise) — a range
    with no beginning is not a range.
    """
    args = ["add", _positional(title, field="task title")]
    if date:
        args += ["--date", date]
    if date_end:
        args += ["--date-end", date_end]
    args += _type_flags(types)
    if effort:
        args += ["--effort", effort]
    if status:
        args += ["--status", status]
    if project:
        args += ["--project", project]
    if dod:
        args += ["--dod", dod]
    return _run(cli, args)


def list_tasks(
    *,
    today: bool = False,
    date: str | None = None,
    status: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    no_date: bool = False,
    cli: str = DEFAULT_CLI,
) -> Any:
    """List tasks, optionally filtered by today/date/status, a date range, or no date.

    ``date_from``/``date_to`` map to notion-cli's ``--date-from``/``--date-to``,
    which push a server-side Date range filter (no client-side 100-row cap).

    ``no_date`` is the BACKLOG: rows with no Date at all — a project's road to
    closure, from which a day is assembled. It contradicts every other date
    filter and notion-cli refuses the combination rather than picking one.
    """
    args = ["list"]
    if no_date:
        args.append("--no-date")
    if today:
        args.append("--today")
    if date:
        args += ["--date", date]
    if date_from:
        args += ["--date-from", date_from]
    if date_to:
        args += ["--date-to", date_to]
    if status:
        args += ["--status", status]
    return _run(cli, args)


def get_task(page_id: str, *, cli: str = DEFAULT_CLI) -> Any:
    """Get a single task by page id."""
    return _run(cli, ["get", page_id])


def update_task(
    page_id: str,
    *,
    title: str | None = None,
    status: str | None = None,
    date: str | None = None,
    date_end: str | None = None,
    types: str | list[str] | None = None,
    effort: str | None = None,
    project: str | None = None,
    dod: str | None = None,
    cli: str = DEFAULT_CLI,
) -> Any:
    """Update one or more fields of a task (``project``/``dod`` as in add_task).

    ``types`` REPLACES the task's Type multi-select and takes the same shape as
    :func:`add_task`'s. It was ``type_: str`` until 2026-07-25 — one concept with
    two names and two shapes across two functions is how a caller ends up passing
    the other one's shape, which is exactly the bug this pair was fixed for.

    ``date`` and ``date_end`` are ONE property at the far end: a PATCH replaces
    the whole Notion Date, so ``date`` alone drops a range's end. A caller that
    moves a multi-day task passes both — which is what
    ``core.project_plan.shift`` does.
    """
    args = ["update", _positional(page_id, field="page id")]
    if title:
        args += ["--title", title]
    if status:
        args += ["--status", status]
    if date:
        args += ["--date", date]
    if date_end:
        args += ["--date-end", date_end]
    args += _type_flags(types)
    if effort:
        args += ["--effort", effort]
    if project:
        args += ["--project", project]
    if dod:
        args += ["--dod", dod]
    return _run(cli, args)


def clear_task_date(page_id: str, *, cli: str = DEFAULT_CLI) -> Any:
    """Remove a task's Date entirely — send the row back to the backlog.

    Separate from :func:`update_task` because "no date" is not a date value:
    ``update_task(date="")`` means "leave the Date alone" here, and an empty
    ``start`` is a 400 at the API. notion-cli >= 0.2.3 expresses it as
    ``--clear-date`` (``{"date": null}``).

    ONE row moves; nothing is created or deleted. Same contract as every other
    move in this loop — a clone is the failure this codebase keeps designing
    against.
    """
    return _run(cli, ["update", _positional(page_id, field="page id"), "--clear-date"])


def complete_task(page_id_or_title: str, *, cli: str = DEFAULT_CLI) -> Any:
    """Mark a task Done (accepts page id or fuzzy title)."""
    return _run(cli, ["complete", page_id_or_title])


def delete_task(page_id: str, *, cli: str = DEFAULT_CLI) -> Any:
    """Archive (soft-delete) a task by page id. ``--yes`` skips confirmation."""
    return _run(cli, ["delete", _positional(page_id, field="page id"), "--yes"])


# ── PROJECT TRACKER group (notion-cli projects <sub>) ─────────────────────────
# A THIRD Notion database, related to tasks by the Project relation. Read-only
# here on purpose — see bots/notion/tools.py for the reasoning.


def list_projects(*, cli: str = DEFAULT_CLI) -> Any:
    """List every project in the Project Tracker (id, title, status, progress…)."""
    return _run(cli, ["list"], group="projects")


def get_project(page_id: str, *, cli: str = DEFAULT_CLI) -> Any:
    """Get a single project by page id."""
    return _run(cli, ["get", _positional(page_id, field="page id")], group="projects")


# The write half of the Project Tracker exists for ONE caller: core.project_plan,
# which turns a reviewed local plan file into projects and tasks. The bot's own
# tools stay read-only (bots/notion/tools.py) — projects are created from a
# reviewed plan, never as a side effect of a chat turn.


def add_project(
    title: str,
    *,
    status: str | None = None,
    priority: str | None = None,
    effort: str | None = None,
    deadline: str | None = None,
    parent: str | None = None,
    cli: str = DEFAULT_CLI,
) -> Any:
    """Create a project. ``priority`` is the LIVE select value — 'HIgh', typo included.

    ``parent`` is an exact project title; notion-cli resolves it to the
    parent/sub-item relation. ``effort`` is the PROJECT enum
    (Small/Medium/Large/Extreme), not the task one (Low/Medium/High/Extreme).
    """
    args = ["add", _positional(title, field="project title")]
    if status:
        args += ["--status", status]
    if priority:
        args += ["--priority", priority]
    if effort:
        args += ["--effort", effort]
    if deadline:
        args += ["--deadline", deadline]
    if parent:
        args += ["--parent", parent]
    return _run(cli, args, group="projects")


def update_project(
    page_id: str,
    *,
    title: str | None = None,
    status: str | None = None,
    priority: str | None = None,
    effort: str | None = None,
    deadline: str | None = None,
    parent: str | None = None,
    cli: str = DEFAULT_CLI,
) -> Any:
    """Update fields of an existing project — including a rename via ``title``.

    A rename goes through here rather than through create+archive: the Project
    relation on every task points at this page id, and re-creating the project
    would orphan all of them.
    """
    args = ["update", _positional(page_id, field="page id")]
    if title:
        args += ["--title", title]
    if status:
        args += ["--status", status]
    if priority:
        args += ["--priority", priority]
    if effort:
        args += ["--effort", effort]
    if deadline:
        args += ["--deadline", deadline]
    if parent:
        args += ["--parent", parent]
    return _run(cli, args, group="projects")


def archive_project(page_id: str, *, cli: str = DEFAULT_CLI) -> Any:
    """Archive (soft-delete) a project by page id."""
    return _run(cli, ["archive", _positional(page_id, field="page id"), "--yes"], group="projects")


# ── HABIT TRACKER group (notion-cli habits <sub>) ─────────────────────────────
# A SEPARATE Notion database (one row per day, a checkbox per habit) — not the
# task tracker. Slugs live in notion-cli's config.HABIT_PROPS.


def list_habits_today(*, cli: str = DEFAULT_CLI) -> Any:
    """Return today's habit row (list of flattened rows, possibly empty)."""
    return _run(cli, ["today"], group="habits")


def check_habit(
    habit: str,
    *,
    off: bool = False,
    date: str | None = None,
    cli: str = DEFAULT_CLI,
) -> Any:
    """Set (or clear with ``off=True``) a habit checkbox for a day.

    ``habit`` is a slug like ``cold``/``training``/``wake-up``. ``date`` is
    YYYY-MM-DD (default: today). The day's row is created by notion-cli if it
    does not exist yet.
    """
    args = ["check", habit]
    if date:
        args += ["--date", date]
    if off:
        args.append("--off")
    return _run(cli, args, group="habits")


def habit_stats(*, days: int = 7, cli: str = DEFAULT_CLI) -> Any:
    """Return density % per habit over the last ``days`` (missing day = not done)."""
    return _run(cli, ["stats", "--days", str(days)], group="habits")
