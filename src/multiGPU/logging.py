"""Rank-aware logging utilities for multiGPU package.

Lightweight per-rank file logging (no multiprocessing queues).
This avoids shutdown hangs on some platforms (notably Windows) that can
occur when `multiprocessing.Queue` feeder threads aren't joined.
Use `setup_logging` from each MPI rank and call `shutdown_logging` on exit.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from typing import Optional

_listener: Optional[logging.handlers.QueueListener] = None  # API compat


class RankFilter(logging.Filter):
    def __init__(self, rank: int):
        super().__init__()
        self.rank = rank

    def filter(self, record: logging.LogRecord) -> bool:
        # attach rank to record for formatting
        record.rank = getattr(record, "rank", self.rank)
        return True


class ConsoleFilter(logging.Filter):
    """Allow only warnings/errors or records explicitly marked as general.

    This keeps rank-specific chatter out of the console (general) log while
    still showing important warnings/errors. To mark a record as general,
    log with extra={"general": True}.
    """

    def filter(
        self, record: logging.LogRecord
    ) -> bool:  # type: ignore[override]
        if record.levelno >= logging.WARNING:
            return True
        return bool(getattr(record, "general", False))


def _make_formatter():
    fmt = (
        "%(asctime)s - rank=%(rank)s - %(levelname)s - "
        "%(name)s - %(message)s"
    )

    # Use a SafeFormatter that provides a default for missing fields like
    # `rank`.

    class SafeFormatter(logging.Formatter):
        def format(
            self, record: logging.LogRecord
        ) -> str:  # type: ignore[override]
            if not hasattr(record, "rank"):
                # attach a safe default if missing
                setattr(record, "rank", "-")
            return super().format(record)

    return SafeFormatter(fmt)


def setup_logging(
    results_dir: str, rank: int = 0, size: int = 1, console: bool = True
):
    """Initialize logging for this MPI rank (per-rank file, optional console).

    - `results_dir`: directory to place `logs/` into (will be created)
    - `rank`: MPI rank integer
    - `size`: total MPI size (unused; retained for API stability)
    - `console`: whether to also emit to stderr (rank 0 only)

    Returns: configured root logger
    """
    os.makedirs(results_dir, exist_ok=True)
    root = logging.getLogger()

    # Remove existing handlers to avoid duplicates in interactive sessions
    for h in list(root.handlers):
        try:
            root.removeHandler(h)
        except Exception:
            pass

    # Configure root log level (env override). Default to WARNING to keep
    # quiet.
    _lvl_name = os.environ.get("MULTIGPU_LOG_LEVEL", "WARNING").upper()
    _lvl = getattr(logging, _lvl_name, logging.WARNING)
    root.setLevel(_lvl)
    root.addFilter(RankFilter(rank))

    # Per-rank file handler (disabled in quiet mode unless forced)
    quiet_mode = (
        _lvl >= logging.WARNING
        and os.environ.get("MULTIGPU_QUIET", "1") == "1"
    )
    want_rank_files = os.environ.get("MULTIGPU_RANK_FILES", "0") == "1"
    if (not quiet_mode) or want_rank_files:
        per_rank_log = os.path.join(
            results_dir, "logs", f"rank{rank:03d}.log"
        )
        os.makedirs(os.path.dirname(per_rank_log), exist_ok=True)
        fh_rank = logging.FileHandler(per_rank_log, mode="a")
        fh_rank.setLevel(_lvl)
        fh_rank.setFormatter(_make_formatter())
        fh_rank.addFilter(RankFilter(rank))
        root.addHandler(fh_rank)

    # Optional console for rank 0 only (keeps cluster logs tidy)
    if console and rank == 0:
        # Stream to stdout (not stderr) so Slurm writes these into .out,
        # not .err
        sh = logging.StreamHandler(stream=sys.stdout)
        sh.setLevel(_lvl)
        sh.setFormatter(_make_formatter())
        sh.addFilter(RankFilter(rank))
        sh.addFilter(ConsoleFilter())
        root.addHandler(sh)

    return root


def shutdown_logging():
    """Flush and close all handlers.

    Keeping this simple ensures fast, clean shutdown without hanging
    on background threads.
    """
    root = logging.getLogger()
    for h in list(root.handlers):
        try:
            h.flush()
            h.close()
        except Exception:
            pass
        try:
            root.removeHandler(h)
        except Exception:
            pass
