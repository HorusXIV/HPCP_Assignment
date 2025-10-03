"""Rank-aware logging utilities for the multi-GPU solver.

Provides simple, robust logging per MPI rank and optional rank-0 console
output. The implementation avoids background queue threads to minimize
shutdown hazards and keeps cluster logs tidy by default.

Environment variables:
    MULTIGPU_LOG_LEVEL: Global level name (e.g., INFO, DEBUG). Default:
        WARNING.
    MULTIGPU_VERBOSE: If set to an integer > 0, enables extra metrics and
        forces per-rank file logging.
    MULTIGPU_RANK_FILES: If set to "1", enables per-rank file logging.
    MULTIGPU_QUIET: If "1" (default) and level >= WARNING, suppress per-rank
        files unless overridden by MULTIGPU_VERBOSE or MULTIGPU_RANK_FILES.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from typing import Optional

_listener: Optional[logging.handlers.QueueListener] = None


class RankFilter(logging.Filter):
    """Attach the MPI rank to each log record.

    Args:
      rank: MPI rank used to populate ``record.rank`` when missing.
    """

    def __init__(self, rank: int):
        super().__init__()
        self.rank = rank

    def filter(self, record: logging.LogRecord) -> bool:
        record.rank = getattr(record, "rank", self.rank)
        return True


class ConsoleFilter(logging.Filter):
    """Filter console output to warnings/errors or explicitly general records.

    Messages with ``extra={"general": True}`` will be shown on rank 0 even
    if they are below WARNING.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        return bool(getattr(record, "general", False))


def _make_formatter():
    """Return a formatter that tolerates a missing ``rank`` attribute."""
    fmt = (
        "%(asctime)s - rank=%(rank)s - %(levelname)s - %(name)s - %(message)s"
    )

    class SafeFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            if not hasattr(record, "rank"):
                setattr(record, "rank", "-")
            return super().format(record)

    return SafeFormatter(fmt)


def verbose_enabled() -> bool:
    """Return whether verbose logging is enabled via MULTIGPU_VERBOSE."""
    try:
        return int(os.environ.get("MULTIGPU_VERBOSE", "0")) > 0
    except Exception:
        return False


def setup_logging(
    results_dir: str, rank: int = 0, size: int = 1, console: bool = True
) -> logging.Logger:
    """Initialize per-rank logging and optional rank-0 console output.

    Args:
      results_dir: Directory where a ``rank_logs/`` subfolder will be created
        when per-rank files are enabled.
      rank: MPI rank id.
      size: MPI world size. Reserved for API stability; not used.
      console: Whether to stream to stdout on rank 0.

    Returns:
      The configured root logger (root namespace).
    """
    os.makedirs(results_dir, exist_ok=True)
    root = logging.getLogger()

    for h in list(root.handlers):
        try:
            root.removeHandler(h)
        except Exception:
            pass

    _lvl_name = os.environ.get("MULTIGPU_LOG_LEVEL", "WARNING").upper()
    _lvl = getattr(logging, _lvl_name, logging.WARNING)
    root.setLevel(logging.NOTSET)
    root.addFilter(RankFilter(rank))

    quiet_mode = (
        _lvl >= logging.WARNING
        and os.environ.get("MULTIGPU_QUIET", "1") == "1"
    )
    want_rank_files = (
        os.environ.get("MULTIGPU_RANK_FILES", "0") == "1" or verbose_enabled()
    )
    if (not quiet_mode) or want_rank_files:
        per_rank_log = os.path.join(
            results_dir, "rank_logs", f"rank{rank:03d}.log"
        )
        os.makedirs(os.path.dirname(per_rank_log), exist_ok=True)
        fh_rank = logging.FileHandler(per_rank_log, mode="a")
        fh_rank.setLevel(_lvl)
        fh_rank.setFormatter(_make_formatter())
        fh_rank.addFilter(RankFilter(rank))
        root.addHandler(fh_rank)

    if console and rank == 0:
        sh = logging.StreamHandler(stream=sys.stdout)
        sh.setLevel(logging.INFO)
        sh.setFormatter(_make_formatter())
        sh.addFilter(RankFilter(rank))
        sh.addFilter(ConsoleFilter())
        root.addHandler(sh)

    return root


def shutdown_logging():
    """Flush and close handlers for a clean shutdown."""
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
