"""Rank-aware logging utilities for the multi-GPU solver.

Provides simple, robust logging per MPI rank without background queue
threads. This minimizes shutdown hazards on some platforms and keeps Slurm
outputs tidy by limiting console logs to rank 0.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from typing import Optional

_listener: Optional[logging.handlers.QueueListener] = None  # API compat


class RankFilter(logging.Filter):
    """Attach the MPI rank to log records.

    Args:
        rank: Integer MPI rank to record as ``record.rank``.
    """

    def __init__(self, rank: int):
        super().__init__()
        self.rank = rank

    def filter(self, record: logging.LogRecord) -> bool:
        # attach rank to record for formatting
        record.rank = getattr(record, "rank", self.rank)
        return True


class ConsoleFilter(logging.Filter):
    """Filter console messages to warnings/errors or "general" records.

    To mark a message as general (shown on rank 0 console), log with
    ``extra={"general": True}``.
    """

    def filter(
        self,
        record: logging.LogRecord
    ) -> bool:  # type: ignore[override]
        if record.levelno >= logging.WARNING:
            return True
        return bool(getattr(record, "general", False))


def _make_formatter():
    """Return a formatter that tolerates missing ``rank`` attribute."""
    fmt = (
        "%(asctime)s - rank=%(rank)s - %(levelname)s - "
        "%(name)s - %(message)s"
    )

    # Use a SafeFormatter that provides a default for missing fields like
    # `rank`.

    class SafeFormatter(logging.Formatter):
        def format(
                self,
                record: logging.LogRecord
        ) -> str:  # type: ignore[override]
            if not hasattr(record, "rank"):
                # attach a safe default if missing
                setattr(record, "rank", "-")
            return super().format(record)

    return SafeFormatter(fmt)


def setup_logging(
        results_dir: str,
        rank: int = 0,
        size: int = 1,
        console: bool = True
        ) -> logging.Logger:
    """Initialize per-rank logging and optional rank-0 console output.

    Args:
        results_dir: Directory under which a ``logs/`` subfolder will be
            created for per-rank logs when enabled.
        rank: MPI rank id.
        size: MPI world size (unused; reserved for API stability).
        console: If True, stream to stdout on rank 0.

    Returns:
        The configured root logger.
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
        per_rank_log = os.path.join(results_dir, "logs", f"rank{rank:03d}.log")
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
