"""Rank-aware logging utilities for multiGPU package.

Provides a Queue-based logging setup so multiple processes (MPI ranks)
can safely emit log records without colliding. Creates per-rank log files
and an optional root aggregated log. Use `setup_logging` from processes
and call `shutdown_logging` on exit.
"""
from __future__ import annotations

import logging
import logging.handlers
import multiprocessing
import os
from typing import Optional

_listener: Optional[logging.handlers.QueueListener] = None


class RankFilter(logging.Filter):
    def __init__(self, rank: int):
        super().__init__()
        self.rank = rank

    def filter(self, record: logging.LogRecord) -> bool:
        # attach rank to record for formatting
        record.rank = getattr(record, "rank", self.rank)
        return True


def _make_formatter():
    fmt = (
        "%(asctime)s - rank=%(rank)s - %(levelname)s - "
        "%(name)s - %(message)s"
    )

    # Use a SafeFormatter that provides a default for missing fields like
    # `rank`.

    class SafeFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
            if not hasattr(record, "rank"):
                # attach a safe default if missing
                setattr(record, "rank", "-")
            return super().format(record)

    return SafeFormatter(fmt)


def setup_logging(
    results_dir: str, rank: int = 0, size: int = 1, console: bool = True
):
    """Initialize a process-safe queue logger for this process.

    - `results_dir`: directory to place `logs/` into (will be created)
    - `rank`: MPI rank integer
    - `size`: total MPI size (used to create aggregated log on rank 0)
    - `console`: whether to also emit to stderr (guarded per-rank)

    Returns: logger for convenience (root logger configured)
    """
    global _listener

    os.makedirs(results_dir, exist_ok=True)
    log_queue = multiprocessing.Queue(-1)

    # Create the queue listener only on rank 0 (aggregator)
    if rank == 0:
        handlers = []
        # aggregated log file
        agg_logfile = os.path.join(results_dir, "logs", "aggregated.log")
        os.makedirs(os.path.dirname(agg_logfile), exist_ok=True)
        fh = logging.FileHandler(agg_logfile, mode="a")
        fh.setFormatter(_make_formatter())
        handlers.append(fh)

        # Console handler for rank 0
        ch = logging.StreamHandler()
        ch.setFormatter(_make_formatter())
        handlers.append(ch)

        _listener = logging.handlers.QueueListener(log_queue, *handlers)
        _listener.start()

    # Configure per-process root logger to send to queue
    qh = logging.handlers.QueueHandler(log_queue)
    root = logging.getLogger()
    # Remove existing handlers to avoid duplicate logs in interactive use
    for h in list(root.handlers):
        root.removeHandler(h)

    root.setLevel(logging.INFO)
    # Ensure every record has a `rank` attribute. This prevents KeyError
    # when formatters expect the `rank` field.
    root.addFilter(RankFilter(rank))
    root.addHandler(qh)

    # Also add a per-rank file to capture this worker's logs locally
    per_rank_log = os.path.join(results_dir, "logs", f"rank{rank:03d}.log")
    os.makedirs(os.path.dirname(per_rank_log), exist_ok=True)
    fh_rank = logging.FileHandler(per_rank_log, mode="a")
    fh_rank.setFormatter(_make_formatter())
    # To ensure rank shows up in records, add a filter
    fh_rank.addFilter(RankFilter(rank))
    # Add the file handler as an additional handler that writes directly.
    # This bypasses the queue to help ensure process-local logs are written
    # even if the queue listener stops unexpectedly.
    root.addHandler(fh_rank)

    # Optional: console per rank (useful for debugging small runs)
    if console and rank != 0:
        ch = logging.StreamHandler()
        ch.setFormatter(_make_formatter())
        ch.addFilter(RankFilter(rank))
        root.addHandler(ch)

    return root


def shutdown_logging():
    """Shut down the queue listener (if started) and flush handlers."""
    global _listener
    root = logging.getLogger()
    # remove queue handler(s)
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

    if _listener is not None:
        try:
            _listener.stop()
        finally:
            _listener = None
