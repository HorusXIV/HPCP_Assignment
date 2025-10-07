"""Preemption and signal handling helpers.

Registers signal handlers that trigger a user callback (e.g., checkpoint) and
attempt a best-effort MPI barrier before exiting with a non-zero status so
job schedulers can resubmit.
"""

from __future__ import annotations
import signal
import sys
import threading
from typing import Callable, Optional

try:
    from mpi4py import MPI
except Exception:
    MPI = None


def _call_callback(cb: Callable[[], None], timeout: float = 30.0):
    """Invoke ``cb`` and suppress exceptions to avoid aborting the handler.

    The callback typically persists partial results or writes a marker file so
    schedulers can resume work later. Errors are printed and ignored.
    """
    try:
        cb()
    except Exception:
        # best-effort: suppress exceptions so the shutdown sequence continues
        try:
            import traceback

            traceback.print_exc()
        except Exception:
            pass


def register_preempt_handlers(
    callback: Callable[[], None], comm=None, signals: Optional[list] = None
):
    """Register handlers for preemption/termination signals.

    Args:
        callback: Zero-arg function invoked within the handler thread.
        comm: Optional MPI communicator used to attempt a barrier after the
            callback returns.
        signals: Optional list of ``signal.SIG*``; defaults to SIGTERM/SIGINT
            and SIGUSR1 when available.
    """
    if signals is None:
        # common scheduler signals: SIGTERM, SIGINT;
        signals = [signal.SIGTERM, signal.SIGINT]
        if hasattr(signal, "SIGUSR1"):
            signals.append(signal.SIGUSR1)

    def _handler(signum, frame):
        name = (
            signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
        )
        print(
            (
                f"[preempt] Received signal {signum} ({name}), "
                "running checkpoint callback..."
            ),
            file=sys.stderr,
        )
        t = threading.Thread(target=_call_callback, args=(callback,))
        t.start()
        t.join(timeout=30.0)
        try:
            if comm is not None:
                comm.Barrier()
        except Exception:
            pass
        sys.exit(2)

    for s in signals:
        try:
            signal.signal(s, _handler)
        except Exception:
            pass


if __name__ == "__main__":
    print("preempt helper loaded")
