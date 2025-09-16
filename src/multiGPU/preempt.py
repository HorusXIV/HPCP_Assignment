"""Preemption and signal handling helpers.

Register signal handlers that trigger checkpointing and graceful shutdown on
SIGTERM/SIGHUP and other platform-specific signals (e.g. SLURM's
SIGUSR1 for advanced preemption hooks).

Usage:
    from src.multiGPU.preempt import register_preempt_handlers

    def on_save():
        ck.save(state, step=step, async_write=False)

    register_preempt_handlers(on_save, comm=comm)

The function will attempt to call the supplied callback and then exit the
process with a non-zero code to notify the scheduler.
"""
from __future__ import annotations

import os
import signal
import sys
import threading
from typing import Callable, Optional

try:
    from mpi4py import MPI
except Exception:
    MPI = None


def _call_callback(cb: Callable[[], None], timeout: float = 30.0):
    try:
        cb()
    except Exception:
        # best-effort: suppress exceptions so the shutdown sequence continues
        try:
            import traceback
            traceback.print_exc()
        except Exception:
            pass


def register_preempt_handlers(callback: Callable[[], None], comm=None, signals: Optional[list] = None):
    """Register handlers for preemption signals.

    callback: function with no arguments that performs safe checkpointing.
    comm: optional MPI communicator; if provided this function will attempt
    to call `comm.Barrier()` after the callback to synchronize ranks.
    """
    if signals is None:
        # common scheduler signals: SIGTERM, SIGINT; SLURM may use SIGUSR1 for preemption
        signals = [signal.SIGTERM, signal.SIGINT]
        if hasattr(signal, 'SIGUSR1'):
            signals.append(signal.SIGUSR1)

    def _handler(signum, frame):
        name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
        print(f"[preempt] Received signal {signum} ({name}), running checkpoint callback...", file=sys.stderr)
        # call callback in a separate thread to avoid signal handler restrictions
        t = threading.Thread(target=_call_callback, args=(callback,))
        t.start()
        t.join(timeout=30.0)
        # if MPI available, attempt to barrier
        try:
            if comm is not None:
                comm.Barrier()
        except Exception:
            pass
        # exit with non-zero to indicate abnormal termination to scheduler
        sys.exit(2)

    for s in signals:
        try:
            signal.signal(s, _handler)
        except Exception:
            # some signals cannot be caught on certain platforms
            pass


if __name__ == "__main__":
    print("preempt helper loaded")
