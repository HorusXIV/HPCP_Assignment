"""Checkpoint manager for multiGPU workflows.

Provides atomic, optionally asynchronous saving of checkpoints. Supports
single-file and rank-sharded checkpoints for MPI/distributed runs.

Usage:
    from src.multiGPU.checkpoint import CheckpointManager

    ck = CheckpointManager(outdir="/scratch/you/checkpoints", keep=5, comm=comm, rank=rank)
    ck.save({'step': step, 'model': model_state_dict, 'opt': opt_state_dict})

The manager writes a temp file and renames it into place atomically so that
partial files are avoided. In MPI settings, the root rank can write a
master index while workers write per-rank shards.
"""
from __future__ import annotations

import os
import time
import glob
import tempfile
import shutil
import threading
import pickle
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

try:
    from mpi4py import MPI
except Exception:
    MPI = None


class CheckpointManager:
    def __init__(self, outdir: str, prefix: str = "ckpt", keep: int = 5, comm=None, rank: Optional[int] = None):
        self.outdir = os.path.abspath(outdir)
        os.makedirs(self.outdir, exist_ok=True)
        self.prefix = prefix
        self.keep = max(1, int(keep))
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._lock = threading.Lock()
        self.comm = comm if comm is not None else (MPI.COMM_WORLD if MPI is not None else None)
        if rank is None:
            try:
                self.rank = self.comm.Get_rank() if self.comm is not None else 0
            except Exception:
                self.rank = 0
        else:
            self.rank = rank

    def _filename(self, step: Optional[int] = None, rank: Optional[int] = None) -> str:
        ts = time.strftime("%Y%m%d_%H%M%S")
        parts = [self.prefix, ts]
        if step is not None:
            parts.append(f"step{int(step)}")
        if rank is not None:
            parts.append(f"rank{int(rank)}")
        name = "-".join(parts) + ".pkl"
        return os.path.join(self.outdir, name)

    def save(self, state: Dict[str, Any], step: Optional[int] = None, async_write: bool = True) -> str:
        """Save checkpoint. Returns the final checkpoint path.

        If async_write is True, schedule write in background and return the
        target path immediately. Caller should ensure program keeps running
        long enough for background write to complete or call `wait()`.
        """
        outpath = self._filename(step=step, rank=self.rank if self.comm is not None else None)
        tmpfd, tmppath = tempfile.mkstemp(prefix="tmp_ckpt_", dir=self.outdir)
        os.close(tmpfd)

        def _write():
            # ensure atomic write by writing to tmp then renaming
            try:
                with open(tmppath, "wb") as f:
                    # use pickle for general python objects; consumer must use pickle.load
                    pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(tmppath, outpath)
                self._prune()
            finally:
                # cleanup tmppath if it still exists
                if os.path.exists(tmppath):
                    try:
                        os.remove(tmppath)
                    except Exception:
                        pass

        if async_write:
            # schedule background write
            self._executor.submit(_write)
        else:
            _write()

        return outpath

    def wait(self, timeout: Optional[float] = None) -> None:
        """Block until background writes finish."""
        self._executor.shutdown(wait=True, timeout=timeout)
        # recreate executor for future saves
        self._executor = ThreadPoolExecutor(max_workers=1)

    def list_checkpoints(self) -> list:
        pattern = os.path.join(self.outdir, f"{self.prefix}-*.pkl")
        files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        return files

    def latest(self) -> Optional[str]:
        files = self.list_checkpoints()
        return files[0] if files else None

    def load(self, path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if path is None:
            path = self.latest()
        if path is None or not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj

    def _prune(self) -> None:
        # Keep the newest `self.keep` checkpoints, remove older ones
        files = self.list_checkpoints()
        if len(files) <= self.keep:
            return
        old = files[self.keep:]
        for p in old:
            try:
                os.remove(p)
            except Exception:
                pass


def example_usage():
    # Demonstration only; do not import MPI-heavy code in module import path
    ck = CheckpointManager(outdir="./checkpoints", keep=3)
    fake_state = {"step": 10, "model": {"w": [1, 2, 3]}, "optimizer": {"lr": 1e-3}}
    path = ck.save(fake_state, step=10, async_write=False)
    print("Saved:", path)


if __name__ == "__main__":
    example_usage()
