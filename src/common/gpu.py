# src/common/gpu.py
from __future__ import annotations
from contextlib import ContextDecorator
from typing import Optional

def available() -> bool:
    try:
        import cupy as cp  # noqa: F401
        return True
    except Exception:
        return False

def set_device(idx: int) -> None:
    import cupy as cp
    cp.cuda.Device(idx).use()

def sync() -> None:
    import cupy as cp
    cp.cuda.Stream.null.synchronize()

class CudaEventTimer(ContextDecorator):
    def __enter__(self):
        import cupy as cp
        self.cp = cp
        self._start = cp.cuda.Event(); self._end = cp.cuda.Event()
        self._start.record()
        self.seconds = None
        return self
    def __exit__(self, exc_type, exc, tb):
        self._end.record()
        self._end.synchronize()
        ms = self.cp.cuda.get_elapsed_time(self._start, self._end)
        self.seconds = ms / 1e3
        return False  # don't suppress exceptions

def cuda_event_timer():
    return CudaEventTimer()

_last_cuda_secs: Optional[float] = None
def last_cuda_elapsed() -> Optional[float]:
    return _last_cuda_secs

def pinned_empty(shape, dtype):
    import cupy as cp, numpy as np
    nbytes = np.dtype(dtype).itemsize * int(np.prod(shape))
    mem = cp.cuda.PinnedMemory().alloc(nbytes)
    arr = np.frombuffer(mem, dtype=dtype, count=int(np.prod(shape))).reshape(shape)
    arr._pinned_mem = mem   # attach handle to keep it alive
    return arr
