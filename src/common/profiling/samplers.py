# src/common/profiling/samplers.py
from __future__ import annotations
"""
Lightweight background samplers for system and GPU metrics.

Two sampler types are provided:

- SystemSampler
    Samples driver-side metrics (CPU utilization, system/Process RSS, basic
    disk and network counters) at a fixed interval using `psutil`. If `psutil`
    is unavailable, the sampler becomes a no-op.

- NVMLSampler
    Samples per-GPU utilization and memory via NVIDIA's NVML (through
    `pynvml`). If NVML/pynvml are unavailable, the sampler becomes a no-op.

Both samplers:
- Run in a daemon thread started via `.start()` and stopped via `.stop()`.
- Accumulate dictionaries in `.rows`.
- Can persist their rows to CSV via `.write_csv(path)`.

These samplers are intentionally simple and best-effort: failures in optional
dependencies are silently ignored so that profiling does not break workloads.
"""

import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import contextlib

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # type: ignore

# Optional NVML (GPU) sampler
try:
    import pynvml  # type: ignore
except Exception:  # pragma: no cover
    pynvml = None  # type: ignore


@dataclass
class _SamplerBase:
    """
    Minimal base class for periodic background samplers.

    Attributes
    ----------
    interval : float
        Sampling interval in seconds.
    rows : list[dict]
        Collected measurements (append-only).
    _stop : threading.Event
        Internal event to request termination of the sampling loop.
    _thr : threading.Thread | None
        Background thread running the sampling loop, if started.
    """
    interval: float = 0.5
    rows: List[Dict] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _thr: Optional[threading.Thread] = field(default=None, init=False)

    def start(self) -> None:
        """
        Start the sampler thread (idempotent).
        """
        if self._thr is not None:
            return
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()

    def stop(self) -> None:
        """
        Stop the sampler thread and reset the internal state (idempotent).
        """
        if self._thr is None:
            return
        self._stop.set()
        self._thr.join(timeout=self.interval * 4)
        self._thr = None
        self._stop.clear()

    # Subclasses must implement:
    #   def _loop(self) -> None: ...
    #   def write_csv(self, path: Path) -> None: ...
    # `_loop` should honor `self._stop.is_set()` and sleep via `self._stop.wait()`.


class SystemSampler(_SamplerBase):
    """
    Driver-side system sampler (CPU%, memory, I/O, network).

    Dependencies
    ------------
    - Requires `psutil`. If unavailable, `_loop` returns immediately (no-op).

    Collected fields per sample
    ---------------------------
    ts : float
        UNIX timestamp (seconds).
    cpu_total_pct : float
        Total CPU utilization percentage (system-wide).
    mem_used_bytes, mem_total_bytes : int
        System memory usage/total.
    proc_rss_bytes : int
        Resident set size of the current process.
    disk_read_bytes, disk_write_bytes : int
        Cumulative disk I/O (best-effort; may be zero on unsupported platforms).
    net_bytes_sent, net_bytes_recv : int
        Cumulative network I/O (best-effort).
    """

    def _loop(self) -> None:
        if psutil is None:
            return
        p = psutil.Process()
        # Prime CPU percent (first call provides a baseline)
        psutil.cpu_percent(None)
        p.cpu_percent(None)
        while not self._stop.is_set():
            ts = time.time()
            cpu_tot = psutil.cpu_percent(None)
            mem = psutil.virtual_memory()
            rss = p.memory_info().rss
            io = None
            net = None
            try:
                io = psutil.disk_io_counters()
            except Exception:
                pass
            try:
                net = psutil.net_io_counters()
            except Exception:
                pass
            self.rows.append(
                {
                    "ts": ts,
                    "cpu_total_pct": float(cpu_tot),
                    "mem_used_bytes": int(mem.used),
                    "mem_total_bytes": int(mem.total),
                    "proc_rss_bytes": int(rss),
                    "disk_read_bytes": int(getattr(io, "read_bytes", 0) or 0),
                    "disk_write_bytes": int(getattr(io, "write_bytes", 0) or 0),
                    "net_bytes_sent": int(getattr(net, "bytes_sent", 0) or 0),
                    "net_bytes_recv": int(getattr(net, "bytes_recv", 0) or 0),
                }
            )
            self._stop.wait(self.interval)

    def write_csv(self, path: Path) -> None:
        """
        Persist collected rows to a CSV file.

        Parameters
        ----------
        path : pathlib.Path
            Destination path. Parent directories are created if needed.
        """
        if not self.rows:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(self.rows[0].keys())
        import csv

        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.rows)


class NVMLSampler(_SamplerBase):
    """
    Optional GPU sampler (utilization and memory per device).

    Dependencies
    ------------
    - Requires `pynvml` and a working NVML environment. If unavailable,
      `_loop` returns immediately (no-op).

    Parameters
    ----------
    device_indices : list[int] | None
        Subset of GPU indices to sample. If None, samples all visible GPUs.
    """

    def __init__(self, *args, device_indices: Optional[List[int]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.device_indices = device_indices
        self._handles: List = []

    def _loop(self) -> None:
        if pynvml is None:
            return
        try:
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            dev_ids = self.device_indices or list(range(count))
            handles = [
                pynvml.nvmlDeviceGetHandleByIndex(i) for i in dev_ids if i < count
            ]
            self._handles = handles
            while not self._stop.is_set():
                ts = time.time()
                for i, h in zip(dev_ids, handles):
                    util = pynvml.nvmlDeviceGetUtilizationRates(h)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                    self.rows.append(
                        {
                            "ts": ts,
                            "gpu_index": int(i),
                            "gpu_util_pct": float(util.gpu),
                            "mem_util_pct": float(util.memory),
                            "mem_used_bytes": int(mem.used),
                            "mem_total_bytes": int(mem.total),
                        }
                    )
                self._stop.wait(self.interval)
        finally:
            # Always attempt to shut down NVML, but suppress errors.
            with contextlib.suppress(Exception):  # type: ignore
                pynvml.nvmlShutdown()  # type: ignore

    def write_csv(self, path: Path) -> None:
        """
        Persist collected GPU rows to a CSV file.

        Parameters
        ----------
        path : pathlib.Path
            Destination path. Parent directories are created if needed.
        """
        if not self.rows:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(self.rows[0].keys())
        import csv

        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.rows)
