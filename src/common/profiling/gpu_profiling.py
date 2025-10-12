# src/common/profiling/gpu_profiling.py
from __future__ import annotations
"""
GPU-specific profiling utilities using CUDA events.

This module provides:
  • CUDAProfiler: Single-GPU timing with CUDA events
  • MultiGPUProfiler: Coordinated timing across multiple GPUs
  • GPU memory tracking utilities
  • Transfer timing helpers

These profilers complement NVMLSampler by providing kernel-level timing
accuracy via CUDA events, which are more precise than CPU-side timers for
GPU operations.

Dependencies
------------
Requires CuPy. If unavailable, the profilers become no-ops that return
empty dictionaries.

Usage
-----
Single GPU:
    prof = CUDAProfiler(device_id=0)
    prof.section("compute", start=True)
    # ... GPU work ...
    prof.section("compute", start=False)
    timings = prof.get_timings()  # {"compute": 1.234}

Multi-GPU:
    prof = MultiGPUProfiler([0, 1, 2, 3])
    prof.section_all("compute", start=True)
    # ... work on all GPUs ...
    prof.section_all("compute", start=False)
    summary = prof.get_summary()  # Per-GPU breakdown + stats
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None  # type: ignore


class CUDAProfiler:
    """
    Single-GPU profiler using CUDA events for accurate kernel timing.

    CUDA events provide microsecond-precision timing that's synchronized with
    the GPU execution stream, making them more accurate than CPU-side timers
    for GPU operations.

    Parameters
    ----------
    device_id : int, default 0
        GPU device index to profile.
    stream : cupy.cuda.Stream | None, optional
        CUDA stream to use for events. If None, uses the default stream.

    Attributes
    ----------
    device_id : int
        GPU device being profiled.
    events : dict[str, cupy.cuda.Event]
        Recorded CUDA events by name.
    timings : dict[str, float]
        Computed elapsed times in seconds.
    """

    def __init__(self, device_id: int = 0, stream: Optional[Any] = None):
        """Initialize CUDA profiler for a single GPU."""
        if not CUPY_AVAILABLE:
            # Graceful degradation: profiler becomes a no-op
            self.device_id = device_id
            self.stream = None
            self.events = {}
            self.timings = {}
            self._active = False
            return

        self.device_id = device_id
        self.events: Dict[str, Any] = {}  # name -> cupy.cuda.Event
        self.timings: Dict[str, float] = {}
        self._active = True

        with cp.cuda.Device(device_id):
            self.stream = stream if stream is not None else cp.cuda.Stream()

    def mark(self, name: str) -> None:
        """
        Record a CUDA event at the current point in the stream.

        Parameters
        ----------
        name : str
            Event name/label.

        Notes
        -----
        - Events are non-blocking on the CPU.
        - Use section() for start/stop pairs with automatic timing.
        """
        if not self._active:
            return

        with cp.cuda.Device(self.device_id):
            event = cp.cuda.Event()
            event.record(self.stream)
            self.events[name] = event

    def section(self, name: str, *, start: bool) -> None:
        """
        Start or stop a named timing section.

        Parameters
        ----------
        name : str
            Section name/label.
        start : bool
            True to start timing, False to stop and compute elapsed time.

        Notes
        -----
        - Stopping a section synchronizes the end event and computes elapsed time.
        - Elapsed time is stored in self.timings[name] in seconds.
        """
        if not self._active:
            return

        if start:
            self.mark(f"{name}_start")
        else:
            self.mark(f"{name}_end")
            # Compute elapsed time
            start_event = self.events.get(f"{name}_start")
            end_event = self.events.get(f"{name}_end")

            if start_event and end_event:
                end_event.synchronize()  # Wait for GPU to complete
                elapsed_ms = cp.cuda.get_elapsed_time(start_event, end_event)
                self.timings[name] = elapsed_ms / 1000.0  # Convert to seconds

    def get_timings(self) -> Dict[str, float]:
        """
        Get all computed timing sections.

        Returns
        -------
        dict[str, float]
            Mapping of section name -> elapsed time (seconds).
        """
        return self.timings.copy()

    def synchronize(self) -> None:
        """
        Synchronize the profiler's stream (wait for all events to complete).
        """
        if self._active and self.stream:
            self.stream.synchronize()

    def write_json(self, path: Path) -> None:
        """
        Write timings to a JSON file.

        Parameters
        ----------
        path : Path
            Destination JSON path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "device_id": self.device_id,
            "timings": self.timings,
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")


class MultiGPUProfiler:
    """
    Coordinated profiling across multiple GPUs.

    This profiler manages per-GPU CUDAProfiler instances and provides
    utilities for synchronized timing across devices.

    Parameters
    ----------
    device_ids : list[int]
        List of GPU device indices to profile.

    Attributes
    ----------
    device_ids : list[int]
        GPU devices being profiled.
    gpu_profilers : dict[int, CUDAProfiler]
        Per-GPU profiler instances.
    """

    def __init__(self, device_ids: List[int]):
        """Initialize multi-GPU profiler."""
        self.device_ids = device_ids
        self.gpu_profilers: Dict[int, CUDAProfiler] = {}

        for dev_id in device_ids:
            self.gpu_profilers[dev_id] = CUDAProfiler(dev_id)

    def mark_all(self, name: str) -> None:
        """
        Record a CUDA event on all GPUs.

        Parameters
        ----------
        name : str
            Event name/label (same across all GPUs).
        """
        for prof in self.gpu_profilers.values():
            prof.mark(name)

    def section_all(self, name: str, *, start: bool) -> None:
        """
        Start or stop a timing section on all GPUs.

        Parameters
        ----------
        name : str
            Section name/label.
        start : bool
            True to start timing, False to stop.
        """
        for prof in self.gpu_profilers.values():
            prof.section(name, start=start)

    def section_on(self, device_id: int, name: str, *, start: bool) -> None:
        """
        Start or stop a timing section on a specific GPU.

        Parameters
        ----------
        device_id : int
            Target GPU device.
        name : str
            Section name/label.
        start : bool
            True to start timing, False to stop.
        """
        if device_id in self.gpu_profilers:
            self.gpu_profilers[device_id].section(name, start=start)

    def synchronize_all(self) -> None:
        """
        Synchronize all GPU streams (wait for completion on all devices).
        """
        for prof in self.gpu_profilers.values():
            prof.synchronize()

    def get_all_timings(self) -> Dict[int, Dict[str, float]]:
        """
        Get per-GPU timing breakdowns.

        Returns
        -------
        dict[int, dict[str, float]]
            Mapping: device_id -> {section_name: elapsed_seconds}
        """
        return {
            dev_id: prof.get_timings()
            for dev_id, prof in self.gpu_profilers.items()
        }

    def get_summary(self) -> Dict[str, Any]:
        """
        Aggregate timing statistics across all GPUs.

        Returns
        -------
        dict[str, Any]
            Summary containing:
              - Per-section statistics (min, max, mean, std)
              - Per-GPU timings for each section
              - Total time across all GPUs

        Example
        -------
        {
            "compute": {
                "min": 1.234,
                "max": 1.456,
                "mean": 1.345,
                "std": 0.098,
                "per_gpu": {0: 1.234, 1: 1.345, 2: 1.456, 3: 1.345}
            },
            "total_all_gpus": 5.380
        }
        """
        if not CUPY_AVAILABLE:
            return {}

        import numpy as np

        all_timings = self.get_all_timings()

        # Collect all unique section names
        sections = set()
        for timings in all_timings.values():
            sections.update(timings.keys())

        summary = {}
        total_time = 0.0

        for section in sections:
            times = [
                timings.get(section, float('nan'))
                for timings in all_timings.values()
            ]

            # Filter out NaNs for statistics
            valid_times = [t for t in times if not np.isnan(t)]

            if valid_times:
                summary[section] = {
                    "min": float(np.min(valid_times)),
                    "max": float(np.max(valid_times)),
                    "mean": float(np.mean(valid_times)),
                    "std": float(np.std(valid_times)),
                    "per_gpu": {
                        dev_id: timings.get(section, float('nan'))
                        for dev_id, timings in all_timings.items()
                    },
                }
                total_time += np.sum(valid_times)

        summary["total_all_gpus"] = total_time

        return summary

    def write_json(self, path: Path) -> None:
        """
        Write multi-GPU profiling summary to JSON.

        Parameters
        ----------
        path : Path
            Destination JSON path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "device_ids": self.device_ids,
            "summary": self.get_summary(),
            "per_gpu_timings": self.get_all_timings(),
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# GPU Memory Tracking Utilities
# ---------------------------------------------------------------------------

def get_memory_info(device_id: int = 0) -> Dict[str, int]:
    """
    Get current GPU memory usage.

    Parameters
    ----------
    device_id : int, default 0
        GPU device index.

    Returns
    -------
    dict[str, int]
        Memory info with keys:
          - "used": bytes currently allocated
          - "total": total device memory
          - "free": available memory
          - "cached": memory in CuPy's pool (if using memory pool)
    """
    if not CUPY_AVAILABLE:
        return {"used": 0, "total": 0, "free": 0, "cached": 0}

    with cp.cuda.Device(device_id):
        mempool = cp.get_default_memory_pool()
        used = mempool.used_bytes()
        total = cp.cuda.Device().mem_info[1]  # Total memory
        free = cp.cuda.Device().mem_info[0]   # Free memory

        return {
            "used": int(used),
            "total": int(total),
            "free": int(free),
            "cached": int(mempool.total_bytes()),
        }


def get_all_memory_info(device_ids: Optional[List[int]] = None) -> Dict[int, Dict[str, int]]:
    """
    Get memory info for multiple GPUs.

    Parameters
    ----------
    device_ids : list[int] | None
        GPU device indices. If None, queries all available GPUs.

    Returns
    -------
    dict[int, dict[str, int]]
        Mapping: device_id -> memory_info
    """
    if not CUPY_AVAILABLE:
        return {}

    if device_ids is None:
        device_ids = list(range(cp.cuda.runtime.getDeviceCount()))

    return {dev_id: get_memory_info(dev_id) for dev_id in device_ids}


@dataclass
class MemorySnapshot:
    """
    GPU memory snapshot at a point in time.

    Attributes
    ----------
    device_id : int
        GPU device index.
    label : str
        Snapshot label (e.g., "before", "after").
    used : int
        Bytes currently allocated.
    total : int
        Total device memory.
    free : int
        Available memory.
    cached : int
        Memory in CuPy's pool.
    """
    device_id: int
    label: str
    used: int
    total: int
    free: int
    cached: int

    @classmethod
    def capture(cls, device_id: int, label: str = "snapshot") -> "MemorySnapshot":
        """
        Capture a memory snapshot.

        Parameters
        ----------
        device_id : int
            GPU device to snapshot.
        label : str, default "snapshot"
            Label for this snapshot.

        Returns
        -------
        MemorySnapshot
            Captured memory state.
        """
        info = get_memory_info(device_id)
        return cls(
            device_id=device_id,
            label=label,
            **info
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary."""
        return {
            "device_id": self.device_id,
            "label": self.label,
            "used": self.used,
            "total": self.total,
            "free": self.free,
            "cached": self.cached,
        }


class MemoryTracker:
    """
    Track GPU memory usage over time with labeled snapshots.

    Usage
    -----
    tracker = MemoryTracker([0, 1, 2, 3])
    tracker.snapshot("before")
    # ... GPU work ...
    tracker.snapshot("after")
    delta = tracker.compute_delta("before", "after")
    tracker.write_json(path)
    """

    def __init__(self, device_ids: Optional[List[int]] = None):
        """
        Initialize memory tracker.

        Parameters
        ----------
        device_ids : list[int] | None
            GPU devices to track. If None, tracks all available GPUs.
        """
        if not CUPY_AVAILABLE:
            self.device_ids = []
            self.snapshots = {}
            self._active = False
            return

        if device_ids is None:
            device_ids = list(range(cp.cuda.runtime.getDeviceCount()))

        self.device_ids = device_ids
        self.snapshots: Dict[str, List[MemorySnapshot]] = {}
        self._active = True

    def snapshot(self, label: str) -> None:
        """
        Capture memory state on all devices with given label.

        Parameters
        ----------
        label : str
            Snapshot label (e.g., "before", "after", "peak").
        """
        if not self._active:
            return

        snapshots = [
            MemorySnapshot.capture(dev_id, label)
            for dev_id in self.device_ids
        ]
        self.snapshots[label] = snapshots

    def compute_delta(
        self, label_before: str, label_after: str
    ) -> Dict[int, Dict[str, int]]:
        """
        Compute memory usage delta between two snapshots.

        Parameters
        ----------
        label_before : str
            Label of earlier snapshot.
        label_after : str
            Label of later snapshot.

        Returns
        -------
        dict[int, dict[str, int]]
            Per-GPU delta with keys: "used_delta", "cached_delta"
        """
        if not self._active:
            return {}

        before = {s.device_id: s for s in self.snapshots.get(label_before, [])}
        after = {s.device_id: s for s in self.snapshots.get(label_after, [])}

        deltas = {}
        for dev_id in self.device_ids:
            b = before.get(dev_id)
            a = after.get(dev_id)

            if b and a:
                deltas[dev_id] = {
                    "used_delta": a.used - b.used,
                    "cached_delta": a.cached - b.cached,
                    "before_used": b.used,
                    "after_used": a.used,
                }

        return deltas

    def get_peak_usage(self) -> Dict[int, int]:
        """
        Get peak memory usage across all snapshots for each GPU.

        Returns
        -------
        dict[int, int]
            Mapping: device_id -> peak_used_bytes
        """
        if not self._active:
            return {}

        peaks = {}
        for dev_id in self.device_ids:
            max_used = 0
            for snapshots in self.snapshots.values():
                for s in snapshots:
                    if s.device_id == dev_id:
                        max_used = max(max_used, s.used)
            peaks[dev_id] = max_used

        return peaks

    def write_json(self, path: Path) -> None:
        """
        Write all snapshots and deltas to JSON.

        Parameters
        ----------
        path : Path
            Destination JSON path.
        """
        if not self._active:
            return

        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "device_ids": self.device_ids,
            "snapshots": {
                label: [s.to_dict() for s in snapshots]
                for label, snapshots in self.snapshots.items()
            },
            "peak_usage": self.get_peak_usage(),
        }

        # Add deltas if we have before/after
        if "before" in self.snapshots and "after" in self.snapshots:
            data["delta_before_after"] = self.compute_delta("before", "after")

        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

__all__ = [
    "CUDAProfiler",
    "MultiGPUProfiler",
    "MemoryTracker",
    "MemorySnapshot",
    "get_memory_info",
    "get_all_memory_info",
]