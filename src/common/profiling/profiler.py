# src/common/profiling/profiler.py
from __future__ import annotations

"""
Cross-backend profiling utilities.

This module implements a lightweight, backend-agnostic profiler that can be used
by the baseline runner, GPU runners, and the Dask runners alike. It collects:

- Timings via explicit `mark()` and `section()` calls
- Optional Dask performance report (HTML) when a `Client` is provided
- Dask task stream events (CSV) and simple aggregations (CSV)
- Dask worker metrics snapshots (before/after computation)
- System time series sampler (CPU/RAM) and optional GPU (NVML) sampler
- Environment and versions snapshots (JSON)
- Optional GPU timing integration (CUDA events via external profiler)

Artifacts are written into the benchmark directory using a timestamp to
disambiguate runs.

Usage Patterns
--------------

Baseline (NumPy):
    with Profiler(None, benchdir, stamp) as prof:
        prof.section("compute", start=True)
        result = baseline_solver(...)
        prof.section("compute", start=False)

Single GPU (CuPy):
    with Profiler(None, benchdir, stamp, enable_gpu_sampler=True) as prof:
        prof.section("compute", start=True)
        result = gpu_solver(...)
        prof.section("compute", start=False)
        # Optionally register external GPU timings
        prof.register_gpu_timings(cuda_profiler.get_timings())

Dask Multi-Node:
    client = Client()
    with Profiler(client, benchdir, stamp) as prof:
        prof.snapshot_workers("before")  # Capture initial state
        with prof.compute_context():
            result = client.compute(lazy_result)  # Work happens here!
            result = result.result()
        prof.snapshot_workers("after")  # Capture final state
"""

import os
import json
import time
import platform
from pathlib import Path
from typing import Dict, Any, Optional, Iterable, Tuple
from contextlib import contextmanager, ExitStack

try:
    from dask.distributed import performance_report, get_task_stream, Client

    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    Client = None  # type: ignore

from .samplers import SystemSampler, NVMLSampler
from .task_agg import aggregate_task_stream, write_task_csv, write_agg_csv


def _versions_snapshot() -> Dict[str, Any]:
    """
    Capture versions of commonly relevant packages.

    Returns
    -------
    dict
        Mapping of package name → version string (or None if unavailable).
    """

    def _ver(modname: str):
        try:
            m = __import__(modname)
            return getattr(m, "__version__", None) or getattr(m, "version", None)
        except Exception:
            return None

    return {
        "python": platform.python_version(),
        "numpy": _ver("numpy"),
        "scipy": _ver("scipy"),
        "dask": _ver("dask"),
        "distributed": _ver("distributed"),
        "numba": _ver("numba"),
        "cupy": _ver("cupy"),
    }


def _driver_env_snapshot() -> Dict[str, Any]:
    """
    Capture a concise snapshot of the driver environment.

    Includes platform details, threading-related env vars, package versions,
    and (if psutil is available) CPU counts and total memory.

    Returns
    -------
    dict
        Environment snapshot.
    """
    out: Dict[str, Any] = {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "python": platform.python_version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "env_caps": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
        },
        "versions": _versions_snapshot(),
    }
    try:
        import psutil  # type: ignore

        out["cpu"] = {
            "count_logical": psutil.cpu_count(logical=True),
            "count_physical": psutil.cpu_count(logical=False),
        }
        vm = psutil.virtual_memory()
        out["memory"] = {"total_bytes": int(vm.total)}
    except Exception:
        # psutil is optional
        pass
    return out


def _extract_worker_metrics(client) -> Dict[str, Dict[str, Any]]:
    """
    Extract per-worker statistics from a Dask client.

    Parameters
    ----------
    client : dask.distributed.Client
        Active Dask client.

    Returns
    -------
    dict
        Mapping: worker_address -> {
            "nthreads": int,
            "memory_limit": int,
            "memory_used": int,
            "task_count": int,
        }
    """
    if not DASK_AVAILABLE or client is None:
        return {}

    try:
        scheduler_info = client.scheduler_info()
        workers = scheduler_info.get("workers", {})

        metrics = {}
        for addr, info in workers.items():
            metrics[addr] = {
                "nthreads": info.get("nthreads", 0),
                "memory_limit": info.get("memory_limit", 0),
                "memory_used": info.get("metrics", {}).get("memory", 0),
                "task_count": len(info.get("processing", {})),
            }
        return metrics
    except Exception:
        return {}


def _write_worker_metrics_csv(metrics: Dict[str, Dict[str, Any]], path: Path) -> None:
    """
    Write worker metrics to CSV.

    Parameters
    ----------
    metrics : dict
        Output of `_extract_worker_metrics`.
    path : Path
        Destination CSV path.
    """
    if not metrics:
        return

    import csv

    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for addr, m in metrics.items():
        row = {"worker": addr, **m}
        rows.append(row)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


class Profiler:
    """
    Lightweight profiler/context manager for CPU, GPU, and Dask runs.

    Responsibilities
    ----------------
    - Manage timing sections and ad-hoc marks
    - Optionally emit a Dask performance report (HTML)
    - Collect Dask task stream events and write raw/aggregated CSVs
    - Snapshot Dask worker metrics before/after computation
    - Run background system/GPU samplers and persist their time series
    - Write environment/versions/run JSON reports
    - Accept external GPU timings from CUDA event profilers

    Usage
    -----
    >>> with Profiler(client, benchdir, stamp) as prof:
    ...     with prof.compute_context():
    ...         # do work (both perf report + task stream captured)
    ...     prof.section("compute", start=True)
    ...     # ... work ...
    ...     prof.section("compute", start=False)
    """

    def __init__(
            self,
            client: Optional[Any],  # dask.distributed.Client or None
            benchdir: Path,
            stamp: str,
            *,
            enable_perf_html: bool = True,
            enable_task_stream: bool = True,
            enable_worker_snapshots: bool = True,
            enable_system_sampler: bool = True,
            enable_gpu_sampler: bool = False,
    ):
        """
        Parameters
        ----------
        client : dask.distributed.Client | None
            If provided, enables Dask-specific features (perf report, task stream,
            worker snapshots).
        benchdir : Path
            Directory where artifacts will be written.
        stamp : str
            Unique identifier (e.g., timestamp) for naming artifacts.
        enable_perf_html : bool, default True
            Emit Dask performance report HTML (requires `client`).
        enable_task_stream : bool, default True
            Capture Dask task stream events (requires `client`).
        enable_worker_snapshots : bool, default True
            Snapshot Dask worker metrics before/after (requires `client`).
        enable_system_sampler : bool, default True
            Record driver CPU/memory timeseries via `SystemSampler`.
        enable_gpu_sampler : bool, default False
            Record GPU timeseries via `NVMLSampler` (if NVML is available).
        """
        self.client = client
        self.benchdir = Path(benchdir)
        self.stamp = stamp
        self.enable_perf_html = bool(enable_perf_html)
        self.enable_task_stream = bool(enable_task_stream)
        self.enable_worker_snapshots = bool(enable_worker_snapshots)
        self.enable_system_sampler = bool(enable_system_sampler)
        self.enable_gpu_sampler = bool(enable_gpu_sampler)

        # file paths (backend-neutral names)
        self.perf_html = self.benchdir / f"performance_report_{self.stamp}.html"
        self.tasks_csv = self.benchdir / f"tasks_{self.stamp}.csv"
        self.tasks_agg_csv = self.benchdir / f"tasks_agg_{self.stamp}.csv"
        self.sys_ts_csv = self.benchdir / f"system_timeseries_{self.stamp}.csv"
        self.run_json = self.benchdir / f"run_report_{self.stamp}.json"
        self.env_json = self.benchdir / f"env_{self.stamp}.json"

        # timings
        self._t0 = time.perf_counter()
        self._marks: Dict[str, float] = {}
        self._sections: Dict[str, Tuple[float, float]] = {}

        # External GPU timings (registered via register_gpu_timings)
        self._gpu_timings: Dict[str, Any] = {}

        # Worker snapshots
        self._worker_snapshots: Dict[str, Dict[str, Dict[str, Any]]] = {}

        # samplers
        self.sys_sampler = (
            SystemSampler(interval=0.5) if self.enable_system_sampler else None
        )
        self.gpu_sampler = (
            NVMLSampler(interval=0.5) if self.enable_gpu_sampler else None
        )

        self.benchdir.mkdir(parents=True, exist_ok=True)

    # --- timing helpers -------------------------------------------------------
    def mark(self, name: str) -> None:
        """
        Record an ad-hoc timestamp mark.

        Parameters
        ----------
        name : str
            Label for the mark.
        """
        self._marks[name] = time.perf_counter()

    def section(self, name: str, *, start: bool | None = None) -> None:
        """
        Start/stop/toggle a named timing section.

        Parameters
        ----------
        name : str
            Section label.
        start : bool | None
            - True  → start a section (overwrite any previous start)
            - False → stop the section and record elapsed time
            - None  → toggle: start if not started, stop otherwise
        """
        if start is True:
            self._sections[name] = (time.perf_counter(), 0.0)
        elif start is False:
            t1 = time.perf_counter()
            t0, _ = self._sections.get(name, (t1, 0.0))
            self._sections[name] = (t0, t1 - t0)
        else:
            # toggle
            if name not in self._sections or self._sections[name][1] > 0:
                self.section(name, start=True)
            else:
                self.section(name, start=False)

    def register_gpu_timings(self, timings: Dict[str, Any]) -> None:
        """
        Register external GPU timings (from CUDA event profilers).

        Parameters
        ----------
        timings : dict
            Mapping of section names to timing values (seconds).
            Can also be nested dict for multi-GPU scenarios.

        Notes
        -----
        These timings are merged into the final run report JSON.
        """
        self._gpu_timings.update(timings)

    # --- Dask worker snapshots ------------------------------------------------
    def snapshot_workers(self, label: str = "snapshot") -> None:
        """
        Capture current Dask worker metrics and store with given label.

        Parameters
        ----------
        label : str, default "snapshot"
            Label for this snapshot (e.g., "before", "after").

        Notes
        -----
        - Only works when a Dask client is provided.
        - Snapshots are written to CSV files in __exit__.
        """
        if not self.client or not self.enable_worker_snapshots:
            return

        metrics = _extract_worker_metrics(self.client)
        self._worker_snapshots[label] = metrics

    # --- main context ---------------------------------------------------------
    def __enter__(self) -> "Profiler":
        """
        Start configured samplers.

        Returns
        -------
        Profiler
            The profiler instance, for use as a context manager.
        """
        if self.sys_sampler:
            self.sys_sampler.start()
        if self.gpu_sampler:
            self.gpu_sampler.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """
        Stop samplers and write environment and run reports.

        Always writes:
          - env_<stamp>.json  (environment snapshot)
          - run_report_<stamp>.json  (marks, sections, paths, versions)
          - worker_<label>_<stamp>.csv  (for each worker snapshot)
        """
        # stop samplers
        if self.sys_sampler:
            self.sys_sampler.stop()
            self.sys_sampler.write_csv(self.sys_ts_csv)
        if self.gpu_sampler:
            self.gpu_sampler.stop()
            self.gpu_sampler.write_csv(
                self.benchdir / f"gpu_timeseries_{self.stamp}.csv"
            )

        # Write worker snapshots
        for label, metrics in self._worker_snapshots.items():
            csv_path = self.benchdir / f"workers_{label}_{self.stamp}.csv"
            _write_worker_metrics_csv(metrics, csv_path)

        # environment snapshot
        env = _driver_env_snapshot()
        self.env_json.write_text(json.dumps(env, indent=2), encoding="utf-8")

        # run report
        total = time.perf_counter() - self._t0
        report = {
            "stamp": self.stamp,
            "total_seconds": total,
            "marks": self._marks,
            "sections": {k: v[1] for k, v in self._sections.items()},
            "gpu_timings": self._gpu_timings,  # Include external GPU timings
            "paths": {
                "perf_html": str(self.perf_html) if self.enable_perf_html else None,
                "tasks_csv": str(self.tasks_csv) if self.enable_task_stream else None,
                "tasks_agg_csv": str(self.tasks_agg_csv) if self.enable_task_stream else None,
                "system_timeseries_csv": str(self.sys_ts_csv),
                "env_json": str(self.env_json),
            },
            "env_versions": _versions_snapshot(),
            "worker_snapshots": list(self._worker_snapshots.keys()),
        }
        self.run_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # --- unified compute context (NEW - FIXED APPROACH) -----------------------
    @contextmanager
    def compute_context(self):
        """
        Unified context manager for Dask performance report + task stream capture.

        This is the FIXED approach: work happens INSIDE this context, so both
        the performance report and task stream capture actual computation.

        Yields
        ------
        None

        Usage
        -----
        >>> with prof.compute_context():
        ...     result = client.compute(lazy_array)
        ...     result = result.result()  # Wait for completion

        Notes
        -----
        - Automatically handles both perf report and task stream if Dask client exists.
        - For non-Dask runs, this is a no-op context.
        - Task stream events are written automatically in __exit__.
        """
        if not DASK_AVAILABLE or not self.client:
            # Non-Dask: no-op context
            yield
            return

        # Stack both Dask contexts together
        with ExitStack() as stack:
            # 1. Performance report (HTML)
            if self.enable_perf_html:
                stack.enter_context(
                    performance_report(filename=str(self.perf_html))
                )

            # 2. Task stream capture
            ts = None
            if self.enable_task_stream:
                ts = stack.enter_context(
                    get_task_stream(client=self.client, plot=False)
                )

            # Yield control - work happens here!
            yield

            # After work completes, write task stream if captured
            if ts and hasattr(ts, 'data'):
                events = ts.data
                write_task_csv(events, self.tasks_csv)
                agg = aggregate_task_stream(events)
                write_agg_csv(agg, self.tasks_agg_csv)

    # --- backward compatibility (DEPRECATED) ----------------------------------
    def perf_context(self):
        """
        DEPRECATED: Use compute_context() instead.

        Context manager that emits a Dask performance report if `client` exists.

        Returns
        -------
        contextlib.AbstractContextManager
            A performance report context when a client is present and HTML
            reports are enabled; otherwise a no-op context manager.
        """
        if DASK_AVAILABLE and self.client and self.enable_perf_html:
            return performance_report(filename=str(self.perf_html))
        from contextlib import nullcontext
        return nullcontext()

    def capture_task_stream(
            self, events: Optional[Iterable[Dict[str, Any]]] = None
    ) -> None:
        """
        DEPRECATED: Use compute_context() instead.

        Record Dask task stream events and write raw/aggregated CSVs.

        Parameters
        ----------
        events : Iterable[dict] | None
            If None and a client exists, the task stream will be recorded from
            the active client using `get_task_stream`. If provided, the iterable
            is written as-is.

        Notes
        -----
        - If no client is present or task stream capture fails, this is a no-op.
        - WARNING: This method does not capture events from computation because
          no work happens inside the context manager. Use compute_context() instead.
        - Output files:
            * tasks_<stamp>.csv
            * tasks_agg_<stamp>.csv
        """
        if not DASK_AVAILABLE:
            return

        if events is None and self.client and self.enable_task_stream:
            try:
                with get_task_stream(client=self.client, plot=False) as ts:
                    # WARNING: This is the old broken approach - no work happens here!
                    self.client.wait_for_workers(1)
                    pass
                events = ts.data  # type: ignore[attr-defined]
            except Exception:
                events = None

        if events:
            write_task_csv(events, self.tasks_csv)
            agg = aggregate_task_stream(events)
            write_agg_csv(agg, self.tasks_agg_csv)

    # --- manual task stream control (for advanced users) ----------------------
    @contextmanager
    def task_stream_context(self):
        """
        Manual context manager for capturing task stream during computation.

        This is for advanced users who want fine-grained control. Most users
        should use compute_context() instead.

        Yields
        ------
        task_stream | None
            The task stream context object, or None if not available.

        Usage
        -----
        >>> with prof.task_stream_context() as ts:
        ...     result = client.compute(...)  # Actual work here!
        >>> # Task stream automatically written in compute_context
        >>> # Or manually with prof.finalize_task_stream(ts)

        Notes
        -----
        - For Dask clients only; yields None for non-Dask runs.
        - Task stream events must be manually finalized with finalize_task_stream().
        """
        if not DASK_AVAILABLE or not self.client or not self.enable_task_stream:
            yield None
            return

        with get_task_stream(client=self.client, plot=False) as ts:
            yield ts
            # Auto-finalize after context exits
            if ts and hasattr(ts, 'data'):
                events = ts.data
                write_task_csv(events, self.tasks_csv)
                agg = aggregate_task_stream(events)
                write_agg_csv(agg, self.tasks_agg_csv)

    def finalize_task_stream(self, ts=None) -> None:
        """
        DEPRECATED: Task stream is now auto-finalized in task_stream_context().

        Write task stream CSVs after computation completes.

        Parameters
        ----------
        ts : task_stream | None
            Task stream object with .data attribute.

        Notes
        -----
        This is now a no-op; task streams are automatically finalized.
        Kept for backward compatibility.
        """
        # No-op: auto-finalized in context managers
        pass


# ---------------------------------------------------------------------------
# Convenience function for simple profiling
# ---------------------------------------------------------------------------

def simple_profile(
        benchdir: Path,
        stamp: str,
        *,
        client=None,
        enable_gpu: bool = False,
) -> Profiler:
    """
    Create a Profiler with sensible defaults for quick benchmarking.

    Parameters
    ----------
    benchdir : Path
        Benchmark output directory.
    stamp : str
        Unique run identifier.
    client : dask.distributed.Client | None
        Optional Dask client for distributed profiling.
    enable_gpu : bool, default False
        Enable GPU sampling via NVML.

    Returns
    -------
    Profiler
        Configured profiler instance.

    Usage
    -----
    >>> with simple_profile(Path("bench"), "run1") as prof:
    ...     prof.section("work", start=True)
    ...     # ... do work ...
    ...     prof.section("work", start=False)
    """
    return Profiler(
        client=client,
        benchdir=benchdir,
        stamp=stamp,
        enable_perf_html=True,
        enable_task_stream=True,
        enable_worker_snapshots=True,
        enable_system_sampler=True,
        enable_gpu_sampler=enable_gpu,
    )