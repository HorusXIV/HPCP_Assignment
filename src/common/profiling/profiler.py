# src/common/profiling/profiler.py
from __future__ import annotations
"""
Cross-backend profiling utilities.

This module implements a lightweight, backend-agnostic profiler that can be used
by the baseline runner and the Dask runners alike. It collects:

- Timings via explicit `mark()` and `section()` calls
- Optional Dask performance report (HTML) when a `Client` is provided
- Dask task stream events (CSV) and simple aggregations (CSV)
- System time series sampler (CPU/RAM) and optional GPU (NVML) sampler
- Environment and versions snapshots (JSON)

Artifacts are written into the benchmark directory using a timestamp to
disambiguate runs.
"""

import os
import json
import time
import platform
from pathlib import Path
from typing import Dict, Any, Optional, Iterable, Tuple

from dask.distributed import performance_report, get_task_stream, Client

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


class Profiler:
    """
    Lightweight profiler/context manager for CPU and Dask runs.

    Responsibilities
    ----------------
    - Manage timing sections and ad-hoc marks
    - Optionally emit a Dask performance report (HTML)
    - Collect Dask task stream events and write raw/aggregated CSVs
    - Run background system/GPU samplers and persist their time series
    - Write environment/versions/run JSON reports

    Usage
    -----
    >>> with Profiler(client, benchdir, stamp) as prof:
    ...     with prof.perf_context():
    ...         # do work
    ...     prof.section("compute", start=True)
    ...     # ... work ...
    ...     prof.section("compute", start=False)
    """

    def __init__(
        self,
        client: Optional[Client],
        benchdir: Path,
        stamp: str,
        *,
        enable_perf_html: bool = True,
        enable_system_sampler: bool = True,
        enable_gpu_sampler: bool = False,
    ):
        """
        Parameters
        ----------
        client : dask.distributed.Client | None
            If provided, enables Dask-specific features (perf report, task stream).
        benchdir : Path
            Directory where artifacts will be written.
        stamp : str
            Unique identifier (e.g., timestamp) for naming artifacts.
        enable_perf_html : bool, default True
            Emit Dask performance report HTML (requires `client`).
        enable_system_sampler : bool, default True
            Record driver CPU/memory timeseries via `SystemSampler`.
        enable_gpu_sampler : bool, default False
            Record GPU timeseries via `NVMLSampler` (if NVML is available).
        """
        self.client = client
        self.benchdir = Path(benchdir)
        self.stamp = stamp
        self.enable_perf_html = bool(enable_perf_html)
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
            "paths": {
                "perf_html": str(self.perf_html),
                "tasks_csv": str(self.tasks_csv),
                "tasks_agg_csv": str(self.tasks_agg_csv),
                "system_timeseries_csv": str(self.sys_ts_csv),
                "env_json": str(self.env_json),
            },
            "env_versions": _versions_snapshot(),
        }
        self.run_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # --- utilities ------------------------------------------------------------
    def perf_context(self):
        """
        Context manager that emits a Dask performance report if `client` exists.

        Returns
        -------
        contextlib.AbstractContextManager
            A performance report context when a client is present and HTML
            reports are enabled; otherwise a no-op context manager.
        """
        if self.client and self.enable_perf_html:
            return performance_report(filename=str(self.perf_html))
        from contextlib import nullcontext

        return nullcontext()

    def capture_task_stream(
        self, events: Optional[Iterable[Dict[str, Any]]] = None
    ) -> None:
        """
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
        - Output files:
            * tasks_<stamp>.csv
            * tasks_agg_<stamp>.csv
        """
        if events is None and self.client:
            try:
                with get_task_stream(client=self.client, plot=False) as ts:
                    self.client.wait_for_workers(1)
                    pass
                events = ts.data  # type: ignore[attr-defined]
            except Exception:
                events = None

        if events:
            write_task_csv(events, self.tasks_csv)
            agg = aggregate_task_stream(events)
            write_agg_csv(agg, self.tasks_agg_csv)
