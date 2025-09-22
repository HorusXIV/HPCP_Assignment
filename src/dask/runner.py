# src/dask/runner.py
from __future__ import annotations
"""
Dask cluster runner.

This module constructs a local or SLURM-backed Dask cluster, optionally
scales it, and dispatches a user-specified task in the form
`module.submodule:function`.

Highlights
----------
- Caps BLAS/OpenMP threads early (via `threadpoolctl` if available) to avoid
  oversubscription in worker processes.
- Picks sensible defaults for LocalCluster ports/dirs and respects SLURM
  scratch/tmp directories when present.
- Supports adaptive scaling, explicit scaling, or fixed `n_workers`.
- Provides a backward-compat `run_dask_suite` shim to call in-repo suites.
"""

import contextlib
import importlib
import logging
import os
from typing import Any, Callable, Optional, Tuple

from dask.distributed import Client, LocalCluster
from src.common.paths import slurm_context
from src.common.threads import early_env_caps  # ensure BLAS/Omp threads are capped

log = logging.getLogger(__name__)


def _load_callable(spec: str) -> Callable:
    """
    Import and return a callable from a 'module:function' spec.

    Parameters
    ----------
    spec : str
        Import target, e.g. "src.dask.suite:run".

    Returns
    -------
    Callable
        The resolved function object.

    Raises
    ------
    ValueError
        If the resolved attribute is not callable.
    ImportError
        If the module cannot be imported.
    AttributeError
        If the attribute is missing on the imported module.
    """
    mod, fn = spec.split(":", 1)
    m = importlib.import_module(mod)
    f = getattr(m, fn, None)
    if not callable(f):
        raise ValueError(f"Not callable: {spec}")
    return f


def _build_local_cluster(args):
    """
    Create a LocalCluster with safe defaults.

    Notes
    -----
    - Caps BLAS/OpenMP threads in worker processes to 1 to avoid oversubscription.
    - Uses SLURM scratch/tmp as `local_directory` when available.
    - Chooses random free ports by default (Windows- & multi-run friendly).
    """
    # Cap BLAS/OpenMP threads inside worker *processes* to avoid oversubscription.
    # Use 1 as a safe default; inherits to child processes.
    early_env_caps(1)

    n_workers = getattr(args, "n_workers", None)
    threads_per_worker = getattr(args, "threads_per_worker", 1)
    memory_limit = getattr(args, "memory_limit", None)
    processes = bool(getattr(args, "processes", True))

    sctx = slurm_context()
    local_directory = (
        getattr(args, "local_directory", None)
        or sctx.get("tmpdir")
        or sctx.get("scratch")
        or None
    )

    scheduler_port = getattr(args, "scheduler_port", None)
    if scheduler_port in (None, "None", ""):
        scheduler_port = 0  # random free port
    dashboard_address = getattr(args, "dashboard_address", None) or ":0"

    log.info(
        "Starting LocalCluster (n_workers=%s, threads_per_worker=%s, memory_limit=%s, processes=%s) "
        "[slurm job=%s task=%s tmp=%s]",
        n_workers,
        threads_per_worker,
        memory_limit,
        processes,
        sctx.get("job_id"),
        sctx.get("array_task_id"),
        local_directory,
    )

    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit,
        processes=processes,
        scheduler_port=scheduler_port,
        dashboard_address=dashboard_address,
        local_directory=local_directory,
    )
    return cluster


def _build_slurm_cluster(args):
    """
    Create a SLURM-backed Dask cluster (dask-jobqueue).

    Environment fallbacks
    ---------------------
    Reads common environment variables (e.g., DASK_SLURM_QUEUE, SLURM_ACCOUNT)
    when explicit CLI args are absent.

    Logging
    -------
    Logs are placed under SLURM scratch if available, else the submit dir,
    else a local ./dask-logs directory.

    Raises
    ------
    RuntimeError
        If `dask_jobqueue` is not installed.
    """
    # Same early cap so SLURM workers inherit sane BLAS/OpenMP limits.
    early_env_caps(1)

    try:
        from dask_jobqueue import SLURMCluster
    except Exception as e:
        raise RuntimeError(
            "SLURM mode requested but dask_jobqueue is not available. "
            "Install with `pip install dask-jobqueue`."
        ) from e

    sctx = slurm_context()

    def _env(*keys: str, default: Optional[str] = None) -> Optional[str]:
        for k in keys:
            if k in os.environ and os.environ[k]:
                return os.environ[k]
        return default

    queue = getattr(args, "queue", None) or _env(
        "DASK_SLURM_QUEUE", "SLURM_QUEUE", "PARTITION"
    )
    account = (
        getattr(args, "account", None)
        or getattr(args, "project", None)
        or _env("DASK_SLURM_ACCOUNT", "SLURM_ACCOUNT")
    )
    cores = getattr(args, "cores", None) or _env("DASK_SLURM_CORES")
    memory = getattr(args, "memory", None) or _env("DASK_SLURM_MEMORY")
    processes = getattr(args, "processes", None) or _env("DASK_SLURM_PROCESSES")
    walltime = getattr(args, "walltime", None) or _env("DASK_SLURM_WALLTIME")
    interface = getattr(args, "interface", None) or _env("DASK_DISTRIBUTED_INTERFACE")

    # Prefer scratch for logs; fall back to submit dir; last resort: current dir
    log_directory = getattr(args, "log_directory", None)
    if not log_directory:
        if sctx.get("scratch"):
            log_directory = os.path.join(
                sctx["scratch"], f"dask-logs/{sctx.get('job_id') or 'nojid'}"
            )
        elif sctx.get("submit_dir"):
            log_directory = os.path.join(sctx["submit_dir"], "dask-logs")
        else:
            log_directory = "./dask-logs"

    job_extra_raw = getattr(args, "job_extra", None) or _env("DASK_SLURM_JOB_EXTRA")
    job_extra = (
        [s.strip() for s in job_extra_raw.split(",")]
        if isinstance(job_extra_raw, str)
        else job_extra_raw
    )
    env_extra_raw = getattr(args, "env_extra", None)
    env_extra = (
        [s.strip() for s in env_extra_raw]
        if isinstance(env_extra_raw, list)
        else env_extra_raw
    )
    worker_extra_args_raw = getattr(args, "worker_extra_args", None)
    worker_extra_args = (
        [s.strip() for s in worker_extra_args_raw]
        if isinstance(worker_extra_args_raw, list)
        else worker_extra_args_raw
    )
    scheduler_options = getattr(args, "scheduler_options", None)

    log.info(
        "Starting SLURMCluster(queue=%s, account=%s, cores=%s, memory=%s, walltime=%s, interface=%s)",
        queue,
        account,
        cores,
        memory,
        walltime,
        interface,
    )

    cluster = SLURMCluster(
        queue=queue,
        account=account,
        cores=cores,
        memory=memory,
        processes=processes,
        walltime=walltime,
        interface=interface,
        local_directory=sctx.get("scratch") or sctx.get("tmpdir") or None,
        log_directory=log_directory,
        job_extra=job_extra,
        env_extra=env_extra,
        scheduler_options=scheduler_options,
        worker_extra_args=worker_extra_args,
    )
    return cluster


def create_cluster(args) -> Tuple[Any, Client]:
    """
    Build a cluster (local or SLURM) and return it with a connected Client.

    Parameters
    ----------
    args : argparse.Namespace-like
        Parsed CLI arguments with `cluster_mode` and optional cluster settings.

    Returns
    -------
    (cluster, client)
        The constructed Dask cluster and a connected Client.

    Raises
    ------
    ValueError
        If an unknown `--cluster-mode` is requested.
    """
    mode = (getattr(args, "cluster_mode", None) or "local").lower()
    if mode not in {"local", "slurm"}:
        raise ValueError(f"Unknown --cluster-mode '{mode}'. Use 'local' or 'slurm'.")
    cluster = (
        _build_local_cluster(args) if mode == "local" else _build_slurm_cluster(args)
    )
    client = Client(cluster)
    return cluster, client


def _maybe_scale(cluster, args) -> None:
    """
    Optionally scale or adapt the cluster based on args.

    Supported knobs (if present on `args`)
    --------------------------------------
    - adapt_min / adapt_max / adapt_target : enable cluster.adapt(min, max, target)
    - scale : call cluster.scale(scale)
    - n_workers : call cluster.scale(n_workers)
    """
    n_workers = getattr(args, "n_workers", None)
    scale = getattr(args, "scale", None)
    adapt_min = getattr(args, "adapt_min", None)
    adapt_max = getattr(args, "adapt_max", None)
    adapt_target = getattr(args, "adapt_target", None)

    with contextlib.suppress(Exception):
        if adapt_min is not None or adapt_max is not None:
            kw = {}
            if adapt_min is not None:
                kw["minimum"] = int(adapt_min)
            if adapt_max is not None:
                kw["maximum"] = int(adapt_max)
            if adapt_target is not None:
                kw["target"] = int(adapt_target)
            log.info("Enabling adaptive scaling with %s", kw)
            cluster.adapt(**kw)  # type: ignore[attr-defined]
            return

    if scale is not None:
        with contextlib.suppress(Exception):
            log.info("Scaling cluster to: %s", scale)
            cluster.scale(scale)  # type: ignore[attr-defined]
            return

    if n_workers is not None:
        with contextlib.suppress(Exception):
            log.info("Scaling cluster to n_workers=%s", n_workers)
            cluster.scale(int(n_workers))  # type: ignore[attr-defined]


def run(args) -> int:
    """
    Start a Dask cluster and dispatch the requested task.

    Behavior
    --------
    - Logs basic SLURM context if available.
    - Creates cluster + client via `create_cluster(args)`.
    - Optionally scales/adapts the cluster with `_maybe_scale`.
    - If `args.task` is provided, imports and calls it, trying call signatures:
        (client=client, args=args) → (client) → ().
      Afterwards, always closes client and cluster.
    - If no task is provided, exits cleanly after bringing a client up.

    Returns
    -------
    int
        0 on success.
    """
    sctx = slurm_context()
    if any(sctx.values()):
        log.info(
            "SLURM context: job=%s name=%s array_job=%s task=%s submit_dir=%s tmp=%s scratch=%s",
            sctx.get("job_id"),
            sctx.get("job_name"),
            sctx.get("array_job_id"),
            sctx.get("array_task_id"),
            sctx.get("submit_dir"),
            sctx.get("tmpdir"),
            sctx.get("scratch"),
        )

    cluster, client = create_cluster(args)
    log.info("Dashboard: %s", getattr(client, "dashboard_link", None))
    _maybe_scale(cluster, args)

    task = getattr(args, "task", None)
    if task:
        fn = _load_callable(task)
        log.info("Running task: %s", task)
        try:
            try:
                fn(client=client, args=args)
            except TypeError:
                try:
                    fn(client)
                except TypeError:
                    fn()
        finally:
            client.close()
            with contextlib.suppress(Exception):
                cluster.close()
        return 0

    log.info("No task specified; cluster is up. Exiting after establishing client.")
    client.close()
    with contextlib.suppress(Exception):
        cluster.close()
    return 0


def run_dask_suite(*, client=None, args=None, **_ignored) -> int:
    """
    Backward-compatibility shim expected by `src.dask.__init__`.

    Tries to dispatch to:
      - `src.dask.suite:run`
      - `src.suite:run`

    Parameters
    ----------
    client : dask.distributed.Client | None
        An existing Dask client. If None, the shim does nothing.
    args : Any
        Optional argument namespace to pass to the workload.

    Returns
    -------
    int
        0 on success; 0 with a warning if no workload is found.
    """
    if client is None:
        log.warning(
            "run_dask_suite shim called without a Dask client. "
            "This shim is task-level and does not start clusters."
        )
        return 0

    for mod, func in [("src.dask.suite", "run"), ("src.suite", "run")]:
        try:
            m = importlib.import_module(mod)
            fn = getattr(m, func, None)
            if callable(fn):
                log.info("run_dask_suite: dispatching to %s:%s", mod, func)
                try:
                    fn(client=client, args=args)
                except TypeError:
                    try:
                        fn(client)
                    except TypeError:
                        fn()
                return 0
        except Exception:
            continue

    log.info(
        "run_dask_suite shim: no default workload found. "
        "Pass --task your.module:function."
    )
    return 0


__all__ = ["create_cluster", "run", "run_dask_suite"]
