# src/dask/runner.py
# Slurm-aware Dask runner for single-node jobs
# - processes=True (one process/worker)
# - threads_per_worker configurable (default 1)
# - n_workers from Slurm allocation unless overridden

from __future__ import annotations

import os
from contextlib import contextmanager

from dask.distributed import Client, LocalCluster


def _env_default_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default


def recommended_worker_count() -> int:
    """
    Use the Slurm allocation to decide worker count.
    Falls back to os.cpu_count() if Slurm vars are missing.
    """
    n_tasks = _env_default_int("SLURM_NTASKS", 0)
    n_cpus_per_task = _env_default_int("SLURM_CPUS_PER_TASK", 0)

    if n_tasks > 0 and n_cpus_per_task > 0:
        return n_tasks * n_cpus_per_task

    if n_cpus_per_task > 0:
        return n_cpus_per_task

    return max(1, (os.cpu_count() or 1))


def set_thread_env(one: int = 1) -> None:
    """
    Prevent nested parallelism inside each worker.
    """
    os.environ.setdefault("OMP_NUM_THREADS", str(one))
    os.environ.setdefault("MKL_NUM_THREADS", str(one))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(one))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(one))
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    # Dask worker memory policy (tune as needed)
    os.environ.setdefault("DASK_DISTRIBUTED__WORKER__MEMORY__TARGET", "0.80")
    os.environ.setdefault("DASK_DISTRIBUTED__WORKER__MEMORY__SPILL", "0.85")
    os.environ.setdefault("DASK_DISTRIBUTED__WORKER__MEMORY__TERMINATE", "0.95")
    os.environ.setdefault("DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT", "60s")
    os.environ.setdefault("DASK_DISTRIBUTED__COMM__RETRY__DELAY__MIN", "50ms")


@contextmanager
def dask_client_single_node(
    n_workers: int | None = None,
    threads_per_worker: int = 1,
):
    """
    Create a single-node Dask cluster inside the Slurm allocation.
    - processes=True (one process/worker)
    - threads_per_worker configurable (default 1)
    - n_workers defaults to SLURM_CPUS_PER_TASK (or cpu_count)
    """
    set_thread_env(1)
    if n_workers is None:
        n_workers = recommended_worker_count()

    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        processes=True,
        dashboard_address=None,  # disable dashboard in batch jobs
    )
    try:
        client = Client(cluster)
        yield client
    finally:
        try:
            client.close()
        except Exception:
            pass
        try:
            cluster.close()
        except Exception:
            pass
