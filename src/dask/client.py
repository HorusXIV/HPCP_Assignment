# src/dask/client.py
from __future__ import annotations
from contextlib import contextmanager
from typing import Optional
import os

def _set_worker_caps(threads: Optional[int]):
    if threads is None:
        return
    t = str(max(1, int(threads)))
    os.environ.update({
        "OMP_NUM_THREADS": t,
        "OPENBLAS_NUM_THREADS": t,
        "MKL_NUM_THREADS": t,
        "VECLIB_MAXIMUM_THREADS": t,
        "NUMEXPR_NUM_THREADS": t,
    })

@contextmanager
def build_client(
    *,
    n_workers: int = 8,
    threads_per_worker: int = 1,
    processes: bool | None = None,
    memory_limit: str | int | None = "4GB",
    scheduler_address: str | None = None,   # connect to existing cluster if provided
    set_worker_thread_caps: Optional[int] = 1,  # per-worker BLAS/OpenMP cap
):
    """
    Context manager that yields a connected dask.distributed.Client.
    If scheduler_address is None, starts a local cluster.
    """
    from dask.distributed import Client, LocalCluster

    if scheduler_address:
        client = Client(scheduler_address)
        if set_worker_thread_caps is not None:
            client.run(_set_worker_caps, set_worker_thread_caps)
        try:
            yield client
        finally:
            client.close()
        return

    if processes is None:
        # vendor spawns processes itself; avoid daemonic workers
        import platform
        processes = False  # good default everywhere for this workload
        if platform.system() == "Linux":
            # you could set True here if you KNOW vendor won't spawn procs
            processes = False

    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        processes=processes,
        memory_limit=memory_limit,
        dashboard_address=":0",  # auto-pick a port
    )
    client = Client(cluster)
    if set_worker_thread_caps is not None:
        client.run(_set_worker_caps, set_worker_thread_caps)
    try:
        yield client
    finally:
        client.close()
        cluster.close()
