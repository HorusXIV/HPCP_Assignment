"""MPI process and GPU mapping helpers.

Utilities for rank-to-GPU mapping, byte-precise scatter/gather of 2D arrays,
and minimal synchronization suitable for HPC batch execution.
"""

from typing import Optional
import logging
import os

try:
    from mpi4py import MPI
except Exception:
    MPI = None

try:
    import cupy as cp  # type: ignore
except Exception:
    cp = None


def _require_cupy() -> None:
    """Ensure CuPy is importable or raise an informative ImportError."""
    try:
        import cupy  # noqa: F401
    except Exception as e:
        # Keep this message short to satisfy line-length checks
        raise ImportError(
            "CuPy is required for multiGPU execution but could not be "
            f"imported. Original error: {e}"
        )


def init_mpi():
    """Initialize MPI and return a triple ``(comm, rank, size)``.

    Returns a serial-compatible stub when ``mpi4py`` is unavailable.
    """
    if MPI is None:
        return None, 0, 1

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    return comm, rank, size


def get_local_rank_info(comm):
    """Return ``(local_rank, local_size, node_name)`` for this process.

    Prefers a shared-memory split to determine per-node rank/size and falls
    back to hostname grouping if necessary. In serial mode returns
    ``(0, 1, hostname)``.
    """
    import socket

    if comm is None:
        return 0, 1, socket.gethostname()

    node = MPI.Get_processor_name()

    try:
        local_comm = comm.Split_type(MPI.COMM_TYPE_SHARED, 0)
        local_rank = local_comm.Get_rank()
        local_size = local_comm.Get_size()
        try:
            local_comm.Free()
        except Exception:
            pass
        return local_rank, local_size, node
    except Exception:
        all_nodes = comm.allgather(node)
        local_indices = [i for i, n in enumerate(all_nodes) if n == node]
        local_size = len(local_indices)
        rank = comm.Get_rank()
        local_rank = local_indices.index(rank)
        return local_rank, local_size, node


def scatterv_array(comm, array, counts, dtype=None):
    """Scatter rows of a 2D array across ranks using byte counts.

    Args:
        comm: MPI communicator.
        array: 2D array on root; ``None`` on other ranks.
        counts: List of row counts per rank; must sum to total rows.
        dtype: Data type for non-root allocation if ``array`` is ``None``.

    Returns:
        Local 2D slice on each rank.
    """
    import numpy as _np

    rank = comm.Get_rank() if comm is not None else 0
    if comm is None:
        return array

    displs_rows = [sum(counts[:i]) for i in range(len(counts))]

    if rank == 0:
        rows = int(array.shape[0])
        cols = int(array.shape[1])
        flat = array.ravel()
    else:
        rows = None
        cols = None
        flat = None

    rows = comm.bcast(rows, root=0)
    cols = comm.bcast(cols, root=0)

    if sum(counts) != rows:
        raise ValueError(f"counts must sum to rows ({sum(counts)} != {rows})")

    local_rows = int(counts[rank])

    if array is not None:
        recv_dtype = array.dtype
    else:
        if dtype is None:
            raise ValueError("dtype must be provided on non-root ranks")
        recv_dtype = _np.dtype(dtype)

    itemsize = int(_np.dtype(recv_dtype).itemsize)
    recvbuf = _np.empty(local_rows * cols, dtype=recv_dtype)

    sendcounts_bytes = [int(c * cols * itemsize) for c in counts]
    displs_bytes = [int(d * cols * itemsize) for d in displs_rows]

    sendbuf_bytes = flat.view(_np.uint8) if flat is not None else None
    recvbuf_bytes = recvbuf.view(_np.uint8)

    try:
        from src.common.nvtx import nvtx_range  # lazy import
    except Exception:
        from contextlib import contextmanager as _cm

        @_cm
        def nvtx_range(_m):  # type: ignore
            yield
    with nvtx_range("MPI.Scatterv"):
        comm.Scatterv(
            [sendbuf_bytes, sendcounts_bytes, displs_bytes, MPI.BYTE],
            recvbuf_bytes,
            root=0,
        )

    return recvbuf.reshape(local_rows, cols)


def gatherv_array(comm, local_array, counts, root=0):
    """Gather rows from all ranks into a single 2D array on ``root``.

    Args:
        comm: MPI communicator.
        local_array: 2D local block on the current rank.
        counts: Row counts per rank (same list used in scatterv).
        root: Root rank that receives the full array.

    Returns:
        Full array on ``root``; ``None`` on other ranks.
    """
    import numpy as _np

    rank = comm.Get_rank() if comm is not None else 0
    cols = local_array.shape[1]

    sendbuf = local_array.ravel()
    itemsize = int(_np.dtype(local_array.dtype).itemsize)

    sendcounts_bytes = [int(c * cols * itemsize) for c in counts]
    displs_bytes = []
    for i in range(len(sendcounts_bytes)):
        displs_bytes.append(int(sum(sendcounts_bytes[:i])))

    if rank == root:
        total_rows = int(sum(counts))
        recvbuf = _np.empty(total_rows * cols, dtype=local_array.dtype)
    else:
        recvbuf = None

    sendbuf_bytes = sendbuf.view(_np.uint8)
    recvbuf_bytes = recvbuf.view(_np.uint8) if recvbuf is not None else None

    try:
        from src.common.nvtx import nvtx_range  # lazy import
    except Exception:
        from contextlib import contextmanager as _cm

        @_cm
        def nvtx_range(_m):  # type: ignore
            yield
    with nvtx_range("MPI.Gatherv"):
        comm.Gatherv(
            sendbuf_bytes,
            [recvbuf_bytes, sendcounts_bytes, displs_bytes, MPI.BYTE],
            root=root,
        )

    if rank == root:
        return recvbuf.reshape(sum(counts), cols)
    return None


def set_device_for_local_rank(comm, prefer_visible=True):
    """Bind the process to a GPU index based on node-local rank.

    Strategy:
    1) Respect ``CUDA_VISIBLE_DEVICES`` if present.
    2) Otherwise query CuPy for device count; fall back to ``N_GPUS``.
    3) Set ``CUDA_VISIBLE_DEVICES`` and perform a tiny runtime allocation
       to validate the device; return ``-1`` on failure.

    Returns:
        The chosen local device index or ``-1`` when no device is usable.
    """
    if comm is None:
        local_rank = 0
        local_size = 1
    else:
        local_rank, local_size, _ = get_local_rank_info(comm)

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    dev_choice = None
    if cvd:
        devs = [d for d in cvd.split(",") if d != ""]
        if len(devs) > 0:
            dev_choice = devs[local_rank % len(devs)]

    if dev_choice is None:
        try:
            if cp is not None:
                n_gpus = cp.cuda.runtime.getDeviceCount()
                if n_gpus > 0:
                    dev_choice = str(local_rank % n_gpus)
        except Exception:
            dev_choice = None

    if dev_choice is None:
        try:
            n_gpus = int(os.environ.get("N_GPUS", "0"))
            if n_gpus > 0:
                dev_choice = str(local_rank % n_gpus)
        except Exception:
            dev_choice = None

    if dev_choice is None:
        return -1

    os.environ["CUDA_VISIBLE_DEVICES"] = dev_choice

    # perform a tiny runtime test to ensure the device is usable
    try:
        if cp is not None:
            try:
                cp.cuda.Device(0).use()
                _ = cp.zeros((1,), dtype=cp.float32)
            except Exception as e:
                _log = logging.getLogger(__name__)
                _log.warning("GPU bind/runtime test failed: %s", e)
                return -1
    except Exception:
        return -1

    return int(dev_choice)


def map_rank_to_gpu(rank: int, gpus_per_node: Optional[int] = None) -> int:
    """Map a global rank to a local GPU index in a cyclic fashion.

    Args:
        rank: Global MPI rank (0-based).
        gpus_per_node: Optional number of GPUs per node. Defaults to the
            detected device count when available.

    Returns:
        Local GPU index in ``[0, gpus_per_node)`` or ``-1`` when no GPUs are
        available.
    """
    try:
        if cp is not None:
            n_gpus = cp.cuda.runtime.getDeviceCount()
        else:
            n_gpus = int(os.environ.get("N_GPUS", "0"))
    except Exception:
        n_gpus = int(os.environ.get("N_GPUS", "0"))

    if n_gpus <= 0:
        return -1

    if gpus_per_node is None:
        gpus_per_node = n_gpus

    return rank % gpus_per_node


def bind_gpu(gpu_index: int):
    """Bind the current process to a specific GPU via CUDA_VISIBLE_DEVICES.

    Args:
        gpu_index: Non-negative device index to expose to the process. If
            negative, the call is a no-op.

    Notes:
        The caller is responsible for activating the device in CuPy/Numba
        if necessary.
    """
    if gpu_index < 0:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)


def barrier(comm):
    """Synchronize all ranks if MPI is available.

    Args:
        comm: MPI communicator or ``None`` for serial mode.
    """
    if comm is None:
        return
    try:
        from src.common.nvtx import nvtx_range  # lazy import
    except Exception:
        from contextlib import contextmanager as _cm

        @_cm
        def nvtx_range(_m):  # type: ignore
            yield
    with nvtx_range("MPI.Barrier"):
        comm.Barrier()
