"""MPI process and GPU mapping helpers.

Provides a small abstraction on top of mpi4py for rank<->GPU mapping,
collectives, and a simple error/heartbeat mechanism.
"""

from typing import Optional
import logging
import os

try:
    from mpi4py import MPI
except Exception:
    MPI = None

try:
    import cupy as cp
except Exception:
    cp = None


def init_mpi():
    """Initialize MPI and return (comm, rank, size).

    If mpi4py is not available the function returns a serial stub where
    rank=0 and size=1 so code can still run for local debugging.
    """
    if MPI is None:
        return None, 0, 1

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    return comm, rank, size


def get_local_rank_info(comm):
    """Return (local_rank, local_size, node_name) for the calling process.

    Uses `MPI.Get_processor_name()` to group ranks by node. If MPI not
    available returns (0,1,hostname).
    """
    import socket

    if comm is None:
        return 0, 1, socket.gethostname()

    node = MPI.Get_processor_name()
    all_nodes = comm.allgather(node)
    local_indices = [i for i, n in enumerate(all_nodes) if n == node]
    local_size = len(local_indices)
    rank = comm.Get_rank()
    local_rank = local_indices.index(rank)
    return local_rank, local_size, node


def scatterv_array(comm, array, counts, dtype=None):
    """Scatter a 2D array (rows) across ranks using counts list.

    - `array` is only required on root (rank 0); other ranks pass None.
    - `counts` is number of rows to send to each rank and must sum to rows.
    Returns local slice (numpy.ndarray) on each rank.
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

    # use MPI.BYTE so counts are in bytes and avoid mismatched datatypes
    comm.Scatterv(
        [sendbuf_bytes, sendcounts_bytes, displs_bytes, MPI.BYTE],
        recvbuf_bytes,
        root=0,
    )

    return recvbuf.reshape(local_rows, cols)


def gatherv_array(comm, local_array, counts, root=0):
    """Gather a 2D local_array from all ranks into a single array on root.

    Returns the full array on root, and None on other ranks.
    """
    import numpy as _np

    rank = comm.Get_rank() if comm is not None else 0
    cols = local_array.shape[1]

    sendbuf = local_array.ravel()
    itemsize = int(_np.dtype(local_array.dtype).itemsize)

    sendcounts_bytes = [int(c * cols * itemsize) for c in counts]
    displs_bytes = [int(sum(sendcounts_bytes[:i])) for i in range(len(sendcounts_bytes))]

    if rank == root:
        total_rows = int(sum(counts))
        recvbuf = _np.empty(total_rows * cols, dtype=local_array.dtype)
    else:
        recvbuf = None

    sendbuf_bytes = sendbuf.view(_np.uint8)
    recvbuf_bytes = recvbuf.view(_np.uint8) if recvbuf is not None else None

    comm.Gatherv(
        sendbuf_bytes,
        [recvbuf_bytes, sendcounts_bytes, displs_bytes, MPI.BYTE],
        root=root,
    )

    if rank == root:
        return recvbuf.reshape(sum(counts), cols)
    return None


def set_device_for_local_rank(comm, prefer_visible=True):
    """Bind the current process to a GPU according to node-local rank.

    Strategy:
    - If `CUDA_VISIBLE_DEVICES` is already set, parse it and pick the
      device string corresponding to the local rank modulo that list.
    - Otherwise, query CuPy for device count or fall back to env `N_GPUS`.
    - Set `CUDA_VISIBLE_DEVICES` to the chosen device string and perform a
      lightweight runtime health check (tiny allocation) to ensure the
      device is usable. If the test fails, return -1.
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
                logging.getLogger(__name__).warning("GPU bind/runtime test failed: %s", e)
                return -1
    except Exception:
        return -1

    # best-effort numba device select
    try:
        from numba import cuda as _ncuda

        try:
            _ncuda.select_device(0)
        except Exception:
            pass
    except Exception:
        pass

    return int(dev_choice)


def map_rank_to_gpu(rank: int, gpus_per_node: Optional[int] = None) -> int:
    """Map an MPI rank to a local GPU index.

    Strategy: use `CUDA_VISIBLE_DEVICES` if set; otherwise assume GPUs
    are numbered 0..n-1 on each node and bind ranks cyclically.
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
    """Set environment to bind current process to a GPU index (local index).

    This sets `CUDA_VISIBLE_DEVICES` for consistency with child processes.
    The caller should still select the device inside CuPy/numba if needed.
    """
    if gpu_index < 0:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)


def barrier(comm):
    if comm is None:
        return
    comm.Barrier()

