"""Test scatterv/gatherv helpers under MPI.

Run with:
  PYTHONPATH=. srun --mpi=pmix -n 4 poetry run python tests/mpi_scatterv_test.py

This will build a small random array on rank 0, scatter rows across ranks with
`mpi_manager.scatterv_array`, then gather them back and verify the reconstruction
is correct.
"""
from __future__ import annotations

import sys
import numpy as np

# allow running from repo root
from src.multiGPU import mpi_manager as mmpi


def main():
    try:
        from mpi4py import MPI
    except Exception:
        print("mpi4py not available; aborting test")
        sys.exit(1)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # build a small 2D array on root (rows x cols)
    rows = 10
    cols = 6
    if rank == 0:
        arr = np.arange(rows * cols, dtype=np.float64).reshape(rows, cols)
        print(f"root has array shape {arr.shape}")
    else:
        arr = None

    counts = [rows // size + (1 if i < (rows % size) else 0) for i in range(size)]

    # scatter
    local = mmpi.scatterv_array(comm, arr if rank == 0 else None, counts, dtype=np.float64)
    print(f"rank {rank} got local shape {local.shape}")

    # simulate local modification and gather back
    local *= 2.0
    gathered = mmpi.gatherv_array(comm, local, counts, root=0)

    if rank == 0:
        # original arr * 2 expected
        expected = (np.arange(rows * cols, dtype=np.float64).reshape(rows, cols) * 2.0)
        ok = np.allclose(gathered, expected)
        print(f"gathered shape {gathered.shape}, ok={ok}")
        if not ok:
            print("Mismatch detected")
            print("expected:\n", expected)
            print("got:\n", gathered)
            sys.exit(2)


if __name__ == '__main__':
    main()
