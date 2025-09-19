"""Simple MPI smoke test for verifying srun/mpi4py launches.

Usage:
  srun --mpi=pmix -n <N> python tests/mpi_test.py

The script prints one line per rank with rank/size/hostname/pid so you can
confirm how many ranks were started and on which nodes.
"""
import os
import socket
import sys

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    # Print short per-rank line and flush immediately
    print(
        f"r={rank} s={size} h={socket.gethostname()} p={os.getpid()}",
        flush=True,
    )
except Exception as e:
    print(e)
    # mpi4py not present or import failed; report serial fallback
    print("mpi4py not available; running serial fallback")
    print(f"pid={os.getpid()} host={socket.gethostname()}")
    sys.exit(0)