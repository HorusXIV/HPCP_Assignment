import numba.cuda as cuda
import numpy as np


def test():
    @cuda.jit
    def kernel(a):
        idx = cuda.grid(1)
        if idx < a.size:
            a[idx] += 1

    n = 100000
    a = np.zeros(n, dtype=np.float32)

    d_a = cuda.to_device(a)

    threads_per_block = 256
    blocks_per_grid = (n + (threads_per_block - 1)) // threads_per_block

    kernel[blocks_per_grid, threads_per_block](d_a)

    d_a.copy_to_host(a)

    assert np.all(a == 1)
