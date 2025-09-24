import numpy as np
import pytest

try:
    import cupy as cp
    HAS_CUPY = True
except Exception:
    HAS_CUPY = False

from src.multiGPU import gpu_kernels


@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed in environment")
def test_demmap_pos_batched_small():
    # small synthetic problem: nf=3, nt=5, na=8
    nf = 3
    nt = 5
    na = 8
    # create a simple rmatrix (nt x nf)
    rmatrix = np.abs(np.random.randn(nt, nf)) + 0.1
    logt = np.linspace(5.0, 7.0, nt)
    dlogt = np.full(nt, logt[1] - logt[0])

    # synthetic dd and ed
    dd = np.abs(np.random.randn(na, nf)) + 0.1
    ed = np.ones((na, nf)) * 0.05

    dem, edem, elogt, chisq, dn_reg = gpu_kernels.demmap_pos(
        dd, ed, rmatrix, logt, dlogt, glc=np.ones(nf),
        reg_tweak=1.0, max_iter=3, nmu=16
    )

    assert dem.shape == (na, nt)
    assert edem.shape == (na, nt)
    assert elogt.shape == (na, nt)
    assert chisq.shape == (na,)
    assert dn_reg.shape == (na, nf)
    # basic numeric sanity
    assert np.all(np.isfinite(dem))
    assert np.all(dem >= -1e3)
