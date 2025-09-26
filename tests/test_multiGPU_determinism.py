import importlib
import os
import types
import numpy as np
import pytest


def make_cupy_shim_with_streams_off():
    """Cupy shim backed by NumPy, streams/memcpy not available.

    Simulates a CPU-only environment to exercise code paths while ensuring
    determinism. No async streams so ordering issues are avoided.
    """
    cp = types.SimpleNamespace()

    # Core ndarray helpers
    cp.asarray = np.asarray
    cp.asnumpy = lambda x: np.asarray(x)
    cp.zeros = np.zeros
    cp.matmul = np.matmul
    cp.transpose = np.transpose
    cp.diag = np.diag
    cp.sqrt = np.sqrt
    cp.exp = np.exp
    cp.log = np.log
    cp.where = np.where
    cp.array = np.array
    cp.arange = np.arange
    cp.linspace = np.linspace
    cp.maximum = np.maximum
    cp.abs = np.abs
    cp.argmin = np.argmin
    cp.isfinite = np.isfinite
    cp.sum = np.sum
    cp.float64 = np.float64

    # linalg
    linalg = types.SimpleNamespace()
    linalg.svd = lambda A, full_matrices=True: np.linalg.svd(
        np.asarray(A), full_matrices=full_matrices
    )
    linalg.pinv = lambda A: np.linalg.pinv(np.asarray(A))
    cp.linalg = linalg

    cp.finfo = np.finfo
    cp.max = np.max
    cp.min = np.min

    # CUDA stubs: no streams, no runtime features
    class _CUDARuntime:
        @staticmethod
        def getDeviceCount():
            return 0

    class _Cuda:
        runtime = _CUDARuntime()

    cp.cuda = _Cuda()
    return cp


@pytest.fixture(autouse=True)
def _isolate_modules(monkeypatch):
    # ensure we don't leak a real cupy between tests
    for mod in list(importlib.sys.modules.keys()):
        if mod.startswith("src.multiGPU"):
            importlib.sys.modules.pop(mod)
    yield
    for mod in list(importlib.sys.modules.keys()):
        if mod.startswith("src.multiGPU"):
            importlib.sys.modules.pop(mod)


def _load_gpu_kernels_with_shim(monkeypatch):
    cp = make_cupy_shim_with_streams_off()
    monkeypatch.setitem(importlib.sys.modules, "cupy", cp)
    mod = importlib.import_module("src.multiGPU.gpu_kernels")
    importlib.reload(mod)
    return mod


def test_demmap_pos_deterministic_small(monkeypatch):
    # Force deterministic settings
    os.environ["MULTIGPU_NO_FUSE"] = "1"
    os.environ["MULTIGPU_STREAMS"] = "0"  # ensure sync path
    os.environ.pop("MULTIGPU_BATCH_SIZE", None)

    mod = _load_gpu_kernels_with_shim(monkeypatch)

    # small synthetic inputs
    na, nf, nt = 8, 5, 6
    rng = np.random.default_rng(123)
    dd = rng.normal(size=(na, nf))
    ed = np.abs(rng.normal(size=(na, nf))) + 0.1
    rmatrix = np.abs(rng.normal(size=(nt, nf)))
    logt = np.linspace(5.0, 7.0, nt)
    dlogt = np.full(nt, logt[1] - logt[0])

    out1 = mod.demmap_pos(dd, ed, rmatrix, logt, dlogt, np.ones(nf), nmu=32)
    out2 = mod.demmap_pos(dd, ed, rmatrix, logt, dlogt, np.ones(nf), nmu=32)

    for a, b in zip(out1, out2):
        assert np.allclose(a, b, rtol=0, atol=0), "Outputs differ across runs"


def test_demmap_pos_batching_invariance(monkeypatch):
    # Disable fuse and async; then compare full vs tiny batches
    os.environ["MULTIGPU_NO_FUSE"] = "1"
    os.environ["MULTIGPU_STREAMS"] = "0"

    mod = _load_gpu_kernels_with_shim(monkeypatch)

    na, nf, nt = 32, 7, 9
    rng = np.random.default_rng(42)
    dd = rng.normal(size=(na, nf))
    ed = np.abs(rng.normal(size=(na, nf))) + 0.05
    rmatrix = np.abs(rng.normal(size=(nt, nf)))
    logt = np.linspace(5.0, 7.0, nt)
    dlogt = np.full(nt, logt[1] - logt[0])

    # Run with large batch (no override, shim defaults to small but >= na)
    out_large = mod.demmap_pos(dd, ed, rmatrix, logt, dlogt, np.ones(nf), nmu=16)

    # Re-import with enforced tiny batch size via env override
    importlib.reload(mod)
    os.environ["MULTIGPU_BATCH_SIZE"] = "3"
    out_small = mod.demmap_pos(dd, ed, rmatrix, logt, dlogt, np.ones(nf), nmu=16)

    for a, b in zip(out_large, out_small):
        assert np.allclose(a, b, rtol=0, atol=0), "Batching changed results"


def test_shapes_and_basic_properties(monkeypatch):
    os.environ["MULTIGPU_NO_FUSE"] = "1"
    os.environ["MULTIGPU_STREAMS"] = "0"

    mod = _load_gpu_kernels_with_shim(monkeypatch)

    na, nf, nt = 5, 4, 3
    dd = np.ones((na, nf))
    ed = np.ones((na, nf))
    rmatrix = np.eye(nt, nf)
    logt = np.linspace(0, 1, nt)
    dlogt = np.ones(nt)

    dem, edem, elogt, chisq, dn_reg = mod.demmap_pos(
        dd, ed, rmatrix, logt, dlogt, np.ones(nf), reg_tweak=1e-12, nmu=32
    )

    assert dem.shape == (na, nt)
    assert edem.shape == (na, nt)
    assert elogt.shape == (na, nt)
    assert chisq.shape == (na,)
    assert dn_reg.shape == (na, nf)

    # For near-identity response and tiny regularization, predicted data
    # should closely match inputs on the first nt channels.
    assert np.allclose(dd[:, :nt], dn_reg[:, :nt], atol=2e-4)


def test_partition_invariance(monkeypatch):
    # Results should be identical whether processed all-at-once or in chunks
    os.environ["MULTIGPU_NO_FUSE"] = "1"
    os.environ["MULTIGPU_STREAMS"] = "0"

    mod = _load_gpu_kernels_with_shim(monkeypatch)

    na, nf, nt = 27, 6, 8
    rng = np.random.default_rng(7)
    dd = rng.normal(size=(na, nf))
    ed = np.abs(rng.normal(size=(na, nf))) + 0.02
    rmatrix = np.abs(rng.normal(size=(nt, nf)))
    logt = np.linspace(5.0, 7.0, nt)
    dlogt = np.full(nt, logt[1] - logt[0])

    whole = mod.demmap_pos(dd, ed, rmatrix, logt, dlogt, np.ones(nf), nmu=20)

    # split into uneven chunks to stress boundary conditions
    splits = [0, 5, 13, na]
    parts = []
    for s0, s1 in zip(splits[:-1], splits[1:]):
        parts.append(
            mod.demmap_pos(
                dd[s0:s1], ed[s0:s1], rmatrix, logt, dlogt, np.ones(nf), nmu=20
            )
        )

    # stitch
    dem_cat = np.vstack([p[0] for p in parts])
    edem_cat = np.vstack([p[1] for p in parts])
    elogt_cat = np.vstack([p[2] for p in parts])
    chisq_cat = np.concatenate([p[3] for p in parts])
    dnreg_cat = np.vstack([p[4] for p in parts])

    for a, b in zip(whole, (dem_cat, edem_cat, elogt_cat, chisq_cat, dnreg_cat)):
        assert np.allclose(a, b, rtol=0, atol=0), "Partitioning changed results"
