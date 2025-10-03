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


def test_demmap_pos_errors_without_gpu(monkeypatch):
    # CPU-only shim: demmap_pos must raise a clear RuntimeError
    os.environ["MULTIGPU_STREAMS"] = "0"
    os.environ.pop("MULTIGPU_BATCH_SIZE", None)

    mod = _load_gpu_kernels_with_shim(monkeypatch)

    na, nf, nt = 4, 3, 3
    dd = np.ones((na, nf))
    ed = np.ones((na, nf))
    rmatrix = np.eye(nt, nf)
    logt = np.linspace(0, 1, nt)
    dlogt = np.ones(nt)

    with pytest.raises(RuntimeError, match="No CUDA device available"):
        mod.demmap_pos(dd, ed, rmatrix, logt, dlogt, np.ones(nf), nmu=8)


def test_demmap_pos_errors_without_gpu_batch_override(monkeypatch):
    # Even with batch overrides, CPU-only environment must error out
    os.environ["MULTIGPU_STREAMS"] = "0"
    os.environ["MULTIGPU_BATCH_SIZE"] = "3"

    mod = _load_gpu_kernels_with_shim(monkeypatch)

    na, nf, nt = 6, 4, 3
    dd = np.ones((na, nf))
    ed = np.ones((na, nf))
    rmatrix = np.eye(nt, nf)
    logt = np.linspace(0, 1, nt)
    dlogt = np.ones(nt)

    with pytest.raises(RuntimeError, match="No CUDA device available"):
        mod.demmap_pos(dd, ed, rmatrix, logt, dlogt, np.ones(nf), nmu=8)


def test_demmap_pos_errors_without_gpu_shapes(monkeypatch):
    # Previously shape tests; now verify the error is raised without GPUs
    os.environ["MULTIGPU_STREAMS"] = "0"
    mod = _load_gpu_kernels_with_shim(monkeypatch)

    na, nf, nt = 5, 4, 3
    dd = np.ones((na, nf))
    ed = np.ones((na, nf))
    rmatrix = np.eye(nt, nf)
    logt = np.linspace(0, 1, nt)
    dlogt = np.ones(nt)

    with pytest.raises(RuntimeError, match="No CUDA device available"):
        mod.demmap_pos(dd, ed, rmatrix, logt, dlogt, np.ones(nf), nmu=8)


def test_demmap_pos_errors_without_gpu_partition(monkeypatch):
    # Previously partition invariance; now verify error without GPUs
    os.environ["MULTIGPU_STREAMS"] = "0"
    mod = _load_gpu_kernels_with_shim(monkeypatch)

    na, nf, nt = 10, 4, 4
    dd = np.ones((na, nf))
    ed = np.ones((na, nf))
    rmatrix = np.eye(nt, nf)
    logt = np.linspace(0, 1, nt)
    dlogt = np.ones(nt)

    with pytest.raises(RuntimeError, match="No CUDA device available"):
        mod.demmap_pos(dd, ed, rmatrix, logt, dlogt, np.ones(nf), nmu=8)
