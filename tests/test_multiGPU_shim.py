import importlib
import types
import numpy as np


def make_cupy_shim():
    """Return a simple `cupy` shim module backed by NumPy for tests.

    The shim provides the minimal API surface used by `src.multiGPU`:
    - `asarray`, `zeros`, `linalg.svd`, `linalg.pinv`, `transpose`,
      `diag`, `maximum`, `isfinite`, `sum`, `asnumpy`, `finfo` and a
      `.cuda.runtime.getDeviceCount()` implementation that returns 0.
    This keeps tests runnable on machines without GPUs.
    """

    cp = types.SimpleNamespace()

    # array-level helpers
    cp.asarray = np.asarray
    cp.asnumpy = lambda x: np.asarray(x)
    cp.zeros = np.zeros
    cp.transpose = np.transpose
    cp.diag = np.diag
    cp.maximum = np.maximum
    cp.isfinite = np.isfinite
    cp.sum = np.sum

    # linalg namespace
    linalg = types.SimpleNamespace()

    def _svd(A, full_matrices=True):
        return np.linalg.svd(np.asarray(A), full_matrices=full_matrices)

    def _pinv(A):
        return np.linalg.pinv(np.asarray(A))

    linalg.svd = _svd
    linalg.pinv = _pinv
    cp.linalg = linalg

    # dtype/info
    class _Finfo:
        @staticmethod
        def tiny():
            return np.finfo(float).tiny

    cp.finfo = np.finfo
    cp.max = np.max
    cp.min = np.min
    cp.geomspace = np.geomspace

    # minimal cuda runtime stub
    class _CUDARuntime:
        @staticmethod
        def getDeviceCount():
            return 0

    class _Cuda:
        runtime = _CUDARuntime()

        class Device:
            def __init__(self, idx):
                self.idx = idx

            def use(self):
                # no-op on CPU shim
                return

    cp.cuda = _Cuda()

    return cp


def test_import_gpu_kernels_with_shim(monkeypatch):
    # Insert shim into sys.modules so import uses it
    cp = make_cupy_shim()
    monkeypatch.setitem(importlib.sys.modules, "cupy", cp)

    # Reload module under test to ensure it picks up shim
    mod = importlib.import_module("src.multiGPU.kernels")
    importlib.reload(mod)

    # basic call to dem_reg_map with small synthetic inputs
    sigmaa = np.array([2.0, 1.0])
    sigmab = np.array([1.0, 0.5])
    # U: shape (M, nf) - provide a small orthonormal like matrix
    # U should have second dimension == nf (data length) -> shape (M, nf)
    U = np.eye(2)
    W = np.eye(2)
    data = np.array([1.0, 0.5])
    err = np.array([0.1, 0.1])

    mu = mod.dem_reg_map(sigmaa, sigmab, U, W, data, err, reg_tweak=1.0, nmu=10)
    assert isinstance(mu, float)
    assert mu > 0


def test_safe_svd_raises_on_bad_input(monkeypatch):
    # Shim cupy to a module that raises on svd to test exception path
    class BadCupy(types.SimpleNamespace):
        pass

    bad_cp = BadCupy()

    class BadLinalg:
        @staticmethod
        def svd(A, full_matrices=True):
            raise RuntimeError("simulated GPU svd failure")

    bad_cp.asarray = np.asarray
    bad_cp.linalg = BadLinalg()
    bad_cp.asnumpy = lambda x: np.asarray(x)
    monkeypatch.setitem(importlib.sys.modules, "cupy", bad_cp)

    mod = importlib.import_module("src.multiGPU.kernels")
    importlib.reload(mod)

    A = np.eye(3)
    try:
        # safe_svd should raise RuntimeError when cupy.svd fails
        mod.safe_svd(A)
        raised = False
    except RuntimeError:
        raised = True

    assert raised, "safe_svd did not raise on simulated GPU failure"


def test_mpi_manager_mapping_no_gpu(monkeypatch):
    # Ensure mpi_manager functions behave when cupy.runtime reports 0 GPUs
    cp = make_cupy_shim()
    monkeypatch.setitem(importlib.sys.modules, "cupy", cp)

    mmpi = importlib.import_module("src.multiGPU.mpi_manager")
    importlib.reload(mmpi)

    # map_rank_to_gpu should return -1 when zero GPUs
    assert mmpi.map_rank_to_gpu(0) == -1
    # bind_gpu should not error when passed -1
    mmpi.bind_gpu(-1)
