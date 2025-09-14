# src/common/nvtx.py
from contextlib import contextmanager
@contextmanager
def nvtx_range(msg: str):
    try:
        import nvtx
        with nvtx.annotate(msg):
            yield
    except Exception:
        # no-op if nvtx not present
        yield
