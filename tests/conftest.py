import io
import os
from pathlib import Path
from typing import Dict, Any

import pytest


# Registry of mocked GET responses: url -> {content, headers, status}
_REGISTRY: Dict[str, Dict[str, Any]] = {}


def _make_bytes(content):
    if isinstance(content, (bytes, bytearray)):
        return bytes(content)
    return str(content).encode("utf-8")


def _fake_urlopen(req, *args, **kwargs):
    # req may be a urllib.request.Request or a URL string
    try:
        from urllib.request import Request
    except Exception:
        Request = None

    if Request and isinstance(req, Request):
        url = req.full_url
    else:
        url = req

    if url not in _REGISTRY:
        raise AssertionError(
            "Test attempted real network call to %s (no mock registered)" % url
        )

    entry = _REGISTRY[url]
    content = _make_bytes(entry.get("content", b""))

    class _Resp:
        def __init__(self, data: bytes):
            self._buf = io.BytesIO(data)
            # some callers inspect a 'length' attribute
            self.length = len(data)

        def read(self, n: int | None = None):
            return self._buf.read() if n is None else self._buf.read(n)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    return _Resp(content)


def _fake_requests_request(self, method, url, *args, **kwargs):
    import requests

    if url not in _REGISTRY:
        raise AssertionError(
            "Test attempted real network call to %s (no mock registered)" % url
        )

    entry = _REGISTRY[url]
    content = _make_bytes(entry.get("content", b""))

    resp = requests.Response()
    resp.status_code = int(entry.get("status", 200))
    # requests expects _content for bytes payload
    resp._content = content
    resp.headers = entry.get("headers") or {}
    resp.url = url
    return resp


def _fake_httpx_request(self, method, url, *args, **kwargs):
    # httpx.Response API differs; create a minimal shim if httpx is installed
    try:
        import httpx
    except Exception:
        raise AssertionError(
            "httpx used but not available in test environment"
        )

    if url not in _REGISTRY:
        raise AssertionError(
            "Test attempted real network call to %s (no mock registered)" % url
        )

    entry = _REGISTRY[url]
    content = _make_bytes(entry.get("content", b""))
    status = int(entry.get("status", 200))
    headers = entry.get("headers") or {}

    return httpx.Response(status_code=status, content=content, headers=headers)


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    """Autouse fixture that prevents real network GETs during tests.

    Tests must explicitly register expected URLs using the `register_get`
    fixture (preferred) or the `pytest.register_get` helper which is also
    provided for backwards compatibility.
    """
    # Patch urllib.request.urlopen
    try:
        import urllib.request as _urllib_request
        monkeypatch.setattr(_urllib_request, "urlopen", _fake_urlopen)
    except Exception:
        # If urllib isn't present for some reason, ignore
        pass

    # Patch requests.Session.request (covers requests.get/post/...)
    try:
        import requests.sessions as _rs
        monkeypatch.setattr(
            _rs.Session, "request", _fake_requests_request, raising=True
        )
    except Exception:
        pass

    # Patch httpx.Client.request if httpx is installed
    try:
        import httpx
        monkeypatch.setattr(
            httpx.Client, "request", _fake_httpx_request, raising=True
        )
    except Exception:
        # httpx not installed — that's fine
        pass

    # Expose a convenience helper on pytest for quick one-off registrations
    def _reg(url, content=b"", headers=None, status=200):
        _REGISTRY.__setitem__(
            url,
            {
                "content": content,
                "headers": headers or {},
                "status": status,
            },
        )

    pytest.register_get = _reg

    yield

    # Cleanup registry and helper
    _REGISTRY.clear()
    try:
        del pytest.register_get
    except Exception:
        pass


@pytest.fixture
def register_get():
    """Fixture to register a mocked GET response for a URL.

    Usage in tests:
        def test_something(register_get):
            register_get("https://example.com/data", b"hello")
            ... code that triggers a GET to that URL ...
    """
    def _register(
        url: str, content=b"", headers: dict | None = None, status: int = 200
    ):
        _REGISTRY[url] = {
            "content": content,
            "headers": headers or {},
            "status": status,
        }

    return _register


@pytest.fixture(scope="session", autouse=True)
def provide_synthetic_np32(tmp_path_factory):
    """Ensure a minimal reproducible np32 .npz exists for tests that expect
    a real file on disk (some tests spawn subprocesses and can't be
    monkeypatched). This fixture creates a small synthetic stack and sets
    the `HPCP_TEST_INPUT` env var so tests and subprocesses can use it.

    Behavior:
    - If `HPCP_TEST_INPUT` is already set, do nothing.
    - If the default path (data/np32/20170906_12_00_12.npz) exists, do nothing.
    - Otherwise, create a small deterministic .npz under a tmpdir and set
      `HPCP_TEST_INPUT` to point to it.
    """
    import numpy as np

    env_var = "HPCP_TEST_INPUT"
    if os.environ.get(env_var):
        # User provided a path; trust it.
        return

    default_path = (Path(__file__).resolve().parents[2] / "data" / "np32" / "20170906_12_00_12.npz")
    if default_path.exists():
        # Real data already available locally; nothing to do.
        return

    # Create deterministic synthetic data (small to keep CI fast)
    out_dir = tmp_path_factory.mktemp("np32_test")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "20170906_12_00_12.npz"

    # Reproducible RNG
    rng = np.random.default_rng(12345)
    # Create a (6, H, W) array with small H,W
    H = 32
    W = 32
    bands = rng.standard_normal((6, H, W)).astype(np.float32)
    # Save with key 'bands' to match loader expectations
    np.savez(out_file, bands=bands)

    # Export env var so subprocesses inherit it
    os.environ[env_var] = str(out_file)

    return
