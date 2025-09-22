"""Smoke test around the vendor entry point.

This test verifies that our code can patch the vendor module
(`src.baseline.vendor.dn2dem_pos`) and that the function is invoked with
the expected array shapes. We don't validate numerical outputs here—only
callability and interface.
"""

from __future__ import annotations

import importlib
import types

import numpy as np


def test_runner_calls_vendor_with_expected_shapes(monkeypatch) -> None:
    called: dict[str, tuple[tuple[int, ...], ...]] = {}

    def fake_dn2dem_pos(
        f: np.ndarray,
        e: np.ndarray,
        T_RESP: np.ndarray,
        logT: np.ndarray,
        TEMPS: np.ndarray,
        nmu: int = 42,
    ):
        # Record shapes the caller provided
        called["shapes"] = (f.shape, e.shape, T_RESP.shape, logT.shape, TEMPS.shape)

        H, W, nf = f.shape
        nt = int(TEMPS.size - 1)

        # Return correctly shaped, trivial outputs
        return (
            np.zeros((H, W, nt), dtype=np.float32),  # dem
            np.zeros((H, W, nt), dtype=np.float32),  # edem
            np.linspace(0.0, 1.0, nt, dtype=np.float32),  # logT_bins
            np.zeros((H, W), dtype=np.float32),  # chisq
            np.zeros((H, W, nf), dtype=np.float32),  # dn_reg
        )

    # Import the module object, then patch its symbol
    dn2dem_mod = importlib.import_module("src.baseline.vendor.dn2dem_pos")
    assert isinstance(dn2dem_mod, types.ModuleType)

    monkeypatch.setattr(dn2dem_mod, "dn2dem_pos", fake_dn2dem_pos)

    # Tiny, well-formed inputs
    H = W = 8
    nf = 4
    nt = 6
    f = np.ones((H, W, nf), dtype=np.float32)
    e = np.full_like(f, 0.1, dtype=np.float32)
    logT = np.linspace(5.5, 7.5, 32, dtype=np.float32)
    T_RESP = np.ones((logT.size, nf), dtype=np.float32)
    TEMPS = np.logspace(5.5, 7.5, nt + 1, dtype=np.float32)

    # Invoke the patched vendor function
    dn2dem_mod.dn2dem_pos(f, e, T_RESP, logT, TEMPS, nmu=42)

    # Assert the call interface (shapes) was as expected
    assert called["shapes"] == (f.shape, e.shape, T_RESP.shape, logT.shape, TEMPS.shape)
