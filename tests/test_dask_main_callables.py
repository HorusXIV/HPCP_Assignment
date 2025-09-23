"""Tests for _first_importable_callable resolution order."""

from __future__ import annotations

import sys
import types

from src.dask.main import _first_importable_callable


def test_first_importable_callable_prefers_first_present() -> None:
    """Pick the first spec whose module imports and target is callable."""
    # Prepare fake modules; only mod_b exposes a callable `run`
    mod_a = types.ModuleType("fake.mod_a")
    mod_b = types.ModuleType("fake.mod_b")

    def run_b() -> int:  # simple stub
        return 0

    mod_b.run = run_b  # type: ignore[attr-defined]

    # Inject into sys.modules and restore afterward to avoid cross-test bleed
    prev_a = sys.modules.get("fake.mod_a")
    prev_b = sys.modules.get("fake.mod_b")
    sys.modules["fake.mod_a"] = mod_a
    sys.modules["fake.mod_b"] = mod_b
    try:
        specs = ["fake.mod_a:run", "fake.mod_b:run", "fake.mod_c:run"]
        picked = _first_importable_callable(specs)
        assert picked == "fake.mod_b:run"
    finally:
        if prev_a is not None:
            sys.modules["fake.mod_a"] = prev_a
        else:
            sys.modules.pop("fake.mod_a", None)
        if prev_b is not None:
            sys.modules["fake.mod_b"] = prev_b
        else:
            sys.modules.pop("fake.mod_b", None)
