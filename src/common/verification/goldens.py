# src/common/verification/goldens.py
from __future__ import annotations
"""
Generate and persist "golden" reference outputs for the DEM pipeline.

This module provides a single public helper, `write_goldens`, which:
  • Discovers NPZ inputs containing 'bands' shaped (6, H, W)
  • Loads and optionally crops one frame
  • Runs the unified CPU solver (`solve_tile_all`)
  • Writes a compressed NPZ with standardized keys:
        demmap, edemmap, chisq, logT_bins
  • Optionally writes a sidecar JSON with metadata (inputs, sizes, shapes, params)

Typical usage
-------------
>>> write_goldens(
...     data_dir="data/np32",
...     out_npz="data/golden/256x256/baseline.npz",
...     out_meta="data/golden/256x256/baseline.json",
...     sizes="256,256",
...     nmu=42
... )
"""

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from src.common.dataio import default_files, load_np_stack, frame_for_solver
from src.common.solver import solve_tile_all

# -------------------------
# Helpers
# -------------------------


def _parse_sizes(s: Optional[str]) -> Optional[Tuple[int, int]]:
    """
    Parse a size string into (H, W).

    Accepts forms like:
      - "256"        → (256, 256)
      - "256x512"    → (256, 512)
      - "256,512"    → (256, 512)

    Returns
    -------
    (int, int) | None
        Parsed crop or None if input is falsy.
    """
    if not s:
        return None
    s = s.lower().replace("x", ",").replace(" ", "")
    parts = [p for p in s.split(",") if p]
    if len(parts) == 1:
        h = int(parts[0])
        return (h, h)
    if len(parts) >= 2:
        return (int(parts[0]), int(parts[1]))
    return None


def _save_npz(
    path: Path,
    *,
    dem: np.ndarray,
    edem: np.ndarray,
    chisq: np.ndarray,
    logt: np.ndarray,
) -> None:
    """
    Persist solver outputs into a compressed NPZ using standard keys.

    Keys:
      - demmap
      - edemmap
      - chisq
      - logT_bins
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, demmap=dem, edemmap=edem, chisq=chisq, logT_bins=logt)


def _save_meta_json(path: Path, meta: dict) -> None:
    """Write a JSON sidecar (pretty-printed) with run metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


# -------------------------
# Main API
# -------------------------


def write_goldens(
    *,
    data_dir: Path | str,
    out_npz: Path | str,
    out_meta: Optional[Path | str] = None,
    ext: str = "*.npz",
    index: int | slice = -1,
    sizes: Optional[Tuple[int, int]] | Optional[str] = None,
    nmu: int = 42,
    nt: Optional[int] = None,
    extra_meta: Optional[dict] = None,
) -> dict:
    """
    Build a golden NPZ from the current solver.

    Parameters
    ----------
    data_dir : str | pathlib.Path
        Folder with input .npz stacks (each must contain 'bands' shaped (6,H,W)).
    out_npz : str | pathlib.Path
        Output path for the golden NPZ. Arrays are saved under keys:
        demmap, edemmap, chisq, logT_bins.
    out_meta : str | pathlib.Path | None, default None
        Optional JSON sidecar with metadata (inputs, sizes, shapes, params).
    ext : str, default "*.npz"
        Glob used to discover input files in `data_dir`.
    index : int | slice, default -1
        Which frame(s) to load when stacking. Negative indices allowed.
        Slices like `1:4` are accepted if passed by a CLI wrapper.
    sizes : (int, int) | str | None, default None
        Optional crop (H, W). If str, accepts "H", "H,W", or "HxW".
    nmu : int, default 42
        Regularization parameter forwarded to the solver.
    nt : int | None, default None
        Optional DEM bin count override forwarded to the solver.
    extra_meta : dict | None, default None
        Extra metadata merged into the sidecar JSON if `out_meta` is provided.

    Returns
    -------
    dict
        A summary suitable for logging or tests:
        {
          "files": [...],
          "index": "...",
          "sizes": [H, W] | None,
          "nmu": int,
          "nt": int | None,
          "shapes": {
              "dem": [F, H, W, NT] or [H, W, NT],
              "edem": [...],
              "chisq": [...],
              "logT_bins": NT
          },
          "npz_path": "..."
        }

    Raises
    ------
    FileNotFoundError
        If no input files are found.
    ValueError
        If loaded NPZ arrays are missing or have incompatible shapes.
    """
    files = default_files(data_dir, ext=ext)
    if not files:
        raise FileNotFoundError(f"No files found in {data_dir!r} matching {ext!r}")

    # Normalize sizes
    if isinstance(sizes, str):
        sizes = _parse_sizes(sizes)

    # Load and normalize (H, W, 6)
    stack = load_np_stack(files, idx=index, channels_last=True)
    frame = frame_for_solver(stack, 0)

    if sizes:
        H, W = sizes
        frame = frame[:H, :W, :]

    # Compute with the unified solver
    dem, edem, chisq, logt = solve_tile_all(frame, nmu=nmu, nt=nt)

    # Write NPZ
    out_npz = Path(out_npz)
    _save_npz(out_npz, dem=dem, edem=edem, chisq=chisq, logt=logt)

    # Optional sidecar
    summary = {
        "files": [str(p) for p in files],
        "index": str(index),
        "sizes": list(sizes) if sizes else None,
        "nmu": int(nmu),
        "nt": int(nt) if nt is not None else None,
        "shapes": {
            "dem": list(dem.shape),
            "edem": list(edem.shape),
            "chisq": list(chisq.shape),
            "logT_bins": int(logt.shape[0]),
        },
        "npz_path": str(out_npz),
    }
    if extra_meta:
        summary.update(extra_meta)

    if out_meta:
        _save_meta_json(Path(out_meta), summary)

    return summary
