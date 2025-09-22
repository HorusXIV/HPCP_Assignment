# src/common/verification/check.py
from __future__ import annotations
"""
Utilities to compute a DEM result from NPZ inputs and verify it against goldens.

This module offers three convenience layers:

1) `compute_frame`:
   - Loads one frame from a sequence of NPZ files (each with 'bands' shaped (6,H,W)),
     applies an optional crop, and runs the shared solver to produce
     (dem, edem, chisq, logT_bins).

2) `verify_against_golden`:
   - Computes a result from user-specified inputs and compares it to a provided
     golden NPZ using `compare_to_golden`. Returns the comparison report and
     sets `verified=True` on success.

3) `verify_dataset_to_json`:
   - Discovers NPZ files under a directory, performs a single verification
     (by default on the last frame via `index=-1`, or "all" if your dataio
     layer interprets that), and optionally writes the JSON report to disk.

Notes
-----
- This module builds on shared data I/O (`default_files`, `load_np_stack`,
  `frame_for_solver`) and the CPU solver (`solve_tile_all`).
- Indices and sizes follow project-wide conventions: `index=-1` selects the
  last frame unless your stack loader treats -1 as "all"; `sizes=(H,W)` crops
  the frame after loading.
"""

import json
from pathlib import Path
from typing import Optional, Sequence, Tuple

from src.common.dataio import default_files, load_np_stack, frame_for_solver
from src.common.solver import solve_tile_all
from src.common.verification.verify import compare_to_golden


def _parse_sizes(s: Optional[str]) -> Optional[Tuple[int, int]]:
    """
    Parse size strings like "HxW", "H,W", or "H" (square) into a (H, W) tuple.

    Parameters
    ----------
    s : str | None
        Size specification. Examples: "256", "256x512", "256,512".

    Returns
    -------
    (int, int) | None
        Parsed (H, W) or None if `s` is falsy.
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


def compute_frame(
    files: Sequence[Path] | Sequence[str],
    *,
    index: int | slice = -1,
    sizes: Optional[Tuple[int, int]] = None,
    nmu: int = 42,
    nt: Optional[int] = None,
):
    """
    Load one frame from .npz files and compute (dem, edem, chisq, logT_bins).

    Parameters
    ----------
    files : Sequence[pathlib.Path | str]
        List of NPZ files to load from (must contain 'bands': (6,H,W)).
    index : int | slice, default -1
        Which frame to take after stacking. Negative indices allowed.
    sizes : (int, int) | None, default None
        Optional crop (H, W) applied after extracting the frame.
    nmu : int, default 42
        Regularization parameter forwarded to the solver.
    nt : int | None, default None
        Optional DEM bin count override forwarded to the solver.

    Returns
    -------
    (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        (dem, edem, chisq, logT_bins) as produced by `solve_tile_all`.
    """
    stack = load_np_stack(files, idx=index, channels_last=True)
    frame = frame_for_solver(stack, 0)  # (H,W,6)
    if sizes:
        H, W = sizes
        frame = frame[:H, :W, :]
    dem, edem, chisq, logt = solve_tile_all(frame, nmu=nmu, nt=nt)
    return dem, edem, chisq, logt


def verify_against_golden(
    golden_npz: Path | str,
    *,
    files: Sequence[Path] | Sequence[str],
    index: int | slice = -1,
    sizes: Optional[Tuple[int, int]] = None,
    nmu: int = 42,
    nt: Optional[int] = None,
) -> dict:
    """
    Compute a result and compare it against a given golden .npz.

    Parameters
    ----------
    golden_npz : pathlib.Path | str
        Path to the golden NPZ containing reference arrays.
    files : Sequence[pathlib.Path | str]
        Input NPZ files to compute from (see `compute_frame`).
    index, sizes, nmu, nt :
        Forwarded to `compute_frame`.

    Returns
    -------
    dict
        Comparison report from `compare_to_golden` augmented with
        `{"verified": True}` to indicate the comparison ran to completion.
    """
    dem, edem, chisq, logt = compute_frame(
        files, index=index, sizes=sizes, nmu=nmu, nt=nt
    )
    rep = compare_to_golden(
        Path(golden_npz), demmap=dem, edemmap=edem, chisq=chisq, logT_bins=logt
    )
    rep["verified"] = True
    return rep


def verify_dataset_to_json(
    data_dir: Path | str,
    *,
    golden_npz: Path | str,
    ext: str = "*.npz",
    index: int | slice = -1,
    sizes: Optional[Tuple[int, int]] = None,
    nmu: int = 42,
    nt: Optional[int] = None,
    json_out: Optional[Path | str] = None,
) -> dict:
    """
    Discover NPZ files, verify once against a golden, and optionally write JSON.

    Parameters
    ----------
    data_dir : pathlib.Path | str
        Directory to search for input NPZ files.
    golden_npz : pathlib.Path | str
        Path to golden NPZ for comparison.
    ext : str, default "*.npz"
        Glob pattern for discovery (non-recursive).
    index, sizes, nmu, nt :
        Forwarded to `verify_against_golden`.
    json_out : pathlib.Path | str | None, default None
        If provided, write the comparison report as pretty JSON.

    Returns
    -------
    dict
        The verification report.
    """
    files = default_files(data_dir, ext=ext)
    if not files:
        raise FileNotFoundError(f"No files found in {data_dir!r} matching {ext!r}")

    rep = verify_against_golden(
        golden_npz, files=files, index=index, sizes=sizes, nmu=nmu, nt=nt
    )
    if json_out:
        Path(json_out).write_text(json.dumps(rep, indent=2), encoding="utf-8")
    return rep
