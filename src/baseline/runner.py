# src/baseline/runner.py
from __future__ import annotations

"""
Baseline CPU benchmark runner.

This module provides the serial (per-tile) baseline that:
  • Discovers input NPZ stacks
  • Crops/tiles them to the requested size
  • Solves each tile via the shared solver (vendor-aware)
  • Records basic quality/throughput metrics and writes benchmark artifacts

Artifacts are written under a timestamped directory, e.g.:
  ./benchmarking/baseline/20250101-123456/
"""

import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from src.common.dataio import default_files
from src.common.profiling import (
    Profiler,
    write_bench_row,
    write_run_card_md,
    write_json,
)
from src.common.profiling.io_helpers import set_bench_outdir
from src.common.solver import solve_tile_all, get_logt_bins_once  # shared CPU kernels


# ---------------------------------------------------------------------
# Benchmark root helpers
# ---------------------------------------------------------------------


def _base_root_default() -> Path:
    """
    Return the default benchmark root.

    Returns
    -------
    Path
        The path `./benchmarking/baseline` under the current working directory.
    """
    return Path.cwd() / "benchmarking" / "baseline"


def _ensure_timestamped_root(base_root: Optional[Union[str, Path]]) -> Tuple[Path, str]:
    """
    Create a timestamped output directory and configure CSV destinations.

    The directory structure is:
        <base>/<YYYYMMDD-HHMMSS>/

    Parameters
    ----------
    base_root : str | Path | None
        Base directory for benchmark artifacts. If None, uses the default
        `./benchmarking/baseline`.

    Returns
    -------
    (Path, str)
        Tuple of (created directory, timestamp string).

    Side Effects
    ------------
    Calls `set_bench_outdir(out)` so CSV/JSON helpers write into this directory.
    """
    base = Path(base_root) if base_root else _base_root_default()
    base.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = base / stamp
    out.mkdir(parents=True, exist_ok=True)
    set_bench_outdir(out)  # bench.csv / profiling_dask.csv go here too
    return out, stamp


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------


def _npz_band_shape(path: Path) -> Tuple[int, int, int]:
    """
    Inspect an NPZ stack and return the shape of its 'bands' array.

    Expects a (6, H, W) array stored under key 'bands'.

    Parameters
    ----------
    path : Path
        Path to the NPZ file.

    Returns
    -------
    (int, int, int)
        Tuple (C, H, W). Validates that C == 6.

    Raises
    ------
    ValueError
        If the array is missing, has unexpected rank, or channels != 6.
    """
    with np.load(path, allow_pickle=False) as z:
        a = z["bands"]
        if a.ndim != 3 or a.shape[0] != 6:
            raise ValueError(
                f"{path} expected 'bands' with shape (6,H,W), got {a.shape}"
            )
        return int(a.shape[0]), int(a.shape[1]), int(a.shape[2])


def _as_int(val: object, default: int) -> int:
    """
    Convert a value to int, returning a default on failure.

    Parameters
    ----------
    val : object
        Value to convert.
    default : int
        Fallback value.

    Returns
    -------
    int
        Converted integer or `default` if conversion fails.
    """
    try:
        return int(val)
    except Exception:
        return default


def _quality_stats(dem0: np.ndarray, chisq0: np.ndarray) -> Dict[str, float]:
    """
    Compute simple quality statistics for quick sanity checks.

    Parameters
    ----------
    dem0 : np.ndarray
        DEM slice or cube (any shape). Finite/positive fractions are computed
        over all elements.
    chisq0 : np.ndarray
        Chi-square map (any shape). Median/mean are computed over all elements.

    Returns
    -------
    dict
        {
          "dem_finite_frac": float,
          "dem_positive_frac": float,
          "chisq_median": float,
          "chisq_mean": float,
        }
    """
    finite_dem = np.isfinite(dem0)
    positive_dem = dem0 > 0
    return {
        "dem_finite_frac": float(finite_dem.mean()) if dem0.size else float("nan"),
        "dem_positive_frac": float(positive_dem.mean()) if dem0.size else float("nan"),
        "chisq_median": float(np.nanmedian(chisq0)) if chisq0.size else float("nan"),
        "chisq_mean": float(np.nanmean(chisq0)) if chisq0.size else float("nan"),
    }


def _parse_tile_arg(tile: Union[str, Tuple[int, int], None]) -> Tuple[int, int]:
    """
    Parse a tile argument into (Th, Tw).

    Accepts:
      • None                → (256, 256)
      • tuple(int, int)     → returned as (int(tile[0]), int(tile[1]))
      • string "HxW" or "H,W"
      • string "N"          → (N, N)

    Parameters
    ----------
    tile : str | (int, int) | None

    Returns
    -------
    (int, int)
        Tile height and width.
    """
    if tile is None:
        return 256, 256
    if isinstance(tile, tuple):
        return int(tile[0]), int(tile[1])
    s = str(tile)
    if "x" in s.lower():
        a, b = s.lower().split("x")
        return int(a), int(b)
    if "," in s:
        a, b = s.split(",", 1)
        return int(a), int(b)
    v = int(s)
    return v, v


# ---------------------------------------------------------------------
# Public entrypoint (CLI-compatible)
# ---------------------------------------------------------------------


def run_benchmark(
    *,
    # dataset
    use_synthetic: bool = False,
    data_dir: Union[str, Path, None] = None,
    ext: Union[str, Sequence[str]] = "*.npz",
    idx: Union[str, Sequence[int]] = "-1",
    # sizing / compute
    sizes: Sequence[int] = (1024,),  # accepts [H] or [H, W]
    tile: Union[str, Tuple[int, int], None] = None,
    nmu: int = 42,
    repeats: int = 1,  # kept for CLI parity (no-op beyond one pass)
    # outputs (base folder; timestamp subfolder is created automatically)
    outdir: Union[str, Path, None] = None,        # preferred knob
    bench_root: Union[str, Path, None] = None,    # alias (if provided, used as base)
    # misc
    verify: bool = False,
    golden_root: Union[str, Path, None] = None,
    device_str: str = "cpu",
    nvtx_label: Optional[str] = None,
    threads_cap: Optional[int] = None,
    runtime_enforce: bool = False,
) -> Dict[str, Union[str, float, int]]:
    """
    Run the serial baseline benchmark on CPU and write artifacts.

    This function:
      1. Discovers input stacks (NPZ files with 'bands': (6,H,W)).
      2. Selects frames by `idx`:
           - integer -> that frame
           - sequence of ints -> those frames
           - "all" or "-1"    -> all frames
      3. Determines output crop size from `sizes` or the first file.
      4. Tiles each frame by `tile` and solves via `solve_tile_all`.
      5. Records throughput and simple quality metrics.
      6. Writes a bench row (CSV), JSON, and a run card (markdown).

    Parameters
    ----------
    use_synthetic : bool, optional
        If True, use synthetic inputs (not implemented for baseline).
    data_dir : str | Path | None, optional
        Root directory containing NPZ stacks (default: "data/np32").
    ext : str | Sequence[str], optional
        Glob(s) for file discovery (default: "*.npz").
    idx : str | Sequence[int], optional
        Frame selector. "all" or "-1" means all frames. Integer selects one.
    sizes : Sequence[int], optional
        Either [H] (square) or [H, W]. If empty/None, inferred from first file.
    tile : str | (int, int) | None, optional
        Tile size as "ThxTw", "T,T", or single int "T" (square). Default (256,256).
    nmu : int, optional
        Regularization / temperature resolution knob (forwarded to solver).
    repeats : int, optional
        For CLI symmetry; baseline currently executes a single pass.
    outdir : str | Path | None, optional
        Artifact base directory (alias of `bench_root`).
    bench_root : str | Path | None, optional
        Artifact base directory; if set, takes precedence over `outdir`.
    verify : bool, optional
        Whether to verify results against goldens (placeholder here).
    golden_root : str | Path | None, optional
        Root directory containing golden references.
    device_str : str, optional
        Execution device hint (unused in CPU baseline; kept for parity).
    nvtx_label : str | None, optional
        Optional NVTX label (unused here; kept for parity).
    threads_cap : int | None, optional
        Optional threading cap (unused here; kept for parity).
    runtime_enforce : bool, optional
        Skip/enable runtime env enforcement (unused; kept for parity).

    Returns
    -------
    dict
        Summary including:
          {
            "bench_root": str, "stamp": str,
            "frames": int, "H": int, "W": int, "Th": int, "Tw": int,
            "seconds": float
          }

    Raises
    ------
    NotImplementedError
        If `use_synthetic=True` (baseline path does not implement synthetic).
    FileNotFoundError
        If no input files are found under `data_dir` with the given `ext`.
    IndexError
        If a requested frame index is out of range.
    ValueError
        If an input NPZ has unexpected shape for 'bands'.
    """
    if use_synthetic:
        raise NotImplementedError(
            "Baseline synthetic path not wired; supply NPZs instead."
        )

    # ---------------- inputs discovery ----------------
    data_dir = data_dir or "data/np32"
    files = default_files(directory=data_dir, ext=ext)  # correct signature
    if not files:
        patterns = [ext] if isinstance(ext, str) else list(ext)
        raise FileNotFoundError(f"No files found in {data_dir} matching {patterns}")

    # Select frames
    if isinstance(idx, str):
        if idx.lower() in {"all", "-1"}:
            frame_idx: Sequence[int] = list(range(len(files)))
        else:
            i = _as_int(idx, 0)
            if not (0 <= i < len(files)):
                raise IndexError(f"--idx {i} out of range (0..{len(files) - 1})")
            frame_idx = [i]
    else:
        # sequence: take as-is but validate
        frame_idx = [int(i) for i in idx]
        for i in frame_idx:
            if not (0 <= i < len(files)):
                raise IndexError(f"--idx {i} out of range (0..{len(files) - 1})")

    # Determine crop size
    if sizes and len(sizes) >= 2:
        H, W = int(sizes[0]), int(sizes[1])
    elif sizes and len(sizes) == 1:
        H = W = int(sizes[0])
    else:
        _, H0, W0 = _npz_band_shape(Path(files[0]))
        H, W = H0, W0

    Th, Tw = _parse_tile_arg(tile)

    # ---------------- outputs root (project-root timestamped) ----------------
    base = bench_root or outdir  # either can set the base; both optional
    bench_dir, stamp = _ensure_timestamped_root(base)

    # ---------------- load frames ----------------
    def _load_frame_hw6(p: Path, H: int, W: int) -> np.ndarray:
        """
        Load a single NPZ and return a cropped (H, W, 6) float32 array.
        """
        with np.load(p, allow_pickle=False) as z:
            bands = z["bands"].astype(np.float32)  # (6,H0,W0)
        b = bands[:, :H, :W]
        return np.moveaxis(b, 0, -1)  # (H,W,6)

    frames = [_load_frame_hw6(Path(files[i]), H, W) for i in frame_idx]  # (H,W,6)
    stack = np.stack(frames, axis=0)  # (F,H,W,6)
    F = int(stack.shape[0])

    # ---------------- solver config ----------------
    NT, _logT_bins = get_logt_bins_once(nmu=nmu)

    # ---------------- profiling & compute ----------------
    with Profiler(
        client=None, benchdir=bench_dir, stamp=stamp, enable_perf_html=False
    ) as prof:
        prof.section("compute", start=True)

        dem = np.empty((F, H, W, NT), dtype=np.float32)
        edem = np.empty_like(dem)
        chisq = np.empty((F, H, W), dtype=np.float32)

        for fi in range(F):
            for y0 in range(0, H, Th):
                y1 = min(H, y0 + Th)
                for x0 in range(0, W, Tw):
                    x1 = min(W, x0 + Tw)
                    tile6 = stack[fi, y0:y1, x0:x1, :]  # (th,tw,6)
                    _dem, _edem, _chisq, _ = solve_tile_all(tile6, nmu=nmu)
                    dem[fi, y0:y1, x0:x1, :] = _dem
                    edem[fi, y0:y1, x0:x1, :] = _edem
                    chisq[fi, y0:y1, x0:x1] = _chisq

        prof.section("compute", start=False)

    # ---------------- verification (placeholder) ----------------
    verify_ok = bool(verify)
    reports: List[str] = []
    if verify and golden_root:
        reports.append("Verification placeholder — integrate golden check here.")
        verify_ok = True

    # ---------------- bench row & artifacts ----------------
    tiles_per_frame = math.ceil(H / Th) * math.ceil(W / Tw)
    tiles_total = tiles_per_frame * F
    total_s = float((getattr(prof, "_sections", {}).get("compute", (0.0, 0.0))[1]))
    tiles_per_s = tiles_total / total_s if total_s > 0 else float("nan")
    dems_per_s = (F * H * W) / total_s if total_s > 0 else float("nan")

    # basic quality snapshot on first frame
    q = _quality_stats(
        dem[0, ..., 0] if F else np.empty(0), chisq[0] if F else np.empty(0)
    )
    bench = dict(
        stamp=stamp,
        mode="baseline-cpu",
        frames=int(F),
        H=H,
        W=W,
        Th=Th,
        Tw=Tw,
        nmu=int(nmu),
        total_seconds=round(total_s, 6),
        tiles_per_s=float(tiles_per_s),
        dems_per_s=float(dems_per_s),
        verify=bool(verify),
        verify_ok=bool(verify_ok),
        **q,
    )

    write_bench_row(**bench)
    write_run_card_md(
        outdir=bench_dir, stamp=stamp, bench_row=bench, env=None, notes=reports
    )
    write_json(bench, bench_dir / f"bench_{stamp}.json")

    print(
        f"[Baseline] frames={F} tiles/frame={tiles_per_frame} crop={H}x{W} "
        f"total={bench['total_seconds']:.3f}s -> artifacts: {bench_dir}"
    )

    return {
        "bench_root": str(bench_dir),
        "stamp": stamp,
        "frames": F,
        "H": H,
        "W": W,
        "Th": Th,
        "Tw": Tw,
        "seconds": bench["total_seconds"],
    }
