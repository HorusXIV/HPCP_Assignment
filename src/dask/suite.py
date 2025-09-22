# src/dask/suite.py
from __future__ import annotations
"""
Dask execution suite for the DEM baseline.

This module provides a default workload that:
  1) discovers NPZ frames on disk,
  2) builds a lazy (F, H, W, 6) Dask array with optional cropping/tiling,
  3) maps the (vendor-backed) CPU solver over tiles, and
  4) writes benchmark artifacts under ./benchmarking/dask/<timestamp>/.

It is discovered automatically by `python -m src.dask.main` as the default task
(`src.dask.suite:run`) if no explicit `--task` is given.
"""

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import dask.array as da
from dask.distributed import Client

from src.common.dataio import default_files, build_lazy_npz_stack
from src.common.profiling import (
    Profiler,
    write_bench_row,
    write_run_card_md,
    write_json,
)
from src.common.profiling.io_helpers import set_bench_outdir
from src.common.solver import solve_tile_all, get_logt_bins_once


# ---------------- helpers: bench dir ----------------


def _base_root_default() -> Path:
    """Return the default base folder for Dask runs: `<cwd>/benchmarking/dask`."""
    return Path.cwd() / "benchmarking" / "dask"


def _ensure_bench_dir(base_root: Optional[Union[str, Path]]) -> Tuple[Path, str]:
    """
    Create `<base>/<YYYYMMDD-HHMMSS>/` and route bench.csv there.

    Parameters
    ----------
    base_root : str | Path | None
        Base directory to create the timestamped run folder inside. If None,
        uses `_base_root_default()`.

    Returns
    -------
    (outdir, stamp) : (Path, str)
        The created directory and its timestamp suffix.
    """
    import datetime as _dt

    base = Path(base_root) if base_root else _base_root_default()
    base.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = base / stamp
    outdir.mkdir(parents=True, exist_ok=True)
    set_bench_outdir(outdir)  # bench.csv lands in this folder
    return outdir, stamp


# ---------------- helpers: parsing ----------------

NumOrStrOrSeq = Optional[Union[int, str, Sequence[int]]]


def _coerce_pair(val: NumOrStrOrSeq, *, default: Tuple[int, int]) -> Tuple[int, int]:
    """
    Normalize a variety of inputs into a (H, W) integer pair.

    Accepted forms
    --------------
    - None                     → default
    - int                      → (v, v)
    - str  like "HxW", "H,W", "H x W", or single "H"
    - tuple/list               → (H, W) or single-element (H,) → (H, H)

    Parameters
    ----------
    val : int | str | Sequence[int] | None
        Size or tile spec.
    default : (int, int)
        Fallback when `val` is None.

    Returns
    -------
    (H, W) : tuple[int, int]
    """
    if val is None:
        return default

    if isinstance(val, (tuple, list)):
        if len(val) == 0:
            return default
        if len(val) == 1:
            v = int(val[0])
            return (v, v)
        return (int(val[0]), int(val[1]))

    if isinstance(val, int):
        return (val, val)

    s = str(val).strip().lower()
    s = s.strip("()[] ")
    for sep in ("x", ","):
        if sep in s:
            a, b = [t for t in s.replace(" ", "").split(sep) if t]
            return (int(a), int(b))
    parts = [p for p in s.replace(" ", "").split() if p]
    if len(parts) >= 2:
        return (int(parts[0]), int(parts[1]))
    v = int(s)
    return (v, v)


def _parse_hw(sizes: NumOrStrOrSeq, *, default=(2048, 2048)) -> Tuple[int, int]:
    """Parse/canonicalize crop size into (H, W)."""
    return _coerce_pair(sizes, default=default)


def _parse_tile(tile: NumOrStrOrSeq, *, default=(256, 256)) -> Tuple[int, int]:
    """Parse/canonicalize tile size into (Th, Tw)."""
    return _coerce_pair(tile, default=default)


# ---------------- helpers: build frames ----------------


def _build_frames_real(
    data_dir: Union[str, Path],
    ext: str,
    idx: Union[str, int],
    sizes: Tuple[int, int],
    tile: Tuple[int, int],
) -> da.Array:
    """
    Build a lazy Dask array of frames from NPZ files containing 'bands' (6, H, W).

    Parameters
    ----------
    data_dir : str | Path
        Folder with NPZ stacks.
    ext : str
        Glob pattern for inputs (e.g., '*.npz').
    idx : 'all' | int
        Which frames to use: 'all' for all files, or a single 0-based index.
    sizes : (int, int)
        Crop size (H, W) to apply to each frame.
    tile : (int, int)
        Tile (chunk) size (Th, Tw) for dask.

    Returns
    -------
    darr : dask.array.Array
        Lazy array shaped (F, Hc, Wc, 6) with chunks (1, Th, Tw, 6).
    """
    H, W = sizes
    Th, Tw = tile

    files = default_files(data_dir, ext=ext)
    if isinstance(idx, str):
        if idx.lower() == "all":
            frame_idx = list(range(len(files)))
        else:
            frame_idx = [int(idx)]
    else:
        frame_idx = [int(idx)]

    if not frame_idx:
        raise RuntimeError("No input frames resolved from data_dir/ext/idx.")

    # Select only the files we want, then include ALL of those (`idx=slice(None)`)
    # in the lazy stack construction.
    selected_files = [files[i] for i in frame_idx]
    darr = build_lazy_npz_stack(
        selected_files,
        idx=slice(None),  # consume all of the subset
        crop_hw=(H, W),
        tile_hw=(Th, Tw),
    )
    return darr


# ---------------- mapping: solver over tiles ----------------


def _map_solver(
    darr: da.Array, Th: int, Tw: int, NT: int, nmu: int
) -> Tuple[da.Array, da.Array, da.Array]:
    """
    Map the solver across tiles of a (F, H, W, 6) array.

    Returns three arrays:
      dem   : (F, H, W, NT)
      edem  : (F, H, W, NT)
      chisq : (F, H, W)

    Parameters
    ----------
    darr : dask.array.Array
        Input array (F, H, W, 6), chunked as (1, Th, Tw, 6).
    Th, Tw : int
        Tile shape (height, width). Used for output chunk hints.
    NT : int
        DEM temperature bins (solver setting).
    nmu : int
        Regularization/solver parameter.

    Returns
    -------
    (dem, edem, chisq) : tuple[dask.array.Array, dask.array.Array, dask.array.Array]
        DEM volumes and chi² as lazy arrays.
    """

    def _solve_dem(block: np.ndarray, nmu: int, NT: int) -> np.ndarray:
        tile = block[0] if block.ndim == 4 else block  # (Th, Tw, 6)
        dem, _edem, _chisq, _ = solve_tile_all(tile, nmu=nmu, nt=NT)
        return dem[None, ...]

    def _solve_edem(block: np.ndarray, nmu: int, NT: int) -> np.ndarray:
        tile = block[0] if block.ndim == 4 else block
        _dem, edem, _chisq, _ = solve_tile_all(tile, nmu=nmu, nt=NT)
        return edem[None, ...]

    def _solve_chisq(block: np.ndarray, nmu: int, NT: int) -> np.ndarray:
        tile = block[0] if block.ndim == 4 else block
        _dem, _edem, chisq, _ = solve_tile_all(tile, nmu=nmu, nt=NT)
        return chisq[None, :]

    dem_da = darr.map_blocks(
        _solve_dem,
        nmu,
        NT,
        dtype=np.float32,
        chunks=(1, Th, Tw, NT),
        drop_axis=(3,),
        new_axis=(3,),
    )
    edem_da = darr.map_blocks(
        _solve_edem,
        nmu,
        NT,
        dtype=np.float32,
        chunks=(1, Th, Tw, NT),
        drop_axis=(3,),
        new_axis=(3,),
    )
    chisq_da = darr.map_blocks(
        _solve_chisq, nmu, NT, dtype=np.float32, chunks=(1, Th, Tw), drop_axis=(3,)
    )
    return dem_da, edem_da, chisq_da


# ---------------- Default task (auto-discovered by dask.main) ----------------


def run(*, client: Client, args) -> None:
    """
    Default Dask workload entry point.

    Steps
    -----
    1. Parse crop/tile sizes and common flags from `args`.
    2. Build a lazy frame stack from NPZ files (cropped & chunked).
    3. Map the CPU solver over tiles using `map_blocks`.
    4. Profile compute section and write a bench row + run card.

    Parameters
    ----------
    client : dask.distributed.Client
        Active Dask client provided by the runner.
    args : argparse.Namespace-like
        CLI arguments parsed by `src.dask.cli`.

    Notes
    -----
    Artifacts are written into `./benchmarking/dask/<timestamp>/`.
    """
    H, W = _parse_hw(getattr(args, "sizes", None), default=(2048, 2048))
    Th, Tw = _parse_tile(getattr(args, "tile", None), default=(256, 256))
    nmu = int(getattr(args, "nmu", 42) or 42)
    verify = bool(getattr(args, "verify", False))
    bench_root_arg = (
        getattr(args, "bench_root", None) if hasattr(args, "bench_root") else None
    )
    data_dir = getattr(args, "data_dir", None) or "data/np32"
    ext = getattr(args, "ext", "*.npz") or "*.npz"
    idx = getattr(args, "idx", "all")

    bench_dir, stamp = _ensure_bench_dir(bench_root_arg)

    # DEM binning used by the solver
    NT, _logT = get_logt_bins_once(nmu=nmu)

    # Build input frames (real data only)
    darr = _build_frames_real(data_dir, ext, idx, (H, W), (Th, Tw))
    frame_idx = list(range(darr.shape[0]))

    # Map solver
    dem_da, edem_da, chisq_da = _map_solver(darr, Th, Tw, NT, nmu)

    # Profile + compute
    with Profiler(
        client=client, benchdir=bench_dir, stamp=stamp, enable_perf_html=True
    ) as prof:
        prof.section("compute", start=True)
        with prof.perf_context():
            dem, edem, chisq = da.compute(dem_da, edem_da, chisq_da)
        prof.section("compute", start=False)

    # Bench row
    tiles_per_frame = int(np.ceil(H / Th) * np.ceil(W / Tw))
    total_s = float(prof._sections.get("compute", (0.0, 0.0))[1])  # type: ignore[attr-defined]
    bench = dict(
        stamp=stamp,
        mode="dask-cpu",
        frames=len(frame_idx),
        H=H,
        W=W,
        Th=Th,
        Tw=Tw,
        nmu=int(nmu),
        total_seconds=round(total_s, 6),
        tiles_per_frame=tiles_per_frame,
        dem_finite_frac=float(np.isfinite(dem).mean()) if dem.size else float("nan"),
        edem_finite_frac=float(np.isfinite(edem).mean()) if edem.size else float("nan"),
    )
    write_bench_row(**bench)
    write_run_card_md(
        outdir=bench_dir,
        stamp=stamp,
        bench_row=bench,
        env=None,
        notes=[f"verify={verify}"],
    )
    write_json(bench, bench_dir / f"bench_{stamp}.json")

    # Console summary
    px = H * W * len(frame_idx)
    rate = px / bench["total_seconds"] if bench["total_seconds"] > 0 else float("nan")
    print(
        f"[Dask] frames={len(frame_idx)} tiles/frame={tiles_per_frame} crop={H}x{W} "
        f"total={bench['total_seconds']:.3f}s ~ {rate / 1e6:.2f} MPix/s -> artifacts: {bench_dir}"
    )


__all__ = ["run"]
