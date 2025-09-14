# src/common/profiling.py
from __future__ import annotations
import os, sys, json, time, platform, statistics as stats, csv, io
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Callable

import numpy as np

# --- Import the provided solver + internals (baseline/vendor) ---
from src.baseline.vendor.dn2dem_pos import dn2dem_pos
from src.baseline.vendor.demmap_pos import demmap_pos as _demmap_pos, dem_pix as _dem_pix


# ----------------------------
# Thread caps (baseline fairness)
# ----------------------------
def set_single_thread_caps() -> None:
    """Set common BLAS/OpenMP thread caps to 1 (idempotent)."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


# ----------------------------
# Core runner (vanilla solver)
# ----------------------------
def run_dn2dem(
    frame_6hw: np.ndarray,
    T_RESP: np.ndarray,
    T_RESP_LOGT: np.ndarray,
    TEMPS: np.ndarray,
    *,
    nmu: int = 42,
    dtype: np.dtype = np.float32,
    solver_fn: Callable = dn2dem_pos,
):
    """
    Thin wrapper around the provided solver: prepares inputs and calls `solver_fn`.
    frame_6hw : (6, H, W)   -> converts to (H, W, 6) and builds simple errors.
    Returns   : (demmap, edemmap, logT_bins, chisq, dn_reg)
    """
    f = np.moveaxis(frame_6hw, 0, -1).astype(dtype, copy=False)  # (H,W,6)
    f = np.clip(np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0), 0, None)
    e = np.sqrt(f, dtype=np.float32) + 1e-6
    return solver_fn(f, e, T_RESP, T_RESP_LOGT, TEMPS, nmu=nmu)


def time_one(
    frame_6hw: np.ndarray,
    T_RESP: np.ndarray,
    T_RESP_LOGT: np.ndarray,
    TEMPS: np.ndarray,
    *,
    nmu: int = 42,
    dtype: np.dtype = np.float32,
    solver_fn: Callable = dn2dem_pos,
) -> Tuple[float, float]:
    """Run once and return (elapsed_seconds, DEMs_per_second)."""
    t0 = time.perf_counter()
    _ = run_dn2dem(frame_6hw, T_RESP, T_RESP_LOGT, TEMPS, nmu=nmu, dtype=dtype, solver_fn=solver_fn)
    dt = time.perf_counter() - t0
    H, W = frame_6hw.shape[1], frame_6hw.shape[2]
    return dt, (H * W) / dt


# ----------------------------
# Wall-clock benchmark
# ----------------------------
def benchmark_wallclock(
    STACK: np.ndarray,
    T_RESP: np.ndarray,
    T_RESP_LOGT: np.ndarray,
    TEMPS: np.ndarray,
    *,
    sizes: Iterable[int] = (14, 64, 256, 1024),
    repeats: int = 5,
    nmu: int = 42,
    dtype: np.dtype = np.float32,
    outdir: Path | str = "benchmark_out",
    solver_fn: Callable = dn2dem_pos,
) -> List[dict]:
    """
    Run repeated wall-clock timings over several square crops; write CSV + Markdown.
    Returns the list of result rows.
    """
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    rows: List[dict] = []

    for sz in sizes:
        sz = int(min(sz, STACK.shape[2], STACK.shape[3]))
        frame = STACK[0, :, :sz, :sz]
        # warm-up
        _ = time_one(frame, T_RESP, T_RESP_LOGT, TEMPS, nmu=nmu, dtype=dtype, solver_fn=solver_fn)
        # repeats
        dts, thr = [], []
        for _ in range(repeats):
            dt, dems = time_one(frame, T_RESP, T_RESP_LOGT, TEMPS, nmu=nmu, dtype=dtype, solver_fn=solver_fn)
            dts.append(dt); thr.append(dems)

        row = dict(
            size=f"{sz}x{sz}",
            H=sz, W=sz, nf=int(frame.shape[0]),
            repeats=repeats,
            time_mean=float(np.mean(dts)), time_std=float(np.std(dts)),
            time_median=float(stats.median(dts)),
            dems_per_s_mean=float(np.mean(thr)),
            dems_per_s_median=float(stats.median(thr)),
            nmu=int(nmu), dtype=str(dtype),
        )
        rows.append(row)

    # CSV
    csv_path = outdir / "baseline_wallclock.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # Markdown
    md_path = outdir / "summary.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("| Size | repeats | time_median [s] | DEMs/s (median) | dtype | nmu |\n")
        f.write("|------|---------|-----------------|------------------|-------|-----|\n")
        for r in rows:
            f.write(f"| {r['size']} | {r['repeats']} | {r['time_median']:.4f} | {r['dems_per_s_median']:,.0f} | {r['dtype']} | {r['nmu']} |\n")

    return rows


# ----------------------------
# cProfile (call graph)
# ----------------------------
def run_cprofile(
    STACK: np.ndarray,
    T_RESP: np.ndarray,
    T_RESP_LOGT: np.ndarray,
    TEMPS: np.ndarray,
    *,
    sz: int = 256,
    nmu: int = 42,
    dtype: np.dtype = np.float32,
    outdir: Path | str = "benchmark_out",
    solver_fn: Callable = dn2dem_pos,
) -> Tuple[str, str]:
    """
    Run cProfile on a representative crop; write .prof + text summary.
    Returns (prof_binary_path, text_summary_path).
    """
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    sz = int(min(sz, STACK.shape[2], STACK.shape[3]))
    frame = STACK[0, :, :sz, :sz]

    import cProfile, pstats
    pr = cProfile.Profile()
    pr.enable()
    _ = run_dn2dem(frame, T_RESP, T_RESP_LOGT, TEMPS, nmu=nmu, dtype=dtype, solver_fn=solver_fn)
    pr.disable()

    prof_path = outdir / f"profile_dn2dem_pos_{sz}x{sz}.prof"
    pr.dump_stats(str(prof_path))

    txt_path = outdir / f"profile_dn2dem_pos_{sz}x{sz}.txt"
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumtime").print_stats(40)
    txt_path.write_text(s.getvalue(), encoding="utf-8")

    return str(prof_path), str(txt_path)


# ----------------------------
# line_profiler (serial path)
# ----------------------------
def run_line_profiler(
    STACK: np.ndarray,
    T_RESP: np.ndarray,
    T_RESP_LOGT: np.ndarray,
    TEMPS: np.ndarray,
    *,
    sz: int = 14,  # 14x14 -> na=196 (<200) => serial branch in demmap_pos
    nmu: int = 42,
    dtype: np.dtype = np.float32,
    outdir: Path | str = "benchmark_out",
    include_functions: Optional[List[Callable]] = None,
    solver_fn: Callable = dn2dem_pos,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Profile line-by-line at a small crop that forces serial path.
    Writes .lprof (binary) + .txt report. Returns (binary_path, text_path) or (None, None) if unavailable.
    """
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    sz = int(min(sz, STACK.shape[2], STACK.shape[3]))
    frame = STACK[0, :, :sz, :sz]

    try:
        import line_profiler
    except Exception as e:
        (outdir / "line_profiler_skipped.txt").write_text(f"line_profiler not available: {e}\n", encoding="utf-8")
        return None, None

    lp = line_profiler.LineProfiler()
    # default functions to profile
    funcs = include_functions or [dn2dem_pos, _demmap_pos, _dem_pix]
    for fn in funcs:
        lp.add_function(fn)

    def _target():
        _ = run_dn2dem(frame, T_RESP, T_RESP_LOGT, TEMPS, nmu=nmu, dtype=dtype, solver_fn=solver_fn)

    lp_wrapper = lp(_target)
    lp_wrapper()

    lprof_path = outdir / f"lineprofile_{sz}x{sz}.lprof"
    txt_path = outdir / f"lineprofile_{sz}x{sz}.txt"
    lp.dump_stats(str(lprof_path))
    with txt_path.open("w", encoding="utf-8") as f:
        lp.print_stats(stream=f)

    return str(lprof_path), str(txt_path)


# ----------------------------
# Env snapshot (reproducibility)
# ----------------------------
def write_env_snapshot(
    STACK: np.ndarray,
    T_RESP: np.ndarray,
    T_RESP_LOGT: np.ndarray,
    TEMPS: np.ndarray,
    *,
    outdir: Path | str = "benchmark_out",
    extra: Optional[dict] = None,
) -> str:
    """Write a JSON with environment + shapes for reproducibility. Returns path."""
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    info = {
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "thread_caps": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS"),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
        },
        "stack_shape": tuple(map(int, STACK.shape)),
        "tresp_shape": tuple(map(int, T_RESP.shape)),
        "tresp_logt_len": int(T_RESP_LOGT.shape[0]),
        "temps_len": int(TEMPS.shape[0]),
    }
    if extra:
        info.update(extra)
    path = outdir / "env.json"
    path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return str(path)


# ----------------------------
# One-call suite
# ----------------------------
def run_baseline_suite(
    STACK: np.ndarray,
    T_RESP: np.ndarray,
    T_RESP_LOGT: np.ndarray,
    TEMPS: np.ndarray,
    *,
    sizes: Iterable[int] = (14, 64, 256, 1024),
    repeats: int = 5,
    nmu: int = 42,
    dtype: np.dtype = np.float32,
    outdir: Path | str = "benchmark_out",
    solver_fn: Callable = dn2dem_pos,
    force_single_thread: bool = False,
) -> dict:
    """
    Runs env snapshot, wall-clock bench, cProfile, and line_profiler in one go.
    Returns dict of output paths/rows.
    """
    if force_single_thread:
        set_single_thread_caps()
    env_path = write_env_snapshot(STACK, T_RESP, T_RESP_LOGT, TEMPS, outdir=outdir)
    rows = benchmark_wallclock(STACK, T_RESP, T_RESP_LOGT, TEMPS, sizes=sizes, repeats=repeats,
                               nmu=nmu, dtype=dtype, outdir=outdir, solver_fn=solver_fn)
    prof_bin, prof_txt = run_cprofile(STACK, T_RESP, T_RESP_LOGT, TEMPS, sz=min(256, STACK.shape[2]),
                                      nmu=nmu, dtype=dtype, outdir=outdir, solver_fn=solver_fn)
    lbin, ltxt = run_line_profiler(STACK, T_RESP, T_RESP_LOGT, TEMPS, sz=min(14, STACK.shape[2]),
                                   nmu=nmu, dtype=dtype, outdir=outdir, solver_fn=solver_fn)
    return {
        "env_json": env_path,
        "wallclock_rows": rows,
        "cprofile_prof": prof_bin,
        "cprofile_txt": prof_txt,
        "lineprof_bin": lbin,
        "lineprof_txt": ltxt,
        "outdir": str(Path(outdir).resolve()),
    }


# ----------------------------
# Lightweight cross-module bench sink
# ----------------------------
_BENCH_OUTDIR = Path(os.environ.get("BENCH_OUTDIR", "benchmark_out"))

def set_bench_outdir(path: str | Path) -> None:
    """Set the output directory for bench_row CSV."""
    global _BENCH_OUTDIR
    _BENCH_OUTDIR = Path(path)
    _BENCH_OUTDIR.mkdir(parents=True, exist_ok=True)

def bench_row(**kw) -> None:
    """
    Minimal CSV logger used by Dask runner (and others).
    Writes to <outdir>/profiling_dask.csv; outdir can be set via set_bench_outdir or BENCH_OUTDIR env.
    """
    _BENCH_OUTDIR.mkdir(parents=True, exist_ok=True)
    path = _BENCH_OUTDIR / "profiling_dask.csv"
    header_needed = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted(kw.keys()))
        if header_needed:
            w.writeheader()
        w.writerow(kw)
