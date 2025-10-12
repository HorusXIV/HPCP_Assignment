# src/baseline/run.py
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np

from src.baseline.solver import solve_dem  # core inversion
from src.common.solver_utils import get_bins, synthesize_tresp  # single source of truth for bins + synthetic resp
from src.common.dataio.save import (  # standardized NPZ saving
    make_run_dir,
    save_npz_bundle,
    save_meta,
    default_tag,
)


# ----------------------------
# Data loading
# ----------------------------

def _normalize_stack(raw: np.ndarray) -> np.ndarray:
    """
    Normalize an image cube to (F, 6, H, W), dtype float32.
    Accepts:
      - (F, 6, H, W)
      - (F, H, W, 6)
      - (6, H, W)
      - (H, W, 6)
    """
    if raw.ndim == 4 and raw.shape[1] == 6:
        stack = raw
    elif raw.ndim == 4 and raw.shape[-1] == 6:
        stack = np.transpose(raw, (0, 3, 1, 2))
    elif raw.ndim == 3 and raw.shape[0] == 6:
        stack = raw[None, ...]
    elif raw.ndim == 3 and raw.shape[-1] == 6:
        stack = np.transpose(raw, (2, 0, 1))[None, ...]
    else:
        raise ValueError(f"Unsupported stack shape {raw.shape}; expected (F,6,H,W) or (F,H,W,6) or (6,H,W)")

    return stack.astype(np.float32, copy=False)


def load_test_data(path: str | Path | None,
                   calib: str | Path | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load imaging data and calibration for the baseline solver.

    Returns:
      STACK       : (F, 6, H, W) float32
      T_RESP     : (nt, 6) float32           temperature response matrix
      T_RESP_LOGT: (nt,) float32             log10(T) centers
      TEMPS      : (nt+1,) float32           temperature bin edges in Kelvin

    Behavior:
      - Image stack ("stack", "bands", "data", "cube") is required.
      - If 'tresp', 'tresp_logt' and/or 'temps' are missing, we derive:
          * bins (centers + edges) from solver_utils.get_bins()
          * responses from solver_utils.synthesize_tresp() (if missing)
      - If 'calib' is provided, it may be an NPZ with keys or a directory
        containing tresp.npy, tresp_logt.npy, temps.npy.
    """
    if path is None:
        raise ValueError("No data path provided. Pass an NPZ file that contains the image stack.")

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {p}")

    # --- load the image stack ---
    with np.load(p, allow_pickle=False) as z:
        stack_key = next((k for k in ("stack", "bands", "data", "cube") if k in z), None)
        if stack_key is None:
            raise KeyError(f"No stack-like key in {p.name}; tried stack/bands/data/cube. Found: {list(z.files)}")
        STACK = _normalize_stack(z[stack_key])

        # try to get calibration directly
        tresp = z.get("tresp", None)
        tresp_logt = z.get("tresp_logt", z.get("logt", None))
        temps = z.get("temps", None)

    # --- optional calibration source ---
    if calib and (tresp is None or tresp_logt is None or temps is None):
        cpath = Path(calib)
        if cpath.is_dir():
            tnp, lnp, ten = cpath / "tresp.npy", cpath / "tresp_logt.npy", cpath / "temps.npy"
            if tresp is None and tnp.exists(): tresp = np.load(tnp)
            if tresp_logt is None and lnp.exists(): tresp_logt = np.load(lnp)
            if temps is None and ten.exists(): temps = np.load(ten)
        elif cpath.exists():
            with np.load(cpath, allow_pickle=False) as zc:
                if tresp is None: tresp = zc.get("tresp")
                if tresp_logt is None: tresp_logt = zc.get("tresp_logt", zc.get("logt"))
                if temps is None: temps = zc.get("temps")

    # --- bins: always prefer solver_utils if any part missing ---
    if tresp_logt is not None and temps is not None:
        logt_centers = np.asarray(tresp_logt, dtype=np.float32)
        temps_edges = np.asarray(temps, dtype=np.float32)
    else:
        # single source of truth for DEM grid
        logt_centers, temps_edges = get_bins()  # defaults: nt=50, 5.5..7.5

    # --- response matrix: synthesize if missing ---
    nf = STACK.shape[1]  # channels, expect 6
    if tresp is None:
        #tresp = synthesize_tresp(logt_centers, nf=nf)
        tresp = synthesize_tresp(
            logt_centers,
            nf=nf,
            include_jacobian=True,  # redundant if default set as above
            normalize="l1",
            width=0.20,
        )

    # finalize dtypes + sanity
    T_RESP = np.asarray(tresp, dtype=np.float32)
    T_RESP_LOGT = np.asarray(logt_centers, dtype=np.float32)
    TEMPS = np.asarray(temps_edges, dtype=np.float32)

    if T_RESP.ndim != 2 or T_RESP.shape[1] != nf:
        raise ValueError(f"tresp must be (nt,{nf}), got {T_RESP.shape}")
    if T_RESP_LOGT.ndim != 1 or T_RESP_LOGT.shape[0] != T_RESP.shape[0]:
        raise ValueError(f"tresp_logt must be (nt,), got {T_RESP_LOGT.shape} vs nt={T_RESP.shape[0]}")
    if TEMPS.ndim != 1 or TEMPS.shape[0] != T_RESP.shape[0] + 1:
        raise ValueError(f"temps must be edges (nt+1,), got {TEMPS.shape} for nt={T_RESP.shape[0]}")

    return STACK, T_RESP, T_RESP_LOGT, TEMPS


# ----------------------------
# Core runs
# ----------------------------

def crop_center(frame_hw6: np.ndarray, size: int) -> np.ndarray:
    """
    Center-crop a (H, W, 6) frame to (size, size, 6).
    """
    if frame_hw6.ndim != 3 or frame_hw6.shape[-1] != 6:
        raise ValueError(f"Expected frame (H,W,6), got {frame_hw6.shape}")
    H, W, _ = frame_hw6.shape

    if size > H or size > W:
        raise ValueError(f"Requested crop {size} exceeds frame {H}x{W}")
    y0 = (H - size) // 2
    x0 = (W - size) // 2
    return frame_hw6[y0:y0 + size, x0:x0 + size, :]


def run_baseline_solve(frame_6hw: np.ndarray,
                       tresp: np.ndarray,
                       tresp_logt: np.ndarray,
                       temps: np.ndarray,
                       *,
                       validate: bool = True,
                       nmu: int = 42) -> Dict[str, np.ndarray | float | Dict[str, float]]:
    """
    Solve DEM for a single (6,H,W) frame.

    Returns dict with arrays and elapsed_seconds and simple checks.
    """
    if frame_6hw.ndim != 3 or frame_6hw.shape[0] != 6:
        raise ValueError(f"frame_6hw must be (6,H,W), got {frame_6hw.shape}")
    print("DN  min/max:", float(frame_6hw.min()), float(frame_6hw.max()))
    print("R  col sums:", np.sum(tresp, axis=0))  # ~1 for each of 6 columns
    print("R  max     :", float(tresp.max()))
    t0 = time.perf_counter()
    try:
        demmap, edemmap, logt, chisq, dn_reg = solve_dem(
            data_6hw=frame_6hw,
            tresp=tresp,
            tresp_logt=tresp_logt,
            temps=temps,
            nmu=nmu,
            validate=validate,  # optional kwarg
        )
    except TypeError:
        demmap, edemmap, logt, chisq, dn_reg = solve_dem(
            data_6hw=frame_6hw,
            tresp=tresp,
            tresp_logt=tresp_logt,
            temps=temps,
            nmu=nmu,
        )
    elapsed = time.perf_counter() - t0

    # quick sanity metrics
    em_sum = float(np.nanmean(np.sum(demmap, axis=-1)))  # mean EM across image
    chisq_mean = float(np.nanmean(chisq))

    return dict(
        demmap=demmap.astype(np.float32, copy=False),
        edemmap=edemmap.astype(np.float32, copy=False),
        logt=logt.astype(np.float32, copy=False),        # 1-D (nt,)
        chisq=chisq.astype(np.float32, copy=False),
        dn_reg=dn_reg.astype(np.float32, copy=False),
        elapsed_seconds=elapsed,
        checks=dict(em_mean=em_sum, chisq_mean=chisq_mean),
    )


def save_single_result(result: Dict[str, np.ndarray | float | Dict[str, float]],
                       *,
                       approach: str = "baseline",
                       extra_tag: Optional[str] = None) -> Path:
    """
    Save a single run's outputs (arrays + meta) to:
      data/output/{approach}/{timestamp}_{tag}/
    Returns the run directory Path.
    """
    H, W, NT = result["demmap"].shape
    tag = default_tag(extra=[extra_tag, f"{H}x{W}", "single"])
    run_dir = make_run_dir(base="data/output", approach=approach, tag=tag)

    save_npz_bundle(
        run_dir,
        demmap=result["demmap"],
        edemmap=result["edemmap"],
        logt=result["logt"],
        chisq=result["chisq"],
        dn_reg=result["dn_reg"],
    )
    save_meta(run_dir, {
        "approach": approach,
        "mode": "single",
        "elapsed_seconds": float(result["elapsed_seconds"]),
        "demmap_shape": tuple(result["demmap"].shape),
        "edemmap_shape": tuple(result["edemmap"].shape),
        "chisq_shape": tuple(result["chisq"].shape),
        "dn_reg_shape": tuple(result["dn_reg"].shape),
        "logt_len": int(result["logt"].shape[0]),
    })
    return run_dir


# ----------------------------
# Benchmarking utilities
# ----------------------------

def _time_solve_on_size(frame_6hw: np.ndarray,
                        tresp: np.ndarray,
                        tresp_logt: np.ndarray,
                        temps: np.ndarray,
                        size: int,
                        nmu: int,
                        validate: bool,
                        *,
                        return_result: bool = False) -> tuple[float, dict | None]:
    """
    Center-crop to `size`, run one solve, and return (elapsed_seconds, result_or_None).
    If return_result=True, the 'result' is the same dict as run_baseline_solve(...).
    """
    # (6,H,W) -> (H,W,6) for crop
    hw6 = np.moveaxis(frame_6hw, 0, -1)
    hw6_crop = crop_center(hw6, size)
    crop_6hw = np.moveaxis(hw6_crop, -1, 0)

    if return_result:
        # Use the high-level runner to get arrays + elapsed
        res = run_baseline_solve(crop_6hw, tresp, tresp_logt, temps,
                                 validate=validate, nmu=nmu)
        return float(res["elapsed_seconds"]), res

    # Fast path if we only need timing
    t0 = time.perf_counter()
    try:
        _ = solve_dem(data_6hw=crop_6hw, tresp=tresp, tresp_logt=tresp_logt,
                      temps=temps, nmu=nmu, validate=validate)
    except TypeError:
        _ = solve_dem(data_6hw=crop_6hw, tresp=tresp, tresp_logt=tresp_logt,
                      temps=temps, nmu=nmu)
    return time.perf_counter() - t0, None


def run_benchmark(stack_f6hw: np.ndarray,
                  tresp: np.ndarray,
                  tresp_logt: np.ndarray,
                  temps: np.ndarray,
                  *,
                  sizes: list[int],
                  repeats: int = 3,
                  nmu: int = 42,
                  validate: bool = False,
                  benchdir: Path | str = Path("benchmark_out/baseline"),
                  save_outputs: str | bool = False,   # NEW: 'none' | 'first' | 'all' | True
                  approach: str = "baseline") -> Path:
    """
    Wallclock benchmark over sizes×repeats using frame 0.

    save_outputs:
      - False/'none' : don't save arrays (default)
      - True/'first' : save NPZ for the first repeat of each size
      - 'all'        : save NPZ for every repeat
    """
    benchdir = Path(benchdir)
    outdir = benchdir / "wallclock"
    outdir.mkdir(parents=True, exist_ok=True)

    frame = stack_f6hw[0]  # (6,H,W)

    # normalize save policy
    policy = "none"
    if save_outputs is True:
        policy = "first"
    elif isinstance(save_outputs, str):
        policy = save_outputs.lower().strip()

    rows = []
    for sz in sizes:
        for r in range(1, repeats + 1):
            need_result = (policy == "all") or (policy == "first" and r == 1)
            elapsed, res = _time_solve_on_size(frame, tresp, tresp_logt, temps,
                                               sz, nmu, validate,
                                               return_result=need_result)
            rows.append((sz, r, elapsed))
            print(f"[bench] size={sz:4d} repeat={r}/{repeats}  time={elapsed:7.3f}s")

            if res is not None:
                tag = f"bench_size{sz}_rep{r}"
                run_dir = save_single_result(res, approach=approach, extra_tag=tag)
                print(f"[bench] saved arrays -> {run_dir/'results.npz'}")

    # write CSV/MD (unchanged)
    csv_path = outdir / "wallclock.csv"
    md_path  = outdir / "wallclock.md"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("size,repeat,seconds\n")
        for sz, rep, sec in rows: f.write(f"{sz},{rep},{sec:.6f}\n")

    agg = {}
    for sz in sizes:
        secs = [sec for s, _, sec in rows if s == sz]
        agg[sz] = (float(np.mean(secs)), float(np.std(secs)))

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Baseline wallclock benchmark\n\n")
        f.write("| size | repeats | mean [s] | std [s] |\n|---:|---:|---:|---:|\n")
        for sz in sorted(agg):
            mean, std = agg[sz]
            f.write(f"| {sz} | {repeats} | {mean:.3f} | {std:.3f} |\n")

    print(f"\n[bench] wrote: {csv_path}")
    print(f"[bench] wrote: {md_path}")
    return outdir



# ----------------------------
# Optional convenience main (kept minimal)
# ----------------------------

def main() -> None:
    """
    Minimal interactive entrypoint for quick manual runs:
      python -m src.baseline.run --data path/to/file.npz --size 512
    For full CLI options, use src.baseline.main, which owns the CLI and integrates benchmark/saving policies.
    """
    import argparse

    ap = argparse.ArgumentParser(description="Quick baseline runner (dev convenience).")
    ap.add_argument("--data", type=str, required=True, help="NPZ with stack/bands/data/cube")
    ap.add_argument("--calib", type=str, default=None, help="Optional NPZ or folder with tresp/tresp_logt/temps")
    ap.add_argument("--size", type=int, default=512, help="Center crop size")
    ap.add_argument("--nmu", type=int, default=42)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--save", action="store_true", help="Save NPZ outputs to data/output/baseline")
    args = ap.parse_args()

    STACK, T_RESP, T_RESP_LOGT, TEMPS = load_test_data(args.data, calib=args.calib)

    # crop frame 0
    hw6 = np.moveaxis(STACK[0], 0, -1)
    hw6 = crop_center(hw6, args.size)
    frame = np.moveaxis(hw6, -1, 0)

    result = run_baseline_solve(frame, T_RESP, T_RESP_LOGT, TEMPS, validate=args.validate, nmu=args.nmu)

    print(f"[run] elapsed {result['elapsed_seconds']:.3f}s  chisq_mean={result['checks']['chisq_mean']:.3f}")
    if args.save:
        run_dir = save_single_result(result, extra_tag=f"dev_{args.size}")
        print(f"[run] saved {run_dir/'results.npz'}")


if __name__ == "__main__":
    main()
