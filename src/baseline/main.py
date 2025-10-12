# src/baseline/main.py
from __future__ import annotations
"""
CLI entry for the baseline runner that integrates with run.py.

- Keeps CLI handling here (legacy behavior).
- Resolves data (synthetic or from --data-dir/--ext/--idx).
- Decides mode:
    * benchmark if multiple sizes or repeats > 1
    * single otherwise
- Delegates execution to functions in src.baseline.run.
- Saves NPZ for single mode using common dataio helpers.
"""

import argparse
import glob
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

from .run import (
    load_test_data,
    run_baseline_solve,
    run_benchmark,
)
from src.common.dataio import make_run_dir, save_npz_bundle, save_meta, default_tag


def _parse_size_token(tok: str) -> int:
    """Accept '512' or '512x512' and return the square crop size (int)."""
    tok = tok.strip().lower()
    if "x" in tok:
        a, b = tok.split("x", 1)
        a, b = int(a), int(b)
        if a != b:
            # We only support square crops in run_benchmark; pick min to be safe
            return min(a, b)
        return a
    return int(tok)


def _parse_sizes_list(s: Optional[str]) -> List[int]:
    """Parse '--sizes' like '64,256,1024' or '512' into a list[int]."""
    if not s:
        return [512]  # default quick test
    parts = [p for p in s.replace(" ", ",").split(",") if p.strip()]
    return [_parse_size_token(p) for p in parts]


def _resolve_npz_from_dir(data_dir: Optional[str], ext: str, idx: str) -> Optional[Path]:
    """
    From --data-dir/--ext/--idx select a single NPZ file path.
    - idx can be 'all'/'-1' (we take the first match) or an integer.
    - returns None if no directory specified.
    """
    if not data_dir:
        return None
    paths = sorted(Path(data_dir).glob(ext))
    if not paths:
        raise FileNotFoundError(f"No files matching {ext!r} under {data_dir!r}")
    if idx in ("-1", "all"):
        return paths[0]
    try:
        i = int(idx)
    except ValueError:
        raise ValueError(f"--idx must be integer, 'all', or '-1'; got {idx!r}")
    if i < 0 or i >= len(paths):
        raise IndexError(f"--idx {i} out of range [0, {len(paths)-1}] for {data_dir}/{ext}")
    return paths[i]


def main() -> None:
    p = argparse.ArgumentParser(
        prog="hpcp-baseline",
        description="Baseline DEM runner (CLI integrates with run.py).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data selection (directory-based, single file selection)
    p.add_argument("--use-synthetic", action="store_true", default=False,
                   help="Use synthetic inputs (ignore --data-dir).")
    p.add_argument("--data-dir", type=str, default=None,
                   help="Root directory containing NPZ stacks.")
    p.add_argument("--ext", type=str, default="*.npz",
                   help="Glob for stack files under --data-dir.")
    p.add_argument("--idx", type=str, default="0",
                   help="Frame/file selector: integer (0-based), 'all' or '-1' -> first match.")

    # Sizing / repeats (benchmark decision is inferred from these)
    p.add_argument("--sizes", type=str, default="512",
                   help="Size(s): 'N', 'HxW', or comma-separated list.")
    p.add_argument("--repeats", type=int, default=1,
                   help="Repetitions per size (if >1 -> benchmark mode).")

    # Solver options
    p.add_argument("--nmu", type=int, default=42,
                   help="Regularization/temperature resolution knob.")
    p.add_argument("--validate", action="store_true",
                   help="Enable input/output validation.")

    # Output roots
    p.add_argument("--outdir", type=Path, default=Path("benchmark_out/baseline"),
                   help="Benchmark/profiling artifacts root (used in benchmark).")
    p.add_argument("--save-benchmark", choices=["none", "first", "all"], default="none",
                   help="Also save solver outputs during benchmark runs.")
    args = p.parse_args()

    sizes = _parse_sizes_list(args.sizes)
    run_bench = (len(sizes) > 1) or (args.repeats > 1)

    # Load data -> (STACK, T_RESP, T_RESP_LOGT, TEMPS)
    if args.use_synthetic or not args.data_dir:
        STACK, T_RESP, T_RESP_LOGT, TEMPS = load_test_data(None)
        chosen = None
    else:
        chosen = _resolve_npz_from_dir(args.data_dir, args.ext, args.idx)
        STACK, T_RESP, T_RESP_LOGT, TEMPS = load_test_data(chosen)

    print("\nData shapes:")
    print(f"  STACK:        {STACK.shape}")
    print(f"  T_RESP:       {T_RESP.shape}")
    print(f"  T_RESP_LOGT:  {T_RESP_LOGT.shape}")
    print(f"  TEMPS:        {TEMPS.shape}")
    if chosen:
        print(f"  Selected file: {chosen}")

    if run_bench:
        # Ensure outdir exists for benchmark artifacts
        args.outdir.mkdir(parents=True, exist_ok=True)
        print("\n[MODE] Benchmark")
        run_benchmark(
            STACK, T_RESP, T_RESP_LOGT, TEMPS,
            benchdir=args.outdir,
            sizes=sizes,
            repeats=args.repeats,
            nmu=args.nmu,
            validate=args.validate,
            save_outputs=args.save_benchmark,  # NEW
        )
        print(f"\nBenchmark artifacts in: {args.outdir}")
    else:
        # Single solve, save standardized NPZ via dataio helpers
        print("\n[MODE] Single solve")
        frame = STACK[0]  # (6, H, W)
        result = run_baseline_solve(
            frame, T_RESP, T_RESP_LOGT, TEMPS,
            validate=args.validate, nmu=args.nmu,
        )

        print(f"\nSolve complete:")
        print(f"  Elapsed: {result['elapsed_seconds']:.3f} s")
        print(f"  Output shape: {result['demmap'].shape}")
        for k, v in result["checks"].items():
            print(f"  {k}: {v:.4f}")

        # Save under data/output/baseline/{timestamp}_{tag}/
        tag = default_tag(extra=[
            "single",
            f"{result['demmap'].shape[0]}x{result['demmap'].shape[1]}",
            f"idx{args.idx}" if args.idx not in ("-1", "all") else None,
        ])
        run_dir = make_run_dir(base="data/output", approach="baseline", tag=tag)

        save_npz_bundle(
            run_dir,
            demmap=result["demmap"],
            edemmap=result["edemmap"],
            logt=result["logt"],
            chisq=result["chisq"],
            dn_reg=result["dn_reg"],
        )
        save_meta(run_dir, {
            "approach": "baseline",
            "mode": "single",
            "nmu": int(args.nmu),
            "elapsed_seconds": float(result["elapsed_seconds"]),
            "demmap_shape": tuple(result["demmap"].shape),
            "edemmap_shape": tuple(result["edemmap"].shape),
            "chisq_shape": tuple(result["chisq"].shape),
            "dn_reg_shape": tuple(result["dn_reg"].shape),
            "logt_len": int(result["logt"].shape[0]),
            "source_file": str(chosen) if chosen else None,
        })

        print(f"\nResults saved to: {run_dir}/results.npz")
        print(f"Metadata:          {run_dir}/meta.json")


if __name__ == "__main__":
    main()
