# src/baseline/main.py
from __future__ import annotations

"""
Entry point for the baseline DEM runner.

This module wires command-line arguments to the baseline benchmark runner.
It intentionally keeps parsing simple:
  - --sizes accepts a single integer (square) or an "HxW" pair.
  - --tile  accepts a single integer (square) or a "ThxTw" pair.

For multi-size orchestration or more flexible parsing, use higher-level
scripts or the Dask entrypoint.
"""

import argparse
from typing import Optional, Tuple

from .runner import run_benchmark


def _parse_sizes(s: Optional[str]) -> Optional[Tuple[int, int]]:
    """
    Parse a size specifier into (H, W).

    Parameters
    ----------
    s : str or None
        Either an integer string (e.g., "512") interpreted as (512, 512),
        or an "HxW" form (case-insensitive, e.g., "512x256"). If None or empty,
        returns None so the runner can decide a default.

    Returns
    -------
    (int, int) or None
        Parsed (H, W) tuple, or None if not provided.

    Raises
    ------
    ValueError
        If the string is malformed.
    """
    if not s:
        return None
    s_l = s.lower()
    if "x" in s_l:
        a, b = s_l.split("x", 1)
        return int(a), int(b)
    v = int(s)
    return (v, v)


def _parse_tile(s: Optional[str], default: Tuple[int, int] = (256, 256)) -> Tuple[int, int]:
    """
    Parse a tile specifier into (Th, Tw), defaulting to `default` when missing.

    Parameters
    ----------
    s : str or None
        Either an integer string (e.g., "256") interpreted as (256, 256),
        or a "ThxTw" form (e.g., "256x128"). If None or empty, returns `default`.
    default : (int, int)
        Fallback tile size.

    Returns
    -------
    (int, int)
        Parsed (Th, Tw) tile size.

    Raises
    ------
    ValueError
        If the string is malformed.
    """
    if not s:
        return default
    s_l = s.lower()
    if "x" in s_l:
        a, b = s_l.split("x", 1)
        return int(a), int(b)
    v = int(s)
    return (v, v)


def main() -> None:
    """
    Parse CLI arguments and run the baseline benchmark.

    Notes
    -----
    - `--idx` accepts an integer index, "all", or "-1". Project-wide convention:
      "-1" and "all" both mean "process all stacks".
    - `--outdir` is a legacy alias for `--bench-root` and may be removed later.
    """
    p = argparse.ArgumentParser(
        prog="hpcp-baseline",
        description="Baseline DEM benchmark runner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data selection
    p.add_argument("--data-dir", type=str, default=None, help="Root directory of NPZ stacks.")
    p.add_argument("--ext", type=str, default="*.npz", help="Glob pattern for stacks under --data-dir.")
    p.add_argument(
        "--idx",
        type=str,
        default="-1",
        help="Frame selector: integer (0-based), 'all', or '-1' for all stacks.",
    )

    # Problem sizing
    p.add_argument(
        "--sizes",
        type=str,
        default=None,
        help="Output size: integer 'N' (NxN) or 'HxW' (e.g., '1024x512').",
    )
    p.add_argument(
        "--tile",
        type=str,
        default="256",
        help="Tile size: integer 'T' (TxT) or 'ThxTw' (e.g., '256x128').",
    )

    # Solver/benchmark
    p.add_argument("--nmu", type=int, default=42, help="Regularization / temperature resolution knob.")
    p.add_argument("--repeats", type=int, default=1, help="Number of timing repeats per size.")
    p.add_argument("--verify", action="store_true", help="Verify results against goldens if configured.")
    p.add_argument("--golden-root", type=str, default=None, help="Root with golden references for verification.")

    # Outputs
    p.add_argument("--outdir", type=str, default=None, help="Legacy alias for --bench-root.")
    p.add_argument(
        "--bench-root",
        type=str,
        default=None,
        help="Benchmark output root (defaults to ~/benchmarking/baseline if unset).",
    )

    args = p.parse_args()

    sizes = _parse_sizes(args.sizes)
    tile = _parse_tile(args.tile)

    summary = run_benchmark(
        use_synthetic=False,
        data_dir=args.data_dir,
        ext=args.ext,
        idx=args.idx,
        sizes=sizes,
        tile=tile,
        nmu=args.nmu,
        repeats=args.repeats,
        verify=args.verify,
        golden_root=args.golden_root,
        outdir=args.outdir,
        bench_root=args.bench_root,
    )
    # Print a short summary path to help users find outputs.
    print(f"Benchmark artifacts in: {summary['bench_root']}")


if __name__ == "__main__":
    main()
