# src/baseline/cli.py
from __future__ import annotations

"""
Command-line interface for the baseline DEM runner.

This module defines a single helper, `parse_args()`, which builds and parses
all CLI options used by the baseline pipeline. Options are grouped into
dataset/sizing, execution/environment, outputs, and verification.
"""

import argparse


def parse_args() -> argparse.Namespace:
    """
    Build and parse CLI arguments for the baseline DEM runner.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with attributes corresponding to the defined options.

    Notes
    -----
    Key options:
      • --sizes: accepts a single size like "512" (interpreted as 512x512),
        an HxW form like "512x256", or multiple sizes via commas/spaces
        (e.g., "14,64,256,1024"), depending on your main entrypoint’s parser.
      • --idx: frame/index selector. Use an integer (0-based), "all", or "-1"
        to process *all* available stacks.
      • --device: "cpu" or a CUDA device id string like "0".
      • Verification is enabled by default; use --no-verify to disable.
    """
    ap = argparse.ArgumentParser(
        prog="hpcp-baseline",
        description="Baseline DEM benchmark runner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---------------- Dataset / sizing ----------------
    ap.add_argument(
        "--use-synthetic",
        action="store_true",
        default=False,
        help="Use synthetic inputs instead of loading NPZ stacks from disk.",
    )
    ap.add_argument(
        "--sizes",
        type=str,
        default="14,64,256,1024",  # ,2048,4096
        help=(
            "Size(s) to process. Accepts a single int 'N' (treated as NxN), "
            "an 'HxW' form (e.g., '512x256'), or multiple tokens separated by "
            "commas/spaces (e.g., '14,64,256,1024')."
        ),
    )
    ap.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of timing repeats per size for benchmarking.",
    )
    ap.add_argument(
        "--nmu",
        type=int,
        default=42,
        help="Regularization/temperature resolution knob passed to the solver.",
    )
    ap.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Root folder containing NPZ stacks (when not using --use-synthetic).",
    )
    ap.add_argument(
        "--ext",
        type=str,
        default="*.npz",
        help="Glob for stack files under --data-dir.",
    )
    ap.add_argument(
        "--idx",
        type=str,
        default="-1",
        help=(
            "Frame index selector. Use an integer (0-based), 'all', or '-1' to "
            "process all stacks."
        ),
    )

    # ---------------- Execution / environment ----------------
    ap.add_argument(
        "--device",
        default="cpu",
        help="Execution device: 'cpu' or a CUDA device id string like '0'.",
    )
    ap.add_argument(
        "--single-thread",
        action="store_true",
        default=False,
        help="Force single-threaded execution in Python (helpful for profiling).",
    )
    ap.add_argument(
        "--blas-threads",
        type=int,
        default=None,
        help=(
            "Explicitly set BLAS/OpenMP thread count (overrides env like "
            "OPENBLAS_NUM_THREADS, MKL_NUM_THREADS, OMP_NUM_THREADS)."
        ),
    )
    ap.add_argument(
        "--no-runtime-enforce",
        action="store_true",
        default=False,
        help="Skip runtime enforcement of threading env vars.",
    )
    ap.add_argument(
        "--nvtx",
        type=str,
        default=None,
        help="Optional NVTX range label to annotate the run (if NVTX is available).",
    )

    # ---------------- Outputs ----------------
    ap.add_argument(
        "--outdir",
        type=str,
        default="baseline/benchmark_out",
        help="Destination folder for benchmark artifacts (CSV, markdown, etc.).",
    )

    # ---------------- Verification (enabled by default) ----------------
    ver = ap.add_argument_group("verification")
    ver.add_argument(
        "--verify",
        dest="verify",
        action="store_true",
        default=True,
        help="Verify outputs against golden references.",
    )
    ver.add_argument(
        "--no-verify",
        dest="verify",
        action="store_false",
        help="Disable verification against golden references.",
    )
    ver.add_argument(
        "--verify-sizes",
        type=str,
        default=None,
        help="Comma-separated sizes to verify (defaults to the set from --sizes).",
    )
    ver.add_argument(
        "--golden-root",
        type=str,
        default="data/golden",
        help="Root directory with size subfolders containing baseline.npz/json.",
    )
    ver.add_argument(
        "--chisq-mode",
        type=str,
        choices=("exact", "auto", "skip"),
        default="exact",
        help="How to compare χ² maps during verification.",
    )

    return ap.parse_args()
