# src/dask/cli.py
from __future__ import annotations
"""
Dask CLI: parse arguments for the distributed DEM runner.

This CLI stays intentionally compact: the heavy lifting (cluster creation,
tiling, verify hooks, etc.) is done in the runner. Here we only gather
and normalize user inputs.
"""

import argparse
from typing import List, Optional, Tuple

from .cli_slurm import add_slurm_arguments


def _parse_hw(spec: Optional[str]) -> Optional[Tuple[int, int]]:
    """
    Parse a height/width spec.

    Accepts:
      - "256"          -> (256, 256)
      - "256,512"      -> (256, 512)
      - None or ""     -> None

    Returns
    -------
    (H, W) as ints, or None if no spec provided.
    """
    if not spec:
        return None
    parts = [p.strip() for p in str(spec).split(",") if p.strip()]
    if len(parts) == 1:
        n = int(parts[0])
        return (n, n)
    return (int(parts[0]), int(parts[1]))


def get_parser() -> argparse.ArgumentParser:
    """
    Build the argument parser for the Dask runner.
    """
    p = argparse.ArgumentParser(
        prog="dask DEM runner",
        description="Tile frames, distribute work with Dask, and optionally verify against goldens.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -------- Core workload flags (kept tight) --------
    g = p.add_argument_group("Workload")
    g.add_argument(
        "--data-dir",
        default="data/np32",
        help="Directory containing input frames (npy/npz).",
    )
    g.add_argument(
        "--idx",
        default="all",
        help="Which frame(s) to run: 'all' or a single integer index (0-based).",
    )
    g.add_argument(
        "--sizes", default=None, help="Crop size H[,W]; omit to use native size."
    )
    g.add_argument(
        "--tile", default="256", help="Tile size Th[,Tw]; last tiles may be partial."
    )
    g.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="Example per-tile parameter forwarded to the solver.",
    )
    g.add_argument(
        "--nmu",
        type=int,
        default=None,
        help="Example per-tile parameter forwarded to the solver.",
    )

    # -------- Verification flags --------
    v = p.add_argument_group("Verification")
    mx = v.add_mutually_exclusive_group()
    mx.add_argument(
        "--verify",
        dest="verify",
        action="store_true",
        help="Enable verification against goldens (if helpers exist).",
    )
    mx.add_argument(
        "--no-verify", dest="verify", action="store_false", help="Disable verification."
    )
    p.set_defaults(verify=False)
    v.add_argument(
        "--golden-root", default=None, help="Path to golden data (optional)."
    )
    v.add_argument(
        "--chisq-mode",
        choices=("exact", "auto", "skip"),
        default="auto",
        help="Compatibility knob; pass through if used.",
    )

    # -------- Cluster mode (simple) --------
    c = p.add_argument_group("Cluster")
    c.add_argument(
        "--cluster-mode",
        choices=("local", "slurm"),
        default="local",
        help="Run locally or via SLURM (dask-jobqueue).",
    )
    # Local options (few; runner provides smart defaults)
    c.add_argument("--n-workers", type=int, default=None, help="Number of workers.")
    c.add_argument(
        "--threads-per-worker", type=int, default=1, help="Threads per worker."
    )
    c.add_argument(
        "--processes",
        dest="processes",
        action="store_true",
        help="Use processes for workers.",
    )
    c.add_argument(
        "--no-processes",
        dest="processes",
        action="store_false",
        help="Use threads for workers.",
    )
    p.set_defaults(processes=True)

    # Optional: address/port for attaching to an existing scheduler
    c.add_argument(
        "--scheduler-address", default=None, help="Existing scheduler address."
    )
    c.add_argument("--scheduler-port", default=None, help="Existing scheduler port.")

    # -------- Task entry --------
    t = p.add_argument_group("Task")
    t.add_argument(
        "--task", default=None, help="module:function to call after cluster creation."
    )

    add_slurm_arguments(p)
    return p


def parse_args(argv: Optional[List[str]] = None):
    """
    Parse CLI arguments and normalize composite values.

    - `sizes` and `tile` are normalized to `(H, W)` / `(Th, Tw)` tuples.
    - `scheduler_address` / `scheduler_port` are cleaned if set to empty/`None` strings.
    """
    args = get_parser().parse_args(argv)

    # Normalize size/tile strings to tuples
    # args.sizes = _parse_hw(args.sizes)
    # args.tile = _parse_hw(args.tile)

    # Clean up address/port when provided as empty or literal "None"
    if getattr(args, "scheduler_address", None) in ("", "None"):
        args.scheduler_address = None
    if getattr(args, "scheduler_port", None) in ("", "None"):
        args.scheduler_port = None

    return args
