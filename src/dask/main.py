# src/dask/main.py
# Dask driver for DEMREG on AIA bandsets with full profiling suite
from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import re
import time
import warnings
from pathlib import Path
from typing import Tuple

import dask
import dask.array as da
import numpy as np

# Suppress noisy but harmless Dask worker warnings
logging.getLogger('distributed.worker.state_machine').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=UserWarning, module='distributed')

from src.common.dataio import default_files, build_lazy_npz_stack
from src.common.profiling.profiler import Profiler
from src.common.profiling.backend_utils import create_synchronize_fn
from src.common.profiling.checks import basic_checks
from src.common.profiling.io_helpers import bench_row, set_bench_outdir

from .runner import dask_client_single_node
from .suite import build_graph


def parse_size(s: str) -> Tuple[int, int]:
    """Parse size specification (N or NxM) into (H, W) tuple."""
    s = str(s).lower().strip()
    m = re.match(r"^\s*(\d+)\s*x\s*(\d+)\s*$", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    if s.isdigit():
        n = int(s)
        return n, n
    raise ValueError(f"Invalid --sizes value: {s!r} (expected N or NxM)")


def parse_idx(val: str):
    """
    Translate --idx into the selector expected by build_lazy_npz_stack.
    Accepts: "all", "-1" -> None (all frames); integer strings -> int.
    """
    if val is None:
        return None
    v = str(val).strip().lower()
    if v in {"all", "-1"}:
        return None
    try:
        return int(v)
    except Exception:
        raise ValueError(f"--idx must be 'all' or an integer, got: {val!r}")


def now_stamp() -> str:
    """Generate timestamp for file naming."""
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Dask DEMREG on AIA bandsets with full profiling.")

    ap.add_argument("--data-dir", type=str, default="data/np32",
                    help="Directory with NPZ files containing 'bands' (6,H,W).")
    ap.add_argument("--idx", type=str, default="0",
                    help="Frame index or 'all' (default: 0).")
    ap.add_argument("--sizes", type=str, default="4096",
                    help='Crop as N or NxM (default: 4096 = full size).')
    ap.add_argument("--tile", type=int, default=256,
                    help="Tile size for chunking H/W (default: 256).")
    ap.add_argument("--nmu", type=int, default=42,
                    help="Solver parameter (mu-grid size).")
    ap.add_argument("--nt", type=int, default=25,
                    help="Temperature bins for DEM (default: 25).")

    ap.add_argument("--bench-root", type=str, default="benchmarking/dask",
                    help="Directory to write profiling artifacts.")
    ap.add_argument("--save-out", action="store_true",
                    help="If set, save numpy outputs for sanity (small sizes recommended).")
    ap.add_argument("--outdir", type=str, default="data/output/dask",
                    help="Directory to write numpy outputs when --save-out is set.")

    # Dask cluster options
    ap.add_argument("--n-workers", type=int, default=None,
                    help="Override worker count (default: from Slurm allocation).")
    ap.add_argument("--threads-per-worker", type=int, default=1,
                    help="Threads per worker (default: 1).")

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # Setup paths
    data_dir = Path(args.data_dir)
    bench_root = Path(args.bench_root)
    outdir = Path(args.outdir)

    # Parse configuration
    H, W = parse_size(args.sizes)
    tile = int(args.tile)
    nmu = int(args.nmu)
    nt = int(args.nt)
    idx_sel = parse_idx(args.idx)

    # Generate unique timestamp for this run
    stamp = now_stamp()

    # Configure benchmark output directory
    set_bench_outdir(bench_root)

    # Build lazy stack using your data I/O: (F, Hc, Wc, 6) with chunks (1, Th, Tw, 6)
    files = default_files(data_dir, ext="*.npz")
    if not files:
        raise FileNotFoundError(f"No NPZ files found in {data_dir}")

    print(f"[INFO] Found {len(files)} NPZ files in {data_dir}")

    stack_da: da.Array = build_lazy_npz_stack(
        files,
        idx=idx_sel,
        crop_hw=(H, W),
        tile_hw=(tile, tile),
    )

    # Take first frame for this run
    if stack_da.shape[0] < 1:
        raise ValueError("No frames selected from dataset.")
    frame_da = stack_da[0]  # (H, W, 6) with chunks (Th, Tw, 6)

    print(f"[INFO] Frame shape: {frame_da.shape}")
    print(f"[INFO] Frame chunks: {frame_da.chunks}")
    print(f"[INFO] Number of tiles: {frame_da.npartitions}")

    # Launch Dask cluster and run with full profiling
    with dask_client_single_node(
            n_workers=args.n_workers,
            threads_per_worker=args.threads_per_worker,
    ) as client:

        print(f"[INFO] Dask cluster ready")
        print(f"[INFO] Scheduler: {client.scheduler.address}")
        print(f"[INFO] Workers: {len(client.scheduler_info()['workers'])}")

        # Initialize profiler with all features
        with Profiler(
                client=client,
                benchdir=bench_root,
                stamp=stamp,
                enable_perf_html=True,
                enable_task_stream=True,
                enable_worker_snapshots=True,
                enable_system_sampler=True,
                enable_gpu_sampler=False,  # Set to True if using GPUs
        ) as prof:
            # Capture worker state before computation
            prof.snapshot_workers("before")

            # Build computation graph
            prof.section("build_graph", start=True)
            dem_lazy, edem_lazy, chisq_lazy = build_graph(
                frame_da,
                nmu=nmu,
                nt=nt,
            )
            prof.section("build_graph", start=False)

            print(f"[INFO] Graph built: dem{dem_lazy.shape}, tiles={dem_lazy.npartitions}")

            # Run computation with profiling context
            prof.section("compute", start=True)

            with prof.compute_context():
                # Compute summary statistics to force full computation
                # This is more efficient than materializing full arrays
                dem_sum, edem_sum, chisq_sum, chisq_mean = client.compute([
                    dem_lazy.sum(),
                    edem_lazy.sum(),
                    chisq_lazy.sum(),
                    chisq_lazy.mean()
                ], sync=True)

            prof.section("compute", start=False)

            # Capture worker state after computation
            prof.snapshot_workers("after")

            # Get timing
            compute_time = prof._sections.get("compute", (0, 0))[1]

            print(f"[INFO] Computation complete in {compute_time:.2f}s")
            print(f"[INFO] Chi-sq mean: {chisq_mean:.3f}")

    # Compute throughput metrics
    mpixels = (H * W) / 1e6
    mpps = mpixels / compute_time if compute_time > 0 else 0.0

    # Prepare bench row for CSV logging
    bench_data = {
        "timestamp": stamp,
        "impl": "dask",
        "frame_size_H": H,
        "frame_size_W": W,
        "tile": tile,
        "nmu": nmu,
        "nt": nt,
        "n_workers": args.n_workers or "auto",
        "threads_per_worker": args.threads_per_worker,
        "compute_seconds": round(compute_time, 4),
        "mpixels_per_sec": round(mpps, 3),
        "chisq_mean": round(float(chisq_mean), 4),
        "job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "node": os.environ.get("SLURMD_NODENAME", os.uname().nodename),
    }

    # Write to bench.csv
    bench_row(**bench_data)

    print(f"\n{'=' * 60}")
    print(f"[RESULTS] Dask DEMREG Benchmark")
    print(f"{'=' * 60}")
    print(f"  Size:       {H}x{W}")
    print(f"  Tile:       {tile}x{tile}")
    print(f"  Workers:    {len(client.scheduler_info()['workers'])}")
    print(f"  Time:       {compute_time:.3f}s")
    print(f"  Throughput: {mpps:.2f} MPix/s")
    print(f"  Chi-sq:     {chisq_mean:.3f}")
    print(f"{'=' * 60}")
    print(f"[OUTPUT] Profiling artifacts written to: {bench_root}")
    print(f"  - bench.csv (aggregate results)")
    print(f"  - performance_report_{stamp}.html (Dask dashboard)")
    print(f"  - tasks_{stamp}.csv (raw task stream)")
    print(f"  - tasks_agg_{stamp}.csv (aggregated task stats)")
    print(f"  - system_timeseries_{stamp}.csv (CPU/memory over time)")
    print(f"  - run_report_{stamp}.json (full timing breakdown)")
    print(f"  - env_{stamp}.json (environment snapshot)")
    print(f"  - workers_before_{stamp}.csv (worker metrics before)")
    print(f"  - workers_after_{stamp}.csv (worker metrics after)")
    print(f"{'=' * 60}\n")

    # Optional: save outputs for verification
    if args.save_out:
        outdir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Computing and saving outputs to {outdir}...")

        # Compute full arrays (expensive for large sizes!)
        dem_np, edem_np, chisq_np = dask.compute(dem_lazy, edem_lazy, chisq_lazy)

        # Save outputs
        np.save(outdir / f"dem_{stamp}.npy", np.asarray(dem_np))
        np.save(outdir / f"edem_{stamp}.npy", np.asarray(edem_np))
        np.save(outdir / f"chisq_{stamp}.npy", np.asarray(chisq_np))

        # Run basic quality checks
        checks = basic_checks(dem_np, chisq_np)
        print(f"[CHECKS] Quality indicators:")
        print(f"  - Finite fraction:   {checks['finite_frac']:.4f}")
        print(f"  - Positive fraction: {checks['positive_frac']:.4f}")
        print(f"  - Chi-sq median:     {checks['chisq_median']:.4f}")

        print(f"[SAVE] Outputs saved to {outdir}")


if __name__ == "__main__":
    main()