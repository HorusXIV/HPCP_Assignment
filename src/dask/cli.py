from __future__ import annotations
import argparse

def parse_args():
    ap = argparse.ArgumentParser("dask DEM runner")
    # dataset / sizing
    ap.add_argument("--use-synthetic", action="store_true", default=False)
    ap.add_argument("--data-dir", type=str, default=None)
    ap.add_argument("--ext", type=str, default="*.npz")
    ap.add_argument("--idx", type=str, default="-1")
    ap.add_argument("--sizes", type=str, default="512,1024")
    ap.add_argument("--tile", type=str, default="128,128", help="tile_h,tile_w (default 128,128)")
    ap.add_argument("--repeats", type=int, default=1, help="How many times to repeat benchmark run")
    ap.add_argument("--nmu", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="dask/benchmark_out")

    # cluster / scheduler
    ap.add_argument("--scheduler", type=str, default=None,
                    help="Scheduler address to connect to (e.g. tcp://host:8786). If absent, start local cluster.")
    ap.add_argument("--n-workers", type=int, default=None,
                    help="Default: number of physical cores (threads if --processes is used)")
    ap.add_argument("--threads-per-worker", type=int, default=1,
                    help="Threads per worker (default 1).")
    ap.add_argument("--processes", action="store_true", default=True,
                    help="Use processes (better on Linux clusters). Default True.")
    ap.add_argument("--no-processes", dest="processes", action="store_false")
    ap.add_argument("--memory-limit", type=str, default=None,
                    help="Per-worker memory limit (default: auto from system).")

    # threading caps (per worker) + driver
    ap.add_argument("--worker-blas-threads", type=int, default=1,
                    help="Per-worker BLAS/OpenMP cap.")
    ap.add_argument("--single-thread", action="store_true", default=False,
                    help="Driver BLAS/OpenMP=1")
    ap.add_argument("--blas-threads", type=int, default=None,
                    help="Driver BLAS/OpenMP cap (overrides env).")
    return ap.parse_args()
