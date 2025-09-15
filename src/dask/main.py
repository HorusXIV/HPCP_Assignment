from __future__ import annotations
from pathlib import Path
import json
import os

from .cli import parse_args
from .client import build_client
from .runner import run_dask_suite
from src.common.threads import early_env_caps  # driver-side caps


def _parse_sizes(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x]

def _parse_tile(s: str) -> tuple[int, int]:
    a, b = (int(x) for x in s.split(","))
    return a, b

def _resolve_outdir(cli_outdir: str | None) -> Path:
    """
    Resolve outdir. If user passed a relative path, make it relative to the repo root,
    not the current working directory.
    """
    # repo root = src/dask/main.py -> parents[2]
    repo_root = Path(__file__).resolve().parents[2]
    if not cli_outdir:
        return (repo_root / "benchmark_out").resolve()
    p = Path(cli_outdir)
    return (repo_root / p).resolve() if not p.is_absolute() else p

def _write_env_json(outdir: Path) -> str:
    import sys, platform, numpy as np
    info = {
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "platform": platform.platform(),
        "cwd": str(Path().resolve()),
        "repo_root": str(Path(__file__).resolve().parents[2]),
        "dask_worker_space": str((outdir / "dask-worker-space").resolve()),
        "env": {k: os.environ.get(k) for k in [
            "OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"
        ]},
        "numpy": np.__version__,
    }
    path = outdir / "env.json"
    outdir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return str(path)

def main():
    args = parse_args()
    sizes = _parse_sizes(args.sizes)
    tile_h, tile_w = _parse_tile(args.tile)
    outdir = _resolve_outdir(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # driver-side BLAS/OpenMP cap (workers handled inside build_client)
    driver_threads = 1 if args.single_thread else args.blas_threads
    early_env_caps(driver_threads)

    env_json_path = _write_env_json(outdir)

    # connect/start client
    with build_client(
        scheduler_address=args.scheduler,
        n_workers=args.n_workers,
        threads_per_worker=args.threads_per_worker,
        processes=args.processes,
        memory_limit=args.memory_limit,
        set_worker_thread_caps=args.worker_blas_threads,
    ) as client:

        summary = run_dask_suite(
            use_synthetic=args.use_synthetic,
            data_dir=args.data_dir,
            ext=args.ext,
            idx=args.idx,
            sizes=tuple(sizes),
            tile_h=tile_h, tile_w=tile_w,
            repeats=args.repeats,
            nmu=args.nmu,
            outdir=str(outdir),
            scheduler=args.scheduler,
            n_workers=args.n_workers or 0,
            threads_per_worker=args.threads_per_worker,
            processes=bool(args.processes),
            memory_limit=args.memory_limit,
        )

    # make sure some human-friendly files exist
    run_txt = outdir / "RUN_SUMMARY.txt"
    lines = [
        f"Size={summary['size']}  Tile={summary['tile']}  Wall(s)={summary['wall_s']:.3f}",
        f"Workers={summary['n_workers']}  TPW={summary['threads_per_worker']}  Processes={summary['processes']}",
        f"Outdir={summary['outdir']}",
        f"Env JSON: {env_json_path}",
        "—",
        "NOTE: profiling_dask.csv (if any) lives in this same directory.",
    ]
    run_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if summary.get("files_used"):
        fu = outdir / "FILES_USED.txt"
        fu.write_text("\n".join(summary["files_used"]) + "\n", encoding="utf-8")

    print(f"Artifacts written to: {outdir}")
    for l in lines[:3]:
        print(l)
    if summary.get("files_used"):
        print(f"Files used ({len(summary['files_used'])}):")
        for p in summary["files_used"]:
            print(" -", p)

if __name__ == "__main__":
    main()
