from __future__ import annotations
from pathlib import Path

from .cli import parse_args
from .runner import run_dask_suite


def _parse_sizes(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x]


def _parse_tile(s: str) -> tuple[int, int]:
    a, b = (int(x) for x in s.split(","))
    return a, b


def main():
    args = parse_args()
    sizes = _parse_sizes(args.sizes)
    tile_h, tile_w = _parse_tile(args.tile)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary = run_dask_suite(
        # data selection / sizing
        use_synthetic=args.use_synthetic,
        data_dir=args.data_dir,
        ext=args.ext,
        idx=args.idx,
        sizes=tuple(sizes),
        # tiling
        tile_h=tile_h,
        tile_w=tile_w,
        # algo / profiling
        repeats=args.repeats,
        nmu=args.nmu,
        outdir=str(outdir),
        # cluster / scheduler
        scheduler=args.scheduler,
        n_workers=args.n_workers if args.n_workers is not None else 4,
        threads_per_worker=args.threads_per_worker,
        processes=args.processes,
        memory_limit=args.memory_limit if args.memory_limit is not None else "auto",
    )

    print("Artifacts written to:", summary["outdir"])
    print(
        f"Size={summary['size']}  Tile={summary['tile']}  "
        f"Wall(s)={summary['wall_s']:.3f}  "
        f"Workers={summary['n_workers']}  TPW={summary['threads_per_worker']}  "
        f"Processes={summary['processes']}"
    )
    if summary.get("files_used"):
        print(f"Files used ({len(summary['files_used'])}):")
        for p in summary["files_used"]:
            print(" -", p)


if __name__ == "__main__":
    main()
