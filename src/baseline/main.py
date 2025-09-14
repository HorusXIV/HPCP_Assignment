# src/baseline/main.py
from __future__ import annotations
from pathlib import Path

from .cli import parse_args
from .runner import run_benchmark

def main():
    args = parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]
    outdir = Path(args.outdir)

    threads_cap = 1 if args.single_thread else args.blas_threads
    summary = run_benchmark(
        sizes=sizes, repeats=args.repeats, nmu=args.nmu, outdir=str(outdir),
        use_synthetic=args.use_synthetic, ext=args.ext, data_dir=args.data_dir, idx=args.idx,
        device_str=args.device, nvtx_label=args.nvtx,
        threads_cap=threads_cap, runtime_enforce=(not args.no_runtime_enforce),
    )

    print("Artifacts written to:", summary["results"]["outdir"])
    if summary["files_used"]:
        print(f"Files used ({len(summary['files_used'])}):")
        for p in summary["files_used"]:
            print(" -", p)
    print(f"Device: {summary['device']}")
    if summary["threads_cap"] is not None:
        print(f"Thread cap: {summary['threads_cap']} "
              f"{'(runtime-enforced)' if summary['runtime_enforced'] else '(env-only)'}")

if __name__ == "__main__":
    main()
