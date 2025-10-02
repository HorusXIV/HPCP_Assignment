from __future__ import annotations

"""
Single-GPU entrypoint.


Parses CLI args from src.singleGPU.cli and dispatches to the single-GPU runner.
"""

import sys
import logging
from typing import Optional


from .cli import parse_args
from .runner import run_benchmark_single_gpu

def main(argv: Optional[list[str]] = None) -> int:

	if argv is None:
		argv = sys.argv[1:]
	if not logging.getLogger().handlers:
		logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


	args = parse_args(argv)


	_ = run_benchmark_single_gpu(
	data_dir=args.data_dir,
	ext=args.ext,
	idx=args.idx,
	sizes=args.sizes or (),
	tile=args.tile,
	nmu=args.nmu,
	bench_root=args.bench_root,
	verify=args.verify,
	golden_root=args.golden_root,
	device=args.device,
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())