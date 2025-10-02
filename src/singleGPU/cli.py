from __future__ import annotations

import argparse
from typing import Optional, Tuple


def _parse_sizes(s: Optional[str]) -> Optional[Tuple[int, int]]:

	if not s:
		return None
	s_l = s.lower()
	if "x" in s_l:
		a, b = s_l.split("x", 1)
		return int(a), int(b)
	v = int(s)
	return (v, v)


def _parse_tile(s: Optional[str], default: Tuple[int, int] = (256, 256)) -> Tuple[int, int]:

	if not s:
		return default
	s_l = s.lower()
	if "x" in s_l:
		a, b = s_l.split("x", 1)
		return int(a), int(b)
	v = int(s)
	return (v, v)


def build_parser() -> argparse.ArgumentParser:

	p = argparse.ArgumentParser(
		prog="hpcp-singlegpu",
		description="Single-GPU DEM runner (Numba/CUDA accelerated).",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)

	# Data selection
	p.add_argument("--data-dir", type=str, default=None, help="Root directory of NPZ stacks.")
	p.add_argument("--ext", type=str, default="*.npz", help="Glob pattern for stacks under --data-dir.")
	p.add_argument("--idx", type=str, default="-1", help="Frame selector: integer, 'all', or '-1'.")

	# Problem sizing
	p.add_argument("--sizes", type=str, default=None, help="Output size: 'N' or 'HxW'.")
	p.add_argument("--tile", type=str, default="256", help="Tile size: 'T' or 'ThxTw'.")

	# Solver/benchmark
	p.add_argument("--nmu", type=int, default=42, help="Regularization / temperature resolution knob.")
	p.add_argument("--repeats", type=int, default=1, help="Number of timing repeats per size.")
	p.add_argument("--verify", action="store_true", help="Verify results against goldens if configured.")
	p.add_argument("--golden-root", type=str, default=None, help="Root with golden references for verification.")

	# Outputs
	p.add_argument("--bench-root", type=str, default=None, help="Benchmark output root.")

	# GPU and perf toggles
	p.add_argument("--device", type=str, default="cuda:0", help="CUDA device string or 'cpu'.")
	p.add_argument("--threads-cap", type=int, default=None, help="Optional BLAS/OMP threads cap on host.")

	return p


def parse_args(argv: Optional[list[str]] = None):

	parser = build_parser()
	args = parser.parse_args(argv)
	args.sizes = _parse_sizes(args.sizes)
	args.tile = _parse_tile(args.tile)
	return args


__all__ = ["build_parser", "parse_args", "_parse_sizes", "_parse_tile"]


