"""
Baseline package: CLI entrypoint + orchestration for benchmarks.
"""

from .cli import parse_args
from .runner import run_benchmark

__all__ = ("parse_args", "run_benchmark")
