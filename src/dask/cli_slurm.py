# src/dask/cli_slurm.py
from __future__ import annotations
"""
SLURM-related CLI helpers for the Dask runner.

This module piggybacks SLURM (dask-jobqueue) options onto an existing
ArgumentParser. It also provides a small normalization pass to make list-like
flags (that can be provided multiple times or as comma-separated strings)
behave consistently.

Usage
-----
parser = argparse.ArgumentParser(...)
add_slurm_arguments(parser)
args = parser.parse_args()
# Normalize list-like flags to flat lists:
getattr(args, "_normalize_slurm_flags", lambda ns: None)(args)
"""

import argparse
from typing import List, Optional


def _split(items: Optional[List[str]]) -> Optional[List[str]]:
    """
    Split a list of strings by commas, trim whitespace, and flatten.

    Examples
    --------
    ["A,B", "C"] -> ["A", "B", "C"]
    None         -> None
    []           -> None
    """
    if not items:
        return None
    out: List[str] = []
    for it in items:
        out.extend([p.strip() for p in it.split(",") if p.strip()])
    return out or None


def add_slurm_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Attach SLURM (dask-jobqueue) arguments to an existing parser.

    Notes
    -----
    We also register a default attribute on the parser's namespace:
    `_normalize_slurm_flags`, which is a callable that accepts the parsed
    namespace and normalizes list-like options to flat lists. After parsing,
    call:

        getattr(args, "_normalize_slurm_flags", lambda ns: None)(args)

    to ensure `job_extra`, `env_extra`, and `worker_extra_args` are normalized.
    """
    g = parser.add_argument_group("SLURM options (--cluster-mode slurm)")
    g.add_argument("--queue", default=None, help="SLURM partition/queue.")
    g.add_argument("--account", default=None, help="SLURM account / project.")
    g.add_argument("--project", default=None, help="Alias for --account.")
    g.add_argument("--cores", type=int, default=None, help="Cores per job/worker.")
    g.add_argument("--memory", default=None, help="Memory per job/worker (e.g. '8GB').")
    g.add_argument("--walltime", default=None, help="Walltime per job (HH:MM:SS).")
    g.add_argument("--interface", default=None, help="Network interface (e.g., ib0).")
    g.add_argument(
        "--log-directory", default=None, help="Directory for SLURM job logs."
    )
    g.add_argument(
        "--job-extra",
        action="append",
        default=None,
        help="Extra #SBATCH lines (comma-separated or repeated).",
    )
    g.add_argument(
        "--env-extra",
        action="append",
        default=None,
        help="Extra environment setup lines (comma-separated or repeated).",
    )
    g.add_argument(
        "--worker-extra-args",
        action="append",
        default=None,
        help="Extra dask-worker CLI args (comma-separated or repeated).",
    )
    g.add_argument(
        "--scheduler-options",
        default=None,
        help='Scheduler options for SLURMCluster, e.g. \'{"dashboard_address": ":8787"}\'.',
    )

    # Post-parse normalization hook (argparse has no native hook, so we expose a helper)
    parser.set_defaults(_normalize_slurm_flags=lambda ns: _normalize_ns(ns))


def _normalize_ns(ns: argparse.Namespace) -> None:
    """
    Normalize SLURM list-like flags on the parsed namespace.

    Converts possibly repeated / comma-separated values into flat lists:
    - ns.job_extra
    - ns.env_extra
    - ns.worker_extra_args
    """
    ns.job_extra = _split(getattr(ns, "job_extra", None))
    ns.env_extra = _split(getattr(ns, "env_extra", None))
    ns.worker_extra_args = _split(getattr(ns, "worker_extra_args", None))
