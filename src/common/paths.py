# src/common/paths.py
from __future__ import annotations
"""
Common path helpers for locating repository resources and constructing
run/output directories (including SLURM-aware defaults).

Highlights
----------
- `project_root()` : absolute path to the repository root (assumes this file lives at
  `src/common/paths.py`).
- `data_dir(*sub)` : `<repo>/data[/...]` convenience.
- `golden_dir(size)` : `<repo>/data/golden[/size]`.
- `default_run_dir(method, cli_outdir=None)` : picks a sensible output directory with
  the following precedence, then appends a timestamp and SLURM job hints:

    1) Explicit CLI `--outdir` (repo-relative if not absolute)
    2) Env `BENCH_OUTDIR`
    3) `SLURM_TMPDIR` or `{SCRATCH|PROJECT_SCRATCH}`
    4) `<repo>/benchmark_out`

  The final path is:
      <base>/<method>/<YYYYMMDD-HHMMSS[-jobName-jobId[-taskId]]>

- `resolve_relative_to_repo(path)` : make a relative path repo-relative.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Union


# ---------------------------
# Repo / project roots
# ---------------------------


def _repo_root() -> Path:
    """
    Internal: repository root (assumes file is at `src/common/paths.py`).

    Returns
    -------
    pathlib.Path
        Absolute path to the repo root.
    """
    return Path(__file__).resolve().parents[2]


def project_root() -> Path:
    """
    Public alias for the repository root.

    Returns
    -------
    pathlib.Path
        Absolute path to the repo root.
    """
    return _repo_root()


# ---------------------------
# SLURM context
# ---------------------------


def slurm_context() -> dict:
    """
    Collect a minimal SLURM context snapshot from environment variables.

    Returns
    -------
    dict
        {
          "job_id", "job_name",
          "array_job_id", "array_task_id",
          "tmpdir", "submit_dir",
          "scratch"
        }
    """
    env = os.environ
    return {
        "job_id": env.get("SLURM_JOB_ID"),
        "job_name": env.get("SLURM_JOB_NAME"),
        "array_job_id": env.get("SLURM_ARRAY_JOB_ID"),
        "array_task_id": env.get("SLURM_ARRAY_TASK_ID"),
        "tmpdir": env.get("SLURM_TMPDIR"),
        "submit_dir": env.get("SLURM_SUBMIT_DIR"),
        "scratch": env.get("SCRATCH") or env.get("PROJECT_SCRATCH"),
    }


# ---------------------------
# Common data paths
# ---------------------------


def data_dir(*sub: Union[str, Path]) -> Path:
    """
    Construct a path under `<repo>/data`.

    Examples
    --------
    >>> data_dir()
    <repo>/data
    >>> data_dir("np32")
    <repo>/data/np32

    Returns
    -------
    pathlib.Path
    """
    p = project_root() / "data"
    for s in sub:
        p = p / str(s)
    return p


def np32_dir() -> Path:
    """
    Default np32 dataset folder (used when --data-dir is omitted).

    Returns
    -------
    pathlib.Path
    """
    return data_dir("np32")


def golden_dir(size: Optional[Union[int, str]] = None) -> Path:
    """
    Default golden root or a specific size folder.

    Parameters
    ----------
    size : int | str | None
        If provided, returns `<repo>/data/golden/{size}`.

    Returns
    -------
    pathlib.Path
        `<repo>/data/golden` when size is None; otherwise the size subdir.
    """
    base = data_dir("golden")
    return base / str(size) if size is not None else base


# ---------------------------
# Run/output directories
# ---------------------------


def _timestamp() -> str:
    """
    Compact but sortable timestamp of the form YYYYMMDD-HHMMSS.

    Returns
    -------
    str
    """
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def default_run_dir(
    method: str, cli_outdir: Optional[str | os.PathLike] = None
) -> Path:
    """
    Compute a sensible default output directory and create it.

    Precedence for base directory
    -----------------------------
    1) `cli_outdir` (from `--outdir`); repo-relative if not absolute.
    2) Environment variable `BENCH_OUTDIR`
    3) SLURM scratch space: `SLURM_TMPDIR` or `{SCRATCH|PROJECT_SCRATCH}`
    4) `<repo>/benchmark_out`

    The final path suffix is `<method>/<timestamp[-job[-task]]>` where the
    optional job/task components are included when running under SLURM.

    Parameters
    ----------
    method : str
        Logical method name (e.g., "baseline", "dask", "gpu").
    cli_outdir : str | os.PathLike | None, default None
        Optional explicit base directory (e.g. from CLI).

    Returns
    -------
    pathlib.Path
        Absolute path to the created run directory.
    """
    method = str(method).strip().lower()

    # 1) explicit CLI
    if cli_outdir:
        p = Path(cli_outdir)
        base = p if p.is_absolute() else (_repo_root() / p)
    else:
        # 2) env override
        env_out = os.environ.get("BENCH_OUTDIR")
        if env_out:
            base = Path(env_out)
        else:
            sctx = slurm_context()
            # 3) SLURM_TMPDIR / SCRATCH if available; else repo folder
            base = Path(
                sctx.get("tmpdir")
                or sctx.get("scratch")
                or (_repo_root() / "benchmark_out")
            )

    # Suffix for uniqueness and traceability
    sctx = slurm_context()
    job_bits = []
    if sctx.get("job_name"):
        job_bits.append(str(sctx["job_name"]))
    if sctx.get("job_id"):
        job_bits.append(f"job{sctx['job_id']}")
    if sctx.get("array_task_id"):
        job_bits.append(f"task{sctx['array_task_id']}")

    stamp = _timestamp()
    tail = "-".join(filter(None, [stamp] + job_bits)) or stamp

    outdir = (base / method / tail).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def resolve_relative_to_repo(path_like: str | os.PathLike) -> Path:
    """
    Resolve a path against the repository root if it is not absolute.

    Parameters
    ----------
    path_like : str | os.PathLike
        Path to resolve.

    Returns
    -------
    pathlib.Path
        Absolute path.
    """
    p = Path(path_like)
    return p if p.is_absolute() else (_repo_root() / p).resolve()
