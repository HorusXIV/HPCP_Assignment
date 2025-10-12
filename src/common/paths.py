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


# ---------------------------
# Repo / project roots
# ---------------------------


def _repo_root() -> Path:
    """
    Internal: repository root (assumes file is at `src/common/paths.py`).

    Returns
    -------
    Path
        Absolute path to the repo root.
    """
    return Path(__file__).resolve().parents[2]


def project_root() -> Path:
    """
    Get the absolute path to the repository root.

    Returns
    -------
    Path
        Absolute path to the repo root.

    Examples
    --------
    >>> root = project_root()
    >>> (root / "src" / "common").exists()
    True
    """
    return _repo_root()


# ---------------------------
# SLURM context
# ---------------------------


def slurm_context() -> dict[str, str | None]:
    """
    Collect a minimal SLURM context snapshot from environment variables.

    Returns
    -------
    dict[str, str | None]
        Dictionary with SLURM environment variables:
        {
          "job_id", "job_name",
          "array_job_id", "array_task_id",
          "tmpdir", "submit_dir",
          "scratch", "nodelist", "cpus_per_task"
        }

    Examples
    --------
    >>> ctx = slurm_context()
    >>> if ctx["job_id"]:
    ...     print(f"Running in SLURM job {ctx['job_id']}")
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
        "nodelist": env.get("SLURM_NODELIST"),
        "cpus_per_task": env.get("SLURM_CPUS_PER_TASK"),
    }


def is_slurm_job() -> bool:
    """
    Check if currently running inside a SLURM job.

    Returns
    -------
    bool
        True if SLURM_JOB_ID is set, False otherwise.

    Examples
    --------
    >>> if is_slurm_job():
    ...     print("Running on SLURM cluster")
    """
    return os.environ.get("SLURM_JOB_ID") is not None


# ---------------------------
# Common data paths
# ---------------------------


def data_dir(*sub: str | Path) -> Path:
    """
    Construct a path under `<repo>/data`.

    Parameters
    ----------
    *sub : str | Path
        Optional subdirectories to append.

    Returns
    -------
    Path
        Path under data directory.

    Examples
    --------
    >>> data_dir()
    PosixPath('.../data')
    >>> data_dir("np32")
    PosixPath('.../data/np32')
    >>> data_dir("golden", "1024")
    PosixPath('.../data/golden/1024')
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
    Path
        Path to <repo>/data/np32

    Examples
    --------
    >>> np32 = np32_dir()
    >>> np32.name
    'np32'
    """
    return data_dir("np32")


def golden_dir(size: int | str | None = None) -> Path:
    """
    Default golden reference root or a specific size folder.

    Parameters
    ----------
    size : int | str | None, optional
        If provided, returns `<repo>/data/golden/{size}`.

    Returns
    -------
    Path
        `<repo>/data/golden` when size is None; otherwise the size subdir.

    Examples
    --------
    >>> golden_dir()
    PosixPath('.../data/golden')
    >>> golden_dir(1024)
    PosixPath('.../data/golden/1024')
    >>> golden_dir("512x256")
    PosixPath('.../data/golden/512x256')
    """
    base = data_dir("golden")
    return base / str(size) if size is not None else base


def benchmark_dir() -> Path:
    """
    Default benchmark output directory.

    Returns
    -------
    Path
        Path to <repo>/benchmark_out
    """
    return project_root() / "benchmark_out"


# ---------------------------
# Run/output directories
# ---------------------------


def _timestamp(include_microseconds: bool = False) -> str:
    """
    Compact but sortable timestamp.

    Parameters
    ----------
    include_microseconds : bool, default False
        If True, include microseconds for finer granularity.

    Returns
    -------
    str
        Timestamp in format YYYYMMDD-HHMMSS[.ffffff]

    Examples
    --------
    >>> ts = _timestamp()
    >>> len(ts)
    15
    >>> ts = _timestamp(include_microseconds=True)
    >>> len(ts)
    22
    """
    fmt = "%Y%m%d-%H%M%S"
    if include_microseconds:
        fmt += ".%f"
    return datetime.now().strftime(fmt)


def default_run_dir(
        method: str,
        cli_outdir: str | Path | None = None,
        create: bool = True,
) -> Path:
    """
    Compute a sensible default output directory and optionally create it.

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
        Will be lowercased and used as a subdirectory name.
    cli_outdir : str | Path | None, default None
        Optional explicit base directory (e.g. from CLI).
    create : bool, default True
        If True, create the directory (with parents) if it doesn't exist.

    Returns
    -------
    Path
        Absolute path to the run directory.

    Raises
    ------
    ValueError
        If method is empty after stripping.
    OSError
        If create=True but directory creation fails.

    Examples
    --------
    >>> outdir = default_run_dir("baseline")  # doctest: +SKIP
    >>> outdir.exists()
    True
    >>> outdir.name.startswith("20")  # Timestamp starts with year
    True
    """
    method = str(method).strip().lower()
    if not method:
        raise ValueError("method cannot be empty")

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
    job_bits: list[str] = []

    if sctx.get("job_name"):
        # Sanitize job name (remove special chars that might cause issues)
        job_name = str(sctx["job_name"]).replace("/", "-").replace(" ", "_")
        job_bits.append(job_name)

    if sctx.get("job_id"):
        job_bits.append(f"job{sctx['job_id']}")

    if sctx.get("array_task_id"):
        job_bits.append(f"task{sctx['array_task_id']}")

    stamp = _timestamp()
    tail = "-".join(filter(None, [stamp] + job_bits)) or stamp

    outdir = (base / method / tail).resolve()

    if create:
        try:
            outdir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise OSError(
                f"Failed to create output directory {outdir}: {e}"
            ) from e

    return outdir


def resolve_relative_to_repo(path_like: str | Path) -> Path:
    """
    Resolve a path against the repository root if it is not absolute.

    Parameters
    ----------
    path_like : str | Path
        Path to resolve.

    Returns
    -------
    Path
        Absolute path.

    Examples
    --------
    >>> p = resolve_relative_to_repo("data/np32")
    >>> p.is_absolute()
    True
    >>> p.name
    'np32'

    >>> p = resolve_relative_to_repo("/absolute/path")
    >>> str(p)
    '/absolute/path'
    """
    p = Path(path_like)
    return p if p.is_absolute() else (_repo_root() / p).resolve()


def ensure_dir(path: str | Path, parents: bool = True) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Parameters
    ----------
    path : str | Path
        Directory path to ensure exists.
    parents : bool, default True
        If True, create parent directories as needed.

    Returns
    -------
    Path
        Absolute path to the directory.

    Raises
    ------
    OSError
        If directory creation fails.
    FileExistsError
        If path exists but is not a directory.

    Examples
    --------
    >>> tmpdir = ensure_dir("/tmp/test_dir")  # doctest: +SKIP
    >>> tmpdir.exists()
    True
    >>> tmpdir.is_dir()
    True
    """
    p = Path(path).resolve()

    if p.exists():
        if not p.is_dir():
            raise FileExistsError(
                f"Path exists but is not a directory: {p}"
            )
        return p

    try:
        p.mkdir(parents=parents, exist_ok=True)
    except OSError as e:
        raise OSError(f"Failed to create directory {p}: {e}") from e

    return p


def list_data_files(
        pattern: str = "*.npz",
        directory: str | Path | None = None,
) -> list[Path]:
    """
    List data files matching a pattern.

    Parameters
    ----------
    pattern : str, default "*.npz"
        Glob pattern to match.
    directory : str | Path | None, optional
        Directory to search. If None, uses <repo>/data/np32.

    Returns
    -------
    list[Path]
        Sorted list of matching files.

    Examples
    --------
    >>> files = list_data_files("*.npz")  # doctest: +SKIP
    >>> all(f.suffix == ".npz" for f in files)
    True
    """
    if directory is None:
        directory = np32_dir()
    else:
        directory = Path(directory)

    if not directory.exists():
        return []

    return sorted(directory.glob(pattern))


def get_run_metadata() -> dict[str, str | None]:
    """
    Collect metadata about the current run environment.

    Returns
    -------
    dict
        Metadata including:
        - hostname
        - user
        - slurm_job_id
        - slurm_job_name
        - timestamp

    Examples
    --------
    >>> meta = get_run_metadata()
    >>> "hostname" in meta
    True
    >>> "timestamp" in meta
    True
    """
    import socket
    import getpass

    sctx = slurm_context()

    return {
        "hostname": socket.gethostname(),
        "user": getpass.getuser(),
        "slurm_job_id": sctx.get("job_id"),
        "slurm_job_name": sctx.get("job_name"),
        "slurm_array_task_id": sctx.get("array_task_id"),
        "timestamp": _timestamp(include_microseconds=True),
    }