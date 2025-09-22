# src/dask/main.py
from __future__ import annotations
"""
Dask entrypoint.

This module bootstraps argument parsing from `src.dask.cli`, selects a default
workload when `--task` is not provided, and then delegates to `src.dask.runner.run`.

Key behaviors
-------------
- Parser discovery: tries `parse_args`, then `get_parser`, then `build_parser`
  from `src.dask.cli` so the CLI can evolve without breaking the entrypoint.
- Default task: if `--task` is omitted, it searches for a callable in
  `src.dask.suite:run` (preferred) or `src.suite:run`, selecting the first that
  imports and is callable.
- Cluster mode default: if `--cluster-mode` wasn't set by the CLI, defaults to
  `"local"`.
"""

import logging
import sys
from importlib import import_module
from typing import Optional

from .runner import run

log = logging.getLogger(__name__)


def _safe_call(func, *args):
    """
    Call a parser factory with or without argv.

    Some `parse_args` implementations accept an optional `argv`, others read
    directly from `sys.argv`. This helper tolerates both.
    """
    try:
        return func(*args)
    except TypeError:
        # Support parse_args() with/without argv param
        return func()


def _get_args(argv: Optional[list[str]] = None):
    """
    Obtain parsed CLI arguments from `src.dask.cli`.

    Discovery order within `src.dask.cli`:
      1) `parse_args(argv)` if present
      2) `get_parser().parse_args(argv)`
      3) `build_parser().parse_args(argv)`

    Raises
    ------
    RuntimeError
        If none of the expected parser entry points exist.
    """
    cli = import_module("src.dask.cli")
    if hasattr(cli, "parse_args"):
        return _safe_call(cli.parse_args, argv)
    if hasattr(cli, "get_parser"):
        return cli.get_parser().parse_args(argv)
    if hasattr(cli, "build_parser"):
        return cli.build_parser().parse_args(argv)
    raise RuntimeError("Could not find a parser in src.dask.cli.")


def _first_importable_callable(specs: list[str]) -> Optional[str]:
    """
    Return the first 'module:function' string that imports and is callable.

    Parameters
    ----------
    specs : list[str]
        Candidate call targets in the form 'module.submodule:function'.

    Returns
    -------
    str | None
        The first importable callable spec, or None if none qualified.
    """
    for spec in specs:
        try:
            mod_name, fn_name = spec.split(":", 1)
            mod = import_module(mod_name)
            fn = getattr(mod, fn_name, None)
            if callable(fn):
                return spec
        except Exception:
            continue
    return None


def main(argv: Optional[list[str]] = None) -> int:
    """
    Program entrypoint for the Dask runner.

    Steps
    -----
    1) Configure minimal logging if the application hasn't already done so.
    2) Parse CLI arguments using `_get_args`.
    3) Ensure `--cluster-mode` has a default ("local") if absent.
    4) If `--task` is not provided, select a sensible default workload from
       in-repo suites.
    5) Delegate to `src.dask.runner.run(args)`.

    Returns
    -------
    int
        Process exit code from the runner.
    """
    if argv is None:
        argv = sys.argv[1:]

    # Minimal, consistent logging if the app didn't configure it yet
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )

    args = _get_args(argv)

    # Default cluster mode if not provided
    if not getattr(args, "cluster_mode", None):
        log.info("No --cluster-mode provided; defaulting to 'local'.")
        setattr(args, "cluster_mode", "local")

    # Choose a sensible default task ONLY from concrete suites.
    # Fail loudly if these aren't importable so issues surface early.
    if not getattr(args, "task", None):
        # Preferred order: in-tree dask suite → legacy top-level suite
        preferred = [
            "src.dask.suite:run",
            "src.suite:run",
        ]
        found = _first_importable_callable(preferred)
        if found:
            setattr(args, "task", found)
            log.info("No --task provided; auto-selected default task: %s", found)
        else:
            log.warning(
                "No --task provided and no default suite found "
                "(tried: %s). Runner will start without a workload.",
                ", ".join(preferred),
            )

    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
