# src/common/profiling/reporting.py
from __future__ import annotations
"""
Human-friendly reporting helpers for benchmark runs.

This module provides:
  • `write_json`: persist a Python object to pretty-printed JSON
  • `write_run_card_md`: generate a compact Markdown card summarizing a run

Conventions
-----------
- Directories are created as needed.
- Markdown cards include an optional one-line environment blurb (if provided),
  the serialized bench row (as JSON fenced code), and optional notes.
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterable


def write_json(obj: Any, path: Path) -> Path:
    """
    Write a JSON file with pretty formatting.

    Parameters
    ----------
    obj : Any
        Object to serialize with `json.dumps`.
    path : pathlib.Path
        Destination file path.

    Returns
    -------
    pathlib.Path
        The written path.

    Notes
    -----
    - Parent directories are created if missing.
    - UTF-8 encoding is used.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return path


def write_run_card_md(
    *,
    outdir: Path,
    stamp: str,
    bench_row: Dict[str, Any],
    env: Dict[str, Any] | None = None,
    notes: Iterable[str] | None = None,
) -> Path:
    """
    Create a compact Markdown summary card for a single run.

    The card contains:
      - Title with run stamp
      - Optional one-line environment summary (Python/machine/CPU) if `env["platform"]` exists
      - A fenced JSON block containing `bench_row`
      - Optional notes section (list of bullet points)

    Parameters
    ----------
    outdir : pathlib.Path
        Output directory for the Markdown file.
    stamp : str
        Unique identifier for the run (typically a timestamp).
    bench_row : dict[str, Any]
        Benchmark metrics to embed as JSON in the card.
    env : dict[str, Any] | None, optional
        Optional environment snapshot; if it contains a `platform` key with
        fields `python`, `machine`, and `processor`, a one-liner is shown.
    notes : Iterable[str] | None, optional
        Optional list of notes to append as bullet points.

    Returns
    -------
    pathlib.Path
        Path to the generated Markdown file (`run_<stamp>.md`).
    """
    lines = [f"# Run {stamp}", ""]

    if env:
        # If an env snapshot with a 'platform' section was provided,
        # print a one-liner. Otherwise just dump keys at the end.
        plat = env.get("platform") or {}
        if plat:
            lines += [
                "## Environment",
                f"- Python: {plat.get('python')}  |  Machine: {plat.get('machine')}  |  CPU: {plat.get('processor')}",
                "",
            ]

    lines += ["## Bench Row", "```json", json.dumps(bench_row, indent=2), "```", ""]

    if notes:
        lines += ["## Notes"] + [f"- {n}" for n in notes] + [""]

    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"run_{stamp}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
