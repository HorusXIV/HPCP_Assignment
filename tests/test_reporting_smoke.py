"""Smoke test for the human-friendly run card writer.

Ensures `write_run_card_md`:
- creates the markdown file in the provided directory,
- includes the expected header with the run stamp, and
- renders the bench-row section.
"""

from __future__ import annotations

from pathlib import Path

from src.common.profiling.reporting import write_run_card_md


def test_write_run_card_md_smoke(tmp_path: Path) -> None:
    path = write_run_card_md(
        outdir=tmp_path,
        stamp="19700101-000000",
        bench_row={"mode": "baseline-cpu", "frames": 1, "H": 8, "W": 8},
        env={"platform": {"python": "3.x", "machine": "x", "processor": "x"}},
        notes=["smoke"],
    )
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "# Run 19700101-000000" in text
    assert "Bench Row" in text
