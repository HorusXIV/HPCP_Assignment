from __future__ import annotations
from pathlib import Path
from datetime import datetime
from typing import Any, Dict
import json
import numpy as np
import os

__all__ = ["timestamp", "make_run_dir", "save_npz_bundle", "save_meta", "load_npz_bundle", "default_tag"]

def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def default_tag(extra: list[str] | None = None) -> str:
    bits = []
    if "SLURM_JOB_ID" in os.environ:
        bits.append(f"slurm{os.environ['SLURM_JOB_ID']}")
    if extra:
        bits.extend(filter(None, extra))
    return "_".join(bits) if bits else ""

def make_run_dir(base: str | Path, approach: str, tag: str | None = None) -> Path:
    """
    Create output dir: data/output/{approach}/{YYYYMMDD-HHMMSS}_{tag}/
    """
    base = Path(base)
    name = timestamp() + (f"_{tag}" if tag else "")
    out = base / approach / name
    out.mkdir(parents=True, exist_ok=False)
    return out

def save_npz_bundle(run_dir: str | Path, filename: str = "results.npz", **arrays: np.ndarray) -> Path:
    run_dir = Path(run_dir)
    for k, v in arrays.items():
        if not isinstance(v, np.ndarray):
            raise TypeError(f"{k} must be numpy.ndarray, got {type(v).__name__}")
    path = run_dir / filename
    np.savez_compressed(path, **arrays)
    return path

def save_meta(run_dir: str | Path, meta: Dict[str, Any], filename: str = "meta.json") -> Path:
    path = Path(run_dir) / filename
    path.write_text(json.dumps(meta, indent=2))
    return path

def load_npz_bundle(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}