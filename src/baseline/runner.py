# src/baseline/runner.py
from __future__ import annotations
from pathlib import Path
from contextlib import nullcontext
from typing import Sequence

from src.common.threads import early_env_caps, runtime_caps
from src.common.io import load_np_stack, default_files
from src.common.responses import prepare_synthetic_responses
from src.common.profiling import run_baseline_suite
from src.common.backend import xp_for
try:
    from src.common.nvtx import nvtx_range
except Exception:
    def nvtx_range(_): return nullcontext()

def parse_idx(spec: str):
    s = spec.strip()
    if s == "-1": return -1
    if ":" in s:
        parts = [(int(x) if x else None) for x in s.split(":")]
        return slice(*(parts + [None])[:3])
    if "," in s: return [int(x) for x in s.split(",") if x]
    return int(s)

def select_device(device_str: str):
    return None if device_str.lower() == "cpu" else int(device_str)

def load_stack(use_synth: bool, ext: str, data_dir: str | None, idx_spec: str):
    import numpy as np
    if use_synth:
        rng = np.random.default_rng(0)
        return rng.random((1, 6, 1024, 1024), dtype=np.float32) * 1e3, None
    patterns = tuple(p.strip() for p in ext.split(","))
    files = default_files(ext=patterns, directory=data_dir)
    return load_np_stack(files, idx=parse_idx(idx_spec), return_paths=True)

def run_benchmark(
    *, sizes: Sequence[int], repeats: int, nmu: int, outdir: str,
    use_synthetic: bool, ext: str, data_dir: str | None, idx: str,
    device_str: str, nvtx_label: str | None,
    threads_cap: int | None, runtime_enforce: bool
):
    # 1) Set early env caps before importing numpy
    early_env_caps(threads_cap)
    import numpy as np  # after caps

    # 2) Device selection (validates cupy if GPU)
    device = select_device(device_str)
    xp_for(device)  # sets CUDA device if needed

    # 3) Data
    stack, files_used = load_stack(use_synthetic, ext, data_dir, idx)
    nf = stack.shape[1] if stack.ndim == 4 else 6
    T_RESP, T_RESP_LOGT, TEMPS = prepare_synthetic_responses(n_tresp=200, nt=24, nf=nf)

    # 4) Contexts: NVTX + thread runtime caps
    nvtx_ctx = nvtx_range(nvtx_label) if nvtx_label else nullcontext()
    tp_ctx = nullcontext() if not runtime_enforce else runtime_caps(threads_cap)

    with nvtx_ctx:
        with tp_ctx:
            results = run_baseline_suite(
                stack, T_RESP, T_RESP_LOGT, TEMPS,
                sizes=list(sizes), repeats=repeats, nmu=nmu, outdir=str(outdir)
            )

    return {
        "results": results,
        "files_used": files_used,
        "device": "cpu" if device is None else f"cuda:{device}",
        "threads_cap": threads_cap,
        "runtime_enforced": runtime_enforce,
    }
