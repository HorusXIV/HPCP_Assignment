# src/baseline/main.py
from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np

from src.common.profiling import run_baseline_suite
from src.common.responses import prepare_synthetic_responses  # your synthetic helper
from src.common.io import load_np_stack                       # if you want real data

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-synthetic", action="store_true")
    ap.add_argument("--sizes", type=str, default="14,64,256,1024")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--nmu", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="benchmark_out")
    return ap.parse_args()

def main():
    args = parse_args()

    if args.use_synthetic:
        nf, H, W = 6, 1024, 1024
        rng = np.random.default_rng(0)
        STACK = rng.random((1, nf, H, W), dtype=np.float32) * 1e3
    else:
        # Example: auto-glob data/np32/*.npz and load all
        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        NP_DIR = PROJECT_ROOT / "data" / "np32"
        STACK = load_np_stack(sorted(NP_DIR.glob("*.npz")), idx=-1)  # (N,6,H,W)

    T_RESP, T_RESP_LOGT, TEMPS = prepare_synthetic_responses(n_tresp=200, nt=24, nf=STACK.shape[1])

    sizes = [int(s) for s in args.sizes.split(",")]
    out = run_baseline_suite(
        STACK, T_RESP, T_RESP_LOGT, TEMPS,
        sizes=sizes, repeats=args.repeats, nmu=args.nmu, outdir=args.outdir
    )
    print("Artifacts written to:", out["outdir"])

if __name__ == "__main__":
    main()
