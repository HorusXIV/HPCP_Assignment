# src/baseline/cli.py
from __future__ import annotations
import argparse

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-synthetic", action="store_true", default=False)
    ap.add_argument("--sizes", type=str, default="14,64,256,1024") #,2048,4096
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--nmu", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="benchmark_out")
    ap.add_argument("--data-dir", type=str, default=None)
    ap.add_argument("--ext", type=str, default="*.npz")
    ap.add_argument("--idx", type=str, default="-1")
    ap.add_argument("--device", default="cpu")          # 'cpu' or '0'
    ap.add_argument("--single-thread", action="store_true", default=False)
    ap.add_argument("--blas-threads", type=int, default=None)
    ap.add_argument("--no-runtime-enforce", action="store_true", default=False)
    ap.add_argument("--nvtx", type=str, default=None)
    return ap.parse_args()
