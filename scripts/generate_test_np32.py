#!/usr/bin/env python3
"""Generate a small deterministic np32 .npz for local development or CI.

Usage:
    python scripts/generate_test_np32.py --out data/np32/20170906_12_00_12.npz
"""
from pathlib import Path
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path, help="Output .npz path")
    ap.add_argument("--seed", type=int, default=12345, help="RNG seed for reproducibility")
    ap.add_argument("--h", type=int, default=32)
    ap.add_argument("--w", type=int, default=32)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    bands = rng.standard_normal((6, args.h, args.w)).astype(np.float32)
    np.savez(args.out, bands=bands)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
