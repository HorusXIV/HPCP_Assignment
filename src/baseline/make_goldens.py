# src/baseline/make_goldens.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np

# Reuse your runner utilities
from src.baseline.run import load_test_data, run_baseline_solve, crop_center
from src.common.dataio.save import save_npz_bundle, save_meta


def _parse_sizes_csv(s: str) -> List[int]:
    """Parse '128' or '128,256,512' or '256x256' tokens into a list of ints."""
    tokens = [t.strip() for t in s.replace(" ", "").split(",") if t.strip()]
    sizes: List[int] = []
    for t in tokens:
        if "x" in t.lower():
            a, b = t.lower().split("x", 1)
            a, b = int(a), int(b)
            if a != b:
                raise ValueError(f"Non-square token '{t}'. Use N or NxN with N==N.")
            sizes.append(a)
        else:
            sizes.append(int(t))
    if not sizes:
        raise ValueError("No sizes parsed from --sizes.")
    if any(n <= 0 for n in sizes):
        raise ValueError("All sizes must be positive.")
    return sizes


def _resolve_npz(data_dir: Optional[str], ext: str, idx: str) -> Path:
    """Pick one NPZ from a directory using glob and idx ('-1'/'all' or integer)."""
    if not data_dir:
        raise ValueError("--data-dir is required.")
    matches = sorted(Path(data_dir).glob(ext))
    if not matches:
        raise FileNotFoundError(f"No files matching {ext!r} under {data_dir!r}.")
    if idx in ("-1", "all"):
        return matches[0]
    try:
        i = int(idx)
    except ValueError as e:
        raise ValueError(f"--idx must be integer, 'all', or '-1'; got {idx!r}") from e
    if i < 0 or i >= len(matches):
        raise IndexError(f"--idx {i} out of range [0, {len(matches)-1}] for {data_dir}/{ext}")
    return matches[i]


def _save_golden(outroot: Path,
                 size: int,
                 result: dict,
                 source_file: Optional[Path],
                 frame_idx: Optional[int] = None,
                 r_scale: Optional[np.ndarray] = None) -> None:
    """
    Write results.npz and meta.json into:
      data/output/goldens/{size}/[frame_{i}/]
    """
    base = outroot / "goldens" / str(size)
    outdir = base / f"frame_{frame_idx}" if frame_idx is not None else base
    outdir.mkdir(parents=True, exist_ok=True)

    # Arrays
    save_npz_bundle(
        outdir,
        filename="results.npz",
        demmap=result["demmap"],
        edemmap=result["edemmap"],
        logt=result["logt"],
        chisq=result["chisq"],
        dn_reg=result["dn_reg"],
    )

    # Metadata
    H, W, NT = result["demmap"].shape
    meta = dict(
        source_file=str(source_file) if source_file else None,
        size=int(size),
        frame_idx=int(frame_idx) if frame_idx is not None else None,
        demmap_shape=(int(H), int(W), int(NT)),
        edemmap_shape=tuple(int(x) for x in result["edemmap"].shape),
        chisq_shape=tuple(int(x) for x in result["chisq"].shape),
        dn_reg_shape=tuple(int(x) for x in result["dn_reg"].shape),
        logt_len=int(result["logt"].shape[0]),
        elapsed_seconds=float(result["elapsed_seconds"]),
        em_mean=float(result["checks"]["em_mean"]),
        chisq_mean=float(result["checks"]["chisq_mean"]),
        r_scale=list(map(float, r_scale)) if r_scale is not None else None,
    )
    save_meta(outdir, meta, filename="meta.json")
    print(f"[golden] wrote {outdir/'results.npz'} (+ meta.json)")


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="make_goldens",
        description="Create golden outputs under data/output/goldens/{size}/",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data-dir", type=str, required=True,
                    help="Directory with NPZ stacks (contains 'bands'/'stack' etc.)")
    ap.add_argument("--ext", type=str, default="*.npz",
                    help="Glob pattern for stacks inside --data-dir")
    ap.add_argument("--idx", type=str, default="0",
                    help="Which file to use: integer index, or 'all'/'-1' for first match")
    ap.add_argument("--sizes", type=str, default="128",
                    help="Comma-separated list of square crop sizes, e.g. '128,256,512'")
    ap.add_argument("--nmu", type=int, default=42,
                    help="Regularization knob passed to solver")
    ap.add_argument("--validate", action="store_true",
                    help="Enable solver I/O validation if supported")
    ap.add_argument("--out-root", type=Path, default=Path("data/output"),
                    help="Root folder for outputs")
    ap.add_argument("--all-frames", action="store_true",
                    help="Create goldens for every frame in the stack.")
    ap.add_argument("--frames", type=str, default="",
                    help="Comma-separated frame indices to export (overrides --all-frames if set).")
    args = ap.parse_args()

    npz_path = _resolve_npz(args.data_dir, args.ext, args.idx)
    print(f"[golden] using data file: {npz_path}")

    # Load data + calibration; run.py will synthesize bins/responses if absent
    STACK, T_RESP, T_RESP_LOGT, TEMPS = load_test_data(npz_path)

    # Decide which frames to export
    F = STACK.shape[0]
    if args.frames.strip():
        frame_idxs = [int(t) for t in args.frames.split(",") if t.strip()]
    elif args.all_frames:
        frame_idxs = list(range(F))
    else:
        frame_idxs = [0]  # default: one representative frame

    sizes = _parse_sizes_csv(args.sizes)

    for fi in frame_idxs:
        frame_6hw = STACK[fi]  # (6,H,W)
        H, W = frame_6hw.shape[1], frame_6hw.shape[2]

        for size in sizes:
            if size > min(H, W):
                raise ValueError(f"Requested size {size} exceeds frame {H}x{W}.")

            # (6,H,W) -> (H,W,6) -> crop -> back to (6,h,w)
            hw6 = np.moveaxis(frame_6hw, 0, -1)
            hw6_crop = crop_center(hw6, size)
            crop_6hw = np.moveaxis(hw6_crop, -1, 0).astype(np.float32, copy=False)

            # --- hygiene & scaling ---
            # 1) Clamp negatives and add a tiny floor so sqrt/weights don't explode at zeros
            np.maximum(crop_6hw, 0.0, out=crop_6hw)
            crop_6hw += 1.0  # small data floor (counts); tweak if needed

            # 2) Scale synthetic responses per-channel to match DN magnitude
            R = T_RESP.copy()
            chan_means = crop_6hw.mean(axis=(1, 2)).astype(np.float32)  # (6,)
            chan_means = np.clip(chan_means, 1e-6, None)
            R *= chan_means[None, :]

            # --- Diagnostic: naive center-pixel least squares (>=0) ---
            h, w = crop_6hw.shape[1:]
            cy, cx = h // 2, w // 2
            y = crop_6hw[:, cy, cx].astype(np.float64)  # (6,)
            A_T = R.T.astype(np.float64, copy=False)  # (6, nt)
            x_ls, *_ = np.linalg.lstsq(A_T, y, rcond=None)
            x_ls = np.clip(x_ls, 0.0, None)
            print(f"[golden] naive center-pixel EM (lstsq>=0): {float(np.sum(x_ls)):.6g}")

            # --- Run baseline solver on the crop ---
            result = run_baseline_solve(
                crop_6hw,  # (6,h,w)
                R,  # scaled response
                T_RESP_LOGT,
                TEMPS,
                validate=args.validate,
                nmu=args.nmu,
            )
            R = T_RESP.copy()  # or the scaled R you solved with
            dem = result["demmap"].astype(np.float64)  # (h,w,nt)
            A = (R.T).astype(np.float64, copy=False)  # (6, nt)
            dn_hat = np.tensordot(A, dem, axes=([1], [2]))  # (6,h,w)
            result["dn_reg"] = dn_hat.astype(np.float32, copy=False)

            # Guard logt: if solver returns wrong shape, store the known centers
            logt_out = result["logt"]
            if not isinstance(logt_out, np.ndarray) or logt_out.ndim != 1:
                logt_out = T_RESP_LOGT.astype(np.float32)
                result["logt"] = logt_out

            # If solver DEM is all zeros, fall back to NNLS per pixel (optional but practical)
            if float(np.max(result["demmap"])) == 0.0:
                print("[golden] solver DEM is zero — computing NNLS fallback golden for this crop.")
                # Build a per-pixel nonnegative least squares DEM (fast nnls-lite via lstsq+clip)
                nt = logt_out.shape[0]
                dem_nnls = np.empty((h, w, nt), dtype=np.float32)
                edem_nnls = np.zeros_like(dem_nnls)
                chisq_nnls = np.empty((h, w), dtype=np.float32)

                A = A_T  # (6, nt)
                for yy in range(h):
                    Y = crop_6hw[:, yy, :].astype(np.float64)  # (6, w)
                    # Solve each column independently
                    # x = argmin ||A x - y||_2 ; x>=0  (approx via clip)
                    X, *_ = np.linalg.lstsq(A, Y, rcond=None)  # (nt, w)
                    X = np.clip(X, 0.0, None)
                    dem_nnls[yy, :, :] = X.T.astype(np.float32, copy=False)
                    resid = (A @ X - Y)  # (6, w)
                    chisq_nnls[yy, :] = np.sum(resid * resid, axis=0).astype(np.float32)

                # Replace result payload with NNLS fallback
                result["demmap"] = dem_nnls
                result["edemmap"] = edem_nnls
                result["chisq"] = chisq_nnls
                result["logt"] = T_RESP_LOGT.astype(np.float32, copy=False)
                # dn_reg stays zeros in this fallback (no regularization)

            # Save golden (with r_scale saved in meta)
            _save_golden(
                outroot=args.out_root,
                size=size,
                result=result,
                source_file=npz_path,
                frame_idx=fi if len(frame_idxs) > 1 else None,
                r_scale=chan_means,
            )

    print("[golden] done.")


if __name__ == "__main__":
    main()
