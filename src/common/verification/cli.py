# src/common/verification/cli.py
from __future__ import annotations
"""
CLI utilities to create, verify, and benchmark DEM "golden" results.

This module exposes a command-line interface with three subcommands:

1) make-goldens
   Create/update reference outputs ("goldens") for one or more sizes.
   Writes, per-size, a pair:
     - {golden-root}/{HxW}/baseline.npz  (arrays: demmap, edemmap, chisq, logT_bins)
     - {golden-root}/{HxW}/baseline.json (metadata: inputs, shapes, params)

2) verify
   Compute results for a single size and compare against a specified golden NPZ.
   Supports running either the local CPU solver or a small local Dask cluster.

3) bench
   Iterate over a list of sizes, verify each against its golden, and time the run.
   Can optionally write a JSON summary.

Conventions
-----------
- Input stacks are a list of NPZ files (each containing 'bands' shaped (6, H, W)).
- `--index` selects which frame(s) to stack; use "-1" for the last (or "A:B").
- `--sizes` crops after stacking; "256" means (256,256), "256x512" means (256,512).
- For Dask runs `--n-workers` controls the number of workers (1 thread/worker).
- By default verification auto-matches the solver `nt` to the golden's logT bins.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Sequence, Tuple, List

import numpy as np

from src.common.dataio import default_files, load_np_stack, frame_for_solver
from src.common.ssolver import solve_tile_all  # noqa: F401  (imported for side effects if needed)
from src.common.solver import solve_tile_all
# handle both implementations of compare_to_golden: (ok, reports) or dict
from src.common.verification.verify import compare_to_golden


# ---------- helpers ----------


def _parse_sizes(s: Optional[str]) -> Optional[Tuple[int, int]]:
    """
    Parse size strings into an (H, W) tuple.

    Accepts forms like:
      - "256"        → (256, 256)
      - "256x512"    → (256, 512)
      - "256,512"    → (256, 512)

    Returns None if `s` is falsy.
    """
    if not s:
        return None
    s = s.lower().replace("x", ",").replace(" ", "")
    parts = [p for p in s.split(",") if p]
    if len(parts) == 1:
        h = int(parts[0])
        return (h, h)
    if len(parts) >= 2:
        return (int(parts[0]), int(parts[1]))
    return None


def _select_idx_arg(s: str) -> int | slice:
    """
    Parse an index argument that can be either a single integer or a slice "A:B".
    Returns an `int` or a built `slice`.
    """
    s = s.strip()
    if ":" in s:
        a, b = s.split(":", 1)
        start = int(a) if a else None  # type: ignore[assignment]
        stop = int(b) if b else None  # type: ignore[assignment]
        return slice(start, stop)  # type: ignore[return-value]
    return int(s)


def _save_npz(
    path: Path,
    *,
    dem: np.ndarray,
    edem: np.ndarray,
    chisq: np.ndarray,
    logt: np.ndarray,
) -> None:
    """Persist solver outputs to a compressed NPZ with standardized keys."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, demmap=dem, edemmap=edem, chisq=chisq, logT_bins=logt)


def _print_verify_report(rep_obj) -> bool:
    """
    Normalize and print the verification report.

    Handles two shapes:
      - tuple(ok: bool, reports: list[dict])
      - dict with 'ok' and 'reports' keys

    Returns True if verification passed.
    """
    # Normalize both possible return shapes
    if isinstance(rep_obj, tuple) and len(rep_obj) == 2:
        ok, reports = rep_obj
        rep = {"ok": bool(ok), "reports": list(reports)}
    else:
        rep = dict(rep_obj)
        rep.setdefault("ok", all(r.get("equal", True) for r in rep.get("reports", [])))

    ok = bool(rep.get("ok", True))
    status = "OK" if ok else "MISMATCH"
    print(f"[verify] {status}")
    for item in rep.get("reports", []):
        name = item.get("name")
        eq = item.get("equal")
        note = item.get("note", "")
        print(f"  - {name:10s}: {'OK' if eq else 'BAD'}  {note}")
    return ok


def _load_frame_from_data(
    data_dir: str | Path, ext: str, index: int | slice, crop: Optional[Tuple[int, int]]
) -> np.ndarray:
    """
    Discover NPZ files, stack selected frames, and return one (H, W, 6) frame,
    optionally cropped to (H, W).
    """
    files = default_files(data_dir, ext=ext)
    if not files:
        raise FileNotFoundError(f"No files found in {data_dir!r} matching {ext!r}")
    stack = load_np_stack(files, idx=index, channels_last=True)
    frame = frame_for_solver(stack, 0)  # (H,W,6)
    if crop:
        H, W = crop
        frame = frame[:H, :W, :]
    return frame


def _compute_with_backend(
    frame: np.ndarray, *, module: str, nmu: int, nt: Optional[int], n_workers: int = 4
):
    """
    Compute (dem, edem, chisq, logt) with the requested backend and measure time.

    Parameters
    ----------
    frame : np.ndarray
        (H, W, 6) input frame.
    module : {"cpu","baseline","dask","gpu"}
        Backend selector.
    nmu, nt : int | None
        Regularization / DEM bin controls forwarded to the solver.
    n_workers : int, default 4
        For Dask, number of workers in a local cluster.

    Returns
    -------
    (dem, edem, chisq, logt), elapsed_seconds
    """
    t0 = time.perf_counter()
    module = module.lower()
    if module in ("cpu", "baseline"):
        dem, edem, chisq, logt = solve_tile_all(frame, nmu=nmu, nt=nt)
    elif module == "dask":
        from dask.distributed import Client, LocalCluster

        cluster = LocalCluster(
            n_workers=n_workers, threads_per_worker=1, processes=False
        )
        client = Client(cluster)
        try:
            fut = client.submit(solve_tile_all, frame, nmu=nmu, nt=nt)
            dem, edem, chisq, logt = client.gather(fut)
        finally:
            client.close()
            cluster.close()
    elif module == "gpu":
        raise NotImplementedError(
            "GPU backend not wired here; add your CUDA/CuPy path and call it from this switch."
        )
    else:
        raise ValueError(f"Unknown --module {module!r}; use: cpu | dask | gpu")

    dt = time.perf_counter() - t0
    return dem, edem, chisq, logt, dt


def _golden_paths(golden_root: str | Path, size_dir: str) -> Tuple[Path, Path]:
    """Return the standard (npz_path, json_path) under a size directory."""
    size_path = Path(golden_root) / size_dir
    return size_path / "baseline.npz", size_path / "baseline.json"


def _write_golden_pair(
    out_npz: Path, out_json: Path, *, dem, edem, chisq, logt, meta: dict
) -> None:
    """Persist the NPZ + JSON pair for a golden result."""
    _save_npz(out_npz, dem=dem, edem=edem, chisq=chisq, logt=logt)
    out_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")


# ---------- CLI ----------


def get_parser() -> argparse.ArgumentParser:
    """
    Build the top-level argument parser with subcommands:
      - make-goldens
      - verify
      - bench
    """
    p = argparse.ArgumentParser(
        prog="python -m src.common.verification.cli",
        description="Create and verify DEM goldens; benchmark modules against existing goldens.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # make-goldens (writes data/golden/{HxW}/baseline.npz + baseline.json)
    mk = sub.add_parser(
        "make-goldens", help="Create/update goldens under a golden root."
    )
    mk.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory with .npz stacks (e.g., data/np32)",
    )
    mk.add_argument(
        "--golden-root",
        type=str,
        default="data/golden",
        help="Output root for {size}/baseline.npz,json",
    )
    mk.add_argument(
        "--sizes",
        type=str,
        required=True,
        help='Comma-separated sizes, e.g. "64,256,1024"',
    )
    mk.add_argument(
        "--index",
        type=str,
        default="-1",
        help="Which frame(s) to load; default -1 (last)",
    )
    mk.add_argument(
        "--ext", type=str, default="*.npz", help="Glob for files (default: *.npz)"
    )
    mk.add_argument("--nmu", type=int, default=42)
    mk.add_argument("--nt", type=int, default=None)

    # verify (single comparison against one golden NPZ)
    vf = sub.add_parser("verify", help="Verify one run against a golden NPZ.")
    vf.add_argument("--data-dir", type=str, required=True)
    vf.add_argument("--ext", type=str, default="*.npz")
    vf.add_argument("--index", type=str, default="-1")
    vf.add_argument(
        "--sizes", type=str, required=True, help='Crop like "256,256" or "256" (square)'
    )
    vf.add_argument("--golden", type=str, required=True, help="Path to baseline.npz")
    vf.add_argument("--module", type=str, default="cpu", choices=["cpu", "dask", "gpu"])
    vf.add_argument("--nmu", type=int, default=42)
    vf.add_argument("--nt", type=int, default=None)
    vf.add_argument("--n-workers", type=int, default=4)

    # bench (iterate over {size} directories under golden-root, verify + time)
    bn = sub.add_parser(
        "bench", help="Verify and measure runtime against existing goldens by size."
    )
    bn.add_argument("--data-dir", type=str, required=True)
    bn.add_argument("--golden-root", type=str, default="data/golden")
    bn.add_argument(
        "--sizes", type=str, required=True, help='Comma-separated sizes, e.g. "128,256"'
    )
    bn.add_argument("--ext", type=str, default="*.npz")
    bn.add_argument("--index", type=str, default="-1")
    bn.add_argument("--module", type=str, default="cpu", choices=["cpu", "dask", "gpu"])
    bn.add_argument("--nmu", type=int, default=42)
    bn.add_argument("--nt", type=int, default=None)
    bn.add_argument("--n-workers", type=int, default=4)
    bn.add_argument(
        "--json", dest="json_out", type=str, default=None, help="Write a JSON summary"
    )

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Entry point for the CLI.

    Returns
    -------
    int
        Exit code: 0 on success, 2 on verification mismatch (where applicable).
    """
    args = get_parser().parse_args(argv)

    if args.cmd == "make-goldens":
        sizes = [s.strip() for s in str(args.sizes).split(",") if s.strip()]
        idx = _select_idx_arg(args.index)
        files = default_files(args.data_dir, ext=args.ext)
        if not files:
            raise FileNotFoundError(
                f"No files found in {args.data_dir!r} matching {args.ext!r}"
            )

        stack = load_np_stack(files, idx=idx, channels_last=True)

        for s in sizes:
            crop = _parse_sizes(s)
            frame = frame_for_solver(stack, 0)
            if crop:
                H, W = crop
                frame = frame[:H, :W, :]

            dem, edem, chisq, logt = solve_tile_all(frame, nmu=args.nmu, nt=args.nt)

            out_npz, out_json = _golden_paths(args.golden_root, s)
            meta = dict(
                files=[str(p) for p in files],
                index=str(args.index),
                sizes=[crop[0], crop[1]] if crop else None,
                nmu=int(args.nmu),
                nt=(int(args.nt) if args.nt is not None else None),
                shapes=dict(
                    dem=list(dem.shape),
                    edem=list(edem.shape),
                    chisq=list(chisq.shape),
                    logT_bins=int(logt.shape[0]),
                ),
            )
            out_npz.parent.mkdir(parents=True, exist_ok=True)
            _write_golden_pair(
                out_npz, out_json, dem=dem, edem=edem, chisq=chisq, logt=logt, meta=meta
            )
            print(f"[golden] wrote {out_npz} and {out_json}")

        return 0

    if args.cmd == "verify":
        # --- Optimal fix: auto-match NT to golden unless user provided --nt ---
        if args.nt is None:
            try:
                g = np.load(args.golden)
                nt_g = int(g["logT_bins"].shape[0])
                args.nt = nt_g
                print(f"[verify] auto-matching NT to golden: nt={args.nt}")
            except Exception as e:
                print(
                    f"[verify] warning: could not read golden NT ({e}); using solver default"
                )

        crop = _parse_sizes(args.sizes)
        frame = _load_frame_from_data(
            args.data_dir, args.ext, _select_idx_arg(args.index), crop
        )
        dem, edem, chisq, logt, dt = _compute_with_backend(
            frame,
            module=args.module,
            nmu=args.nmu,
            nt=args.nt,
            n_workers=getattr(args, "n_workers", 4),
        )
        ok = _print_verify_report(
            compare_to_golden(
                Path(args.golden), demmap=dem, edemmap=edem, chisq=chisq, logT_bins=logt
            )
        )
        print(f"[time] {args.module} {args.sizes}: {dt:.3f}s")
        return 0 if ok else 2

    if args.cmd == "bench":
        sizes = [s.strip() for s in str(args.sizes).split(",") if s.strip()]
        idx = _select_idx_arg(args.index)
        files = default_files(args.data_dir, ext=args.ext)
        if not files:
            raise FileNotFoundError(
                f"No files found in {args.data_dir!r} matching {args.ext!r}"
            )

        stack = load_np_stack(files, idx=idx, channels_last=True)

        results: List[dict] = []
        print(f"[bench] module={args.module} sizes={sizes} n_workers={args.n_workers}")
        for s in sizes:
            crop = _parse_sizes(s)
            if crop is None:
                raise ValueError(f"Invalid size spec: {s}")
            H, W = crop
            frame = frame_for_solver(stack, 0)[:H, :W, :]

            golden_npz, golden_json = _golden_paths(args.golden_root, s)
            if not golden_npz.exists():
                print(f"[bench] missing golden: {golden_npz}")
                results.append(
                    dict(size=s, ok=False, seconds=None, note="missing golden")
                )
                continue

            # --- Optimal fix in bench: auto-match NT to the size's golden ---
            nt_eff = args.nt
            if nt_eff is None:
                try:
                    g = np.load(golden_npz)
                    nt_eff = int(g["logT_bins"].shape[0])
                    print(f"[bench] {s}: auto nt={nt_eff}")
                except Exception as e:
                    print(
                        f"[bench] {s}: warn: could not read golden NT ({e}); using solver default"
                    )

            dem, edem, chisq, logt, dt = _compute_with_backend(
                frame,
                module=args.module,
                nmu=args.nmu,
                nt=nt_eff,
                n_workers=args.n_workers,
            )
            ok = _print_verify_report(
                compare_to_golden(
                    golden_npz, demmap=dem, edemmap=edem, chisq=chisq, logT_bins=logt
                )
            )
            print(f"[time] {args.module} {s}: {dt:.3f}s")
            results.append(dict(size=s, ok=bool(ok), seconds=float(dt)))

        if args.json_out:
            Path(args.json_out).write_text(
                json.dumps(results, indent=2), encoding="utf-8"
            )
            print(f"[bench] wrote JSON -> {args.json_out}")

        return 0 if all(r.get("ok") for r in results if "ok" in r) else 2

    raise RuntimeError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
