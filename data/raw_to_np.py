#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
raw_to_np.py

Build flat NPZ stacks from raw AIA FITS using a manifest (fast) or a filesystem scan (fallback).

Key changes:
- Flat output layout: <np_out_dir>/YYYYMMDD_HHMMSS.npz
- Each NPZ contains the stack and rich metadata:
    bands            : (Nbands, Y, X)
    wavelengths      : (Nbands,)
    ts_utc           : scalar str (representative ISO time)
    bucket_12s       : scalar str "HH_MM_SS"
    fits_files       : (Nbands,) paths to source FITS
    exptime          : (Nbands,) exposure time (s)
    naxis1, naxis2   : scalars (X, Y) of saved array
    dtype            : scalar str
    quant_meta_json  : scalar str (JSON with quantization details)

Config (config.yaml):
paths:
  raw_dir: ./data/raw
  manifest: ./data/raw_manifest.csv
  np32_dir: ./data/np32
  np16_dir: ./data/np16
  np8_dir:  ./data/np8
wavelengths: [94, 131, 171, 193, 211, 335]
cadence_s: 12
numpy:
  quantize_bits: 32          # 32, 16, or 8
  quantize_strategy: p99     # 'p99' or 'max' (only for 16/8)
pipeline:
  use_manifest: true
  require_all_wavelengths: true
  overwrite_npz: false
"""

import os
import sys
import csv
import json
import warnings
from pathlib import Path
from datetime import timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from astropy.io.fits.verify import VerifyWarning
from sunpy.map import Map
from tqdm import tqdm
import yaml

warnings.simplefilter("ignore", VerifyWarning)

# ---------------- CONFIG ----------------
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

P = cfg["paths"]
RAW_DIR = Path(P["raw_dir"]).resolve()
MANIFEST_PATH = Path(P["manifest"]).resolve()

NPP = cfg["numpy"]
BITS = int(NPP["quantize_bits"])            # 32 / 16 / 8
STRAT = NPP.get("quantize_strategy", NPP.get("quant_strategy", "p99"))  # 'p99' or 'max'

# Choose flat output dir by bit-depth
NP_OUT = {
    32: Path(P["np32_dir"]),
    16: Path(P["np16_dir"]),
     8: Path(P["np8_dir"]),
}[BITS]
NP_OUT = NP_OUT.resolve()

PIPE = cfg.get("pipeline", {})
USE_MANIFEST   = bool(PIPE.get("use_manifest", True))
REQUIRE_ALL    = bool(PIPE.get("require_all_wavelengths", True))
OVERWRITE_NPZ  = bool(PIPE.get("overwrite_npz", False))

BANDS     = [int(w) for w in cfg["wavelengths"]]
CADENCE_S = int(cfg["cadence_s"])

# Concurrency
MAX_WORKERS = max(1, (os.cpu_count() or 4) - 1)
# ----------------------------------------


def bucket_time_str(astropy_time, step=CADENCE_S):
    """Return 'HH_MM_SS' string for the bucket floor at given cadence."""
    dt = astropy_time.datetime.replace(tzinfo=timezone.utc)
    sec = dt.hour * 3600 + dt.minute * 60 + dt.second
    b = (sec // step) * step
    hh, mm, ss = b // 3600, (b % 3600) // 60, b % 60
    return f"{hh:02d}_{mm:02d}_{ss:02d}"


def key_from_time(astropy_time):
    """Return ('YYYYMMDD', 'HH_MM_SS') for the given SunPy time object."""
    dt = astropy_time.datetime.replace(tzinfo=timezone.utc)
    dstr = f"{dt.year:04d}{dt.month:02d}{dt.day:02d}"
    return dstr, bucket_time_str(astropy_time, CADENCE_S)


def discover_raw():
    """Fallback: scan RAW_DIR for FITS and group by (date, bucket) and wavelength."""
    files = list(RAW_DIR.rglob("*.fits")) + list(RAW_DIR.rglob("*.fits.fz"))
    groups = {}
    for fp in tqdm(files, desc="Indexing raw (scan)"):
        try:
            m = Map(str(fp))
            wl = int(m.meta.get("wavelnth", m.meta.get("WAVELNTH", 0)))
            if wl not in BANDS:
                continue
            d8, tstr = key_from_time(m.date)
            key = (d8, tstr)
            d = groups.setdefault(key, {})
            # keep first seen per (key, wl); or prefer closer to bucket floor later if needed
            d.setdefault(wl, []).append((fp.resolve(), wl, m.date, m.data.shape))
        except Exception:
            continue
    return groups


def build_groups_from_manifest(manifest_path: Path):
    """
    Read manifest CSV and group available files by (YYYYMMDD, HH_MM_SS) and wavelength.
    Manifest expected columns: timestamp_utc, bucket_12s, wavelength, filepath
    """
    if not manifest_path.exists():
        return {}

    groups = {}
    with open(manifest_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                wl = int(row["wavelength"])
                if wl not in BANDS:
                    continue
                fp = Path(row["filepath"]).resolve()
                if not fp.exists():
                    continue
                ts = row["timestamp_utc"]
                d8 = ts.replace("-", "")[:8]  # YYYYMMDD
                tstr = row.get("bucket_12s") or ts[11:19].replace(":", "_")  # fallback from ts
                key = (d8, tstr)
                d = groups.setdefault(key, {})
                d.setdefault(wl, []).append((fp, wl, ts, None))
            except Exception:
                continue
    return groups


def choose_best_per_wavelength(entries):
    """
    From a list of candidates (fp, wl, time, shape) for one wavelength within a bucket,
    pick one. Strategy: choose the first (manifest is already bucketed); for scan mode
    we could prefer closest to bucket floor—kept simple here.
    """
    return entries[0] if entries else None


def choose_groups():
    """Return grouped dict: { (YYYYMMDD, HH_MM_SS): {wl: Path, ...}, ... }"""
    raw_groups = {}
    if USE_MANIFEST:
        mgroups = build_groups_from_manifest(MANIFEST_PATH)
        if mgroups:
            print(f"[INFO] Using manifest {MANIFEST_PATH} (bucket groups={len(mgroups)})")
            raw_groups = mgroups
        else:
            print("[WARN] Manifest missing/empty; falling back to raw scan.")
    if not raw_groups:
        print(f"[INFO] Scanning {RAW_DIR} (this may take a bit) …")
        raw_groups = discover_raw()

    # Reduce to single file per wavelength per bucket
    reduced = {}
    for key, per_wl in raw_groups.items():
        r = {}
        for wl, entries in per_wl.items():
            best = choose_best_per_wavelength(entries)
            if best:
                r[wl] = best[0]  # Path
        reduced[key] = r
    return reduced


def npz_path_for_key(root: Path, d8: str, tstr: str) -> Path:
    """Flat output filename: YYYYMMDD_HHMMSS.npz under root."""
    return root / f"{d8}_{tstr}.npz"


def quantize_float32(bands_f32: np.ndarray):
    return bands_f32.astype(np.float32, copy=False), {
        "bits": 32,
        "strategy": "none",
        "scale_per_band": None,
        "offset_per_band": None,
    }


def quantize_uint(bands_f32: np.ndarray, bits: int, strategy: str):
    assert bits in (16, 8)
    max_val = (1 << bits) - 1
    q_list, scales, offsets = [], [], []
    for b in range(bands_f32.shape[0]):
        arr = np.asarray(bands_f32[b], dtype=np.float32)
        arr = np.where(np.isfinite(arr), arr, 0.0)
        arr = np.clip(arr, 0, None)
        if strategy == "p99":
            hi = float(np.nanpercentile(arr, 99.0)) if np.any(arr) else 1.0
        elif strategy == "max":
            hi = float(np.nanmax(arr)) if np.any(arr) else 1.0
        else:
            raise ValueError("quantize_strategy must be 'p99' or 'max'")
        if hi <= 0:
            hi = 1.0
        scale = hi / max_val
        q = np.clip(np.round(arr / scale), 0, max_val).astype(np.uint16 if bits == 16 else np.uint8)
        q_list.append(q)
        scales.append(scale)
        offsets.append(0.0)
    bands_q = np.stack(q_list, axis=0)
    return bands_q, {
        "bits": bits,
        "strategy": strategy,
        "scale_per_band": scales,
        "offset_per_band": offsets,
    }


def read_maps_in_order(filemap: dict):
    """Return (maps_in_order, wavelengths_in_order). Order follows BANDS config."""
    maps = []
    wls = []
    for wl in BANDS:
        if wl in filemap:
            try:
                maps.append(Map(str(filemap[wl])))
                wls.append(wl)
            except Exception:
                # unreadable band; skip it
                pass
    return maps, wls


def build_stack_for_key(d8: str, tstr: str, filemap: dict, out_root: Path):
    out_npz = npz_path_for_key(out_root, d8, tstr)
    if out_npz.exists() and not OVERWRITE_NPZ:
        return "skip"

    # Read maps (in configured band order)
    maps, wls = read_maps_in_order(filemap)
    if not maps:
        return "no_bands"

    if REQUIRE_ALL and (len(wls) != len(BANDS)):
        return "incomplete_bands"

    # Shapes might differ slightly; crop to minimal common shape
    shapes = [m.data.shape for m in maps]
    min_y = min(s[0] for s in shapes)
    min_x = min(s[1] for s in shapes)

    # Stack arrays (crop to (min_y, min_x))
    arrs = [m.data[:min_y, :min_x].astype(np.float32, copy=False) for m in maps]
    bands_f32 = np.stack(arrs, axis=0)  # (Nb, Y, X)

    # Quantize
    if BITS == 32:
        bands_out, qmeta = quantize_float32(bands_f32)
        dtype_note = "float32"
    elif BITS in (16, 8):
        bands_out, qmeta = quantize_uint(bands_f32, BITS, STRAT)
        dtype_note = f"uint{BITS}"
    else:
        raise ValueError("quantize_bits must be 32, 16, or 8")

    # Metadata
    rep_ts = str(maps[0].date.isot)  # representative (first band)
    fits_paths = [str(filemap[wl].resolve()) for wl in wls]
    exptime = [float(m.meta.get("exptime", m.meta.get("EXPTIME", 0.0))) for m in maps]

    # Save to flat NPZ
    np.savez_compressed(
        out_npz,
        bands=bands_out,
        wavelengths=np.asarray(wls, dtype=np.int16),
        ts_utc=np.asarray(rep_ts),
        bucket_12s=np.asarray(tstr),
        fits_files=np.asarray(fits_paths, dtype="U"),
        exptime=np.asarray(exptime, dtype=np.float32),
        naxis1=np.asarray(min_x, dtype=np.int32),
        naxis2=np.asarray(min_y, dtype=np.int32),
        dtype=np.asarray(dtype_note),
        quant_meta_json=json.dumps(qmeta, ensure_ascii=False),
    )
    return "written"


def main():
    NP_OUT.mkdir(parents=True, exist_ok=True)

    groups = choose_groups()
    if not groups:
        print("[INFO] No raw FITS found (or none matched wavelengths).")
        print(f"RAW_DIR: {RAW_DIR}")
        return

    # Filter completeness if required
    complete = {}
    for key, fmap in groups.items():
        if REQUIRE_ALL:
            if all(wl in fmap for wl in BANDS):
                complete[key] = fmap
        else:
            if any(wl in fmap for wl in BANDS):
                complete[key] = fmap

    total = len(groups)
    keys = sorted(complete.keys())
    if not keys:
        print(f"[INFO] No timestamps matched the requirement (require_all={REQUIRE_ALL}).")
        return

    # Skip those already present (unless overwrite)
    todo = []
    for (d8, tstr) in keys:
        out_npz = npz_path_for_key(NP_OUT, d8, tstr)
        if OVERWRITE_NPZ or (not out_npz.exists()):
            todo.append((d8, tstr))

    print(f"Output dir (flat): {NP_OUT}")
    print(f"Quantization: {BITS} bits (strategy={STRAT if BITS in (16,8) else 'none'})")
    print(f"Timestamps: total groups={total} | eligible={len(keys)} | to write={len(todo)} | require_all={REQUIRE_ALL}")
    print(f"Workers: {MAX_WORKERS}")

    written = skipped = failed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {
            ex.submit(build_stack_for_key, d8, tstr, complete[(d8, tstr)], NP_OUT): (d8, tstr)
            for (d8, tstr) in todo
        }
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Building NP stacks"):
            d8, tstr = futs[fut]
            try:
                res = fut.result()
                if res == "written":
                    written += 1
                elif res == "skip":
                    skipped += 1
                else:
                    failed += 1
                    print(f"[WARN] {d8}_{tstr}: {res}")
            except Exception as e:
                failed += 1
                print(f"[WARN] {d8}_{tstr}: exception {e}")

    print("\n=== Summary ===")
    print(f"Wrote:   {written}")
    print(f"Skipped: {skipped}")
    print(f"Failed:  {failed}")
    print(f"Flat NPZ folder: {NP_OUT}")


if __name__ == "__main__":
    main()
