#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert raw SDO/AIA FITS in data/raw/ to calibrated, exposure-normalized FITS in data/fits/,
processing ONLY timestamps that have a COMPLETE 6-band set (94,131,171,193,211,335 Å).

- Prefers using data/raw_manifest.csv (fast); falls back to scanning data/raw/ if missing/empty.
- Optional aiapy.register (level-1.5-like); else Map.rotate(recenter=True)
- Exposure normalization by EXPTIME
- Tile-compressed FITS (.fits.fz, Rice + subtractive dither) with clean headers
- Multithreaded (one timestamp per worker), resumable

Deps: sunpy, astropy, numpy, tqdm
Optional: aiapy
"""

import os
import sys
import csv
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timezone

import numpy as np
from astropy.io import fits
from astropy.io.fits.verify import VerifyWarning
from sunpy.map import Map
from tqdm import tqdm

warnings.simplefilter("ignore", VerifyWarning)

# ====================== CONFIG ======================
import yaml
from pathlib import Path

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

# paths
P = cfg["paths"]
RAW_DIR = Path(P["raw_dir"])
MANIFEST_PATH = Path(P["manifest"])

F = cfg["fits"]
WRITE_COMPRESSED = F["write_compressed"]
QUANTIZE_LEVEL = F["quantize_level"]
OUT_DIR = Path(cfg["paths"]["fits_compressed_dir"] if WRITE_COMPRESSED else cfg["paths"]["fits_dir"])
REQUIRE_COMPLETE = cfg["pipeline"]["require_complete_bandset"]
USE_MANIFEST = cfg["pipeline"]["use_manifest"]
STRICT_MANIFEST = cfg["pipeline"]["strict_manifest"]


BANDS = cfg["wavelengths"]
CADENCE_S = cfg["cadence_s"]

# Concurrency
MAX_WORKERS = max(1, (os.cpu_count() or 4) - 1)  # one timestamp per worker

# aiapy (optional, better registration)
_USE_AIAPY = False
try:
    from aiapy.calibrate import register as aia_register
    _USE_AIAPY = True
except Exception:
    _USE_AIAPY = False
# ====================================================


def bucket_time_str(astropy_time, step=CADENCE_S):
    """Floor-bucket to HH_MM_SS on the cadence."""
    dt = astropy_time.datetime.replace(tzinfo=timezone.utc)
    sec = dt.hour * 3600 + dt.minute * 60 + dt.second
    b = (sec // step) * step
    hh, mm, ss = b // 3600, (b % 3600) // 60, b % 60
    return f"{hh:02d}_{mm:02d}_{ss:02d}"


def discover_raw():
    """
    Scan RAW_DIR for *.fits / *.fits.fz, index by (date_str, bucket_12s).
    Returns: {(date_str, tstr): {wl:int -> Path}}
    """
    files = list(RAW_DIR.glob("*.fits")) + list(RAW_DIR.glob("*.fits.fz"))
    groups = {}
    for fp in tqdm(files, desc="Indexing raw (scan)"):
        try:
            m = Map(str(fp))
            wl = int(m.meta.get("wavelnth", m.meta.get("WAVELNTH", 0)))
            if wl not in BANDS:
                continue
            date_str = m.date.datetime.strftime("%Y-%m-%d")
            tstr = bucket_time_str(m.date)
            key = (date_str, tstr)
            d = groups.setdefault(key, {})
            if wl not in d:  # keep first occurrence per band per bucket
                d[wl] = fp
        except Exception:
            continue
    return groups


def build_groups_from_manifest(manifest_path: Path, bands_required):
    """
    Read manifest CSV into {(date_str, tstr): {wl:int -> Path}}
    Skips rows whose files don't actually exist.
    """
    if not manifest_path.exists():
        return {}
    groups = {}
    with open(manifest_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                wl = int(row["wavelength"])
                if wl not in bands_required:
                    continue
                fp = Path(row["filepath"])
                if not fp.exists():
                    continue
                date_str = row["timestamp_utc"][:10]      # YYYY-MM-DD
                tstr = row["bucket_12s"]                  # HH_MM_SS
                key = (date_str, tstr)
                d = groups.setdefault(key, {})
                if wl not in d:
                    d[wl] = fp
            except Exception:
                continue
    return groups


def choose_groups():
    """Prefer manifest; fallback to scanning FITS if manifest missing/empty."""
    if USE_MANIFEST:
        groups = build_groups_from_manifest(MANIFEST_PATH, BANDS)
        if groups:
            print(f"[INFO] Using manifest {MANIFEST_PATH} (groups={len(groups)})")
            return groups
        print("[WARN] Manifest missing/empty; falling back to raw scan.")
    print("[INFO] Scanning data/raw/ (this may take a bit) …")
    return discover_raw()


def is_timestamp_done(date_str, tstr):
    """Check if all 6 calibrated outputs already exist for this timestamp."""
    ts_dir = OUT_DIR / date_str / tstr
    if not ts_dir.exists():
        return False
    have = 0
    for wl in BANDS:
        p = ts_dir / (f"{wl:03d}A_level1p5_expnorm.fits.fz" if WRITE_COMPRESSED
                      else f"{wl:03d}A_level1p5_expnorm.fits")
        if p.exists():
            have += 1
    return have == 6


def calibrate_map(m: Map) -> Map:
    """Register + exposure-normalize; return new Map(float32)."""
    if _USE_AIAPY:
        m_cal = aia_register(m)
    else:
        m_cal = m.rotate(recenter=True)
    exptime = float(m_cal.meta.get("exptime", m_cal.meta.get("EXPTIME", 1.0)) or 1.0)
    data = (m_cal.data.astype(np.float32) / (exptime if exptime != 0 else 1.0))
    meta = dict(m_cal.meta)
    meta["EXPNORM"] = True
    return Map(data, meta)


def save_fits_compressed(m: Map, out_path: Path, quantize_level: int = QUANTIZE_LEVEL):
    """
    Write a clean tile-compressed FITS:
      - Put science/WCS meta in PrimaryHDU
      - CompImageHDU carries only compression metadata
    """
    hdr = m.fits_header.copy()
    if "BLANK" in hdr:
        del hdr["BLANK"]
    primary = fits.PrimaryHDU(header=hdr)
    comp = fits.CompImageHDU(
        data=m.data.astype("f4"),
        compression_type="RICE_1",
        quantize_method="SUBTRACTIVE_DITHER_1",
        quantize_level=int(quantize_level),
    )
    fits.HDUList([primary, comp]).writeto(str(out_path), overwrite=True)


def save_fits_uncompressed(m: Map, out_path: Path):
    m.save(str(out_path), overwrite=True)


def process_timestamp(date_str, tstr, filemap):
    """
    Calibrate & save all six bands for a timestamp.
    filemap: dict {wl:int -> Path} (guaranteed to have all 6)
    """
    ts_dir = OUT_DIR / date_str / tstr
    ts_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    for wl in BANDS:
        out_path = ts_dir / (f"{wl:03d}A_level1p5_expnorm.fits.fz"
                             if WRITE_COMPRESSED else
                             f"{wl:03d}A_level1p5_expnorm.fits")
        if out_path.exists():
            continue

        m_raw = Map(str(filemap[wl]))
        m_cal = calibrate_map(m_raw)

        if WRITE_COMPRESSED:
            save_fits_compressed(m_cal, out_path, QUANTIZE_LEVEL)
        else:
            save_fits_uncompressed(m_cal, out_path)
        written += 1

    return written


def main():
    if not RAW_DIR.exists():
        print(f"[ERROR] RAW_DIR not found: {RAW_DIR.resolve()}")
        sys.exit(1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"aiapy.register active: {_USE_AIAPY}")
    groups = choose_groups()  # {(date_str, tstr): {wl: path, ...}}
    if not groups:
        print(f"[INFO] No usable raw files found.")
        return

    # keep only complete sets (all 6 bands)
    complete = {k: v for k, v in groups.items() if all(wl in v for wl in BANDS)}
    incomplete = len(groups) - len(complete)
    keys = sorted(complete.keys())

    print(f"Found timestamps: total={len(groups)} | complete={len(complete)} | incomplete(discarded)={incomplete}")
    print(f"Output: {OUT_DIR.resolve()}  | Compressed={WRITE_COMPRESSED}  | Workers={MAX_WORKERS}")

    # Skip timestamps already fully written
    to_process = [(d, t) for (d, t) in keys if not is_timestamp_done(d, t)]
    if not to_process:
        print("[INFO] Nothing to do; all complete timestamps already written.")
        return

    written_total = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {
            ex.submit(process_timestamp, d, t, complete[(d, t)]): (d, t)
            for (d, t) in to_process
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Calibrating timestamps"):
            d, t = futures[fut]
            try:
                written_total += fut.result()
            except Exception as e:
                failed += 1
                print(f"[WARN] Failed {d} {t}: {e}")

    print("\n=== Summary ===")
    print(f"Timestamps complete (processed this run): {len(to_process) - failed}/{len(to_process)}")
    print(f"Files written this run: {written_total}")
    print(f"Failed timestamps: {failed}")
    print(f"Discarded (incomplete) timestamps: {incomplete}")
    print(f"Calibrated FITS in: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
