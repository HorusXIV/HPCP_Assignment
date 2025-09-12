#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a Zarr v2 array from raw SDO/AIA FITS.

- Prefers data/raw_manifest.csv (fast); falls back to scanning data/raw/ (slower).
- Uses raw (un-calibrated) FITS exactly as downloaded.
- Groups frames into 12 s buckets; requires complete 6-band sets (94,131,171,193,211,335 Å).
- Appends per-timestamp stacks (6, 4096, 4096) into a single Zarr v2 dataset:
    data/zarr/aia_raw_6bands.zarr
    dataset name: "aia" with shape (time, band, y, x)
- Resumable: keeps a "done" list (date/tstr keys) in Zarr attrs and skips existing frames.

Deps: sunpy, astropy, numpy, zarr<3, numcodecs, tqdm
"""

import os
import sys
import csv
import json
import warnings
from pathlib import Path
from datetime import timezone, datetime

import numpy as np
from astropy.io.fits.verify import VerifyWarning
from sunpy.map import Map
from tqdm import tqdm

warnings.simplefilter("ignore", VerifyWarning)

# ---------------- CONFIG ----------------
import yaml

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

P = cfg["paths"]
RAW_DIR = Path(P["raw_dir"])
MANIFEST_PATH = Path(P["manifest"])

Z = cfg["zarr"]
ZARR_DIR = Path(P["zarr_dir"])
ZARR_PATH = ZARR_DIR / Z["store_name"]
DATASET = Z["dataset"]
CHUNKS = tuple(Z["chunks"])
COMP_CLEVEL = Z["clevel"]

PIPE = cfg.get("pipeline", {})
USE_MANIFEST = bool(PIPE.get("use_manifest", True))

BANDS = cfg["wavelengths"]
CADENCE_S = cfg["cadence_s"]
# ----------------------------------------


def bucket_time_str(sunpy_time, step=CADENCE_S):
    """Floor-bucket to HH_MM_SS on the cadence."""
    dt = sunpy_time.datetime.replace(tzinfo=timezone.utc)
    sec = dt.hour * 3600 + dt.minute * 60 + dt.second
    b = (sec // step) * step
    hh, mm, ss = b // 3600, (b % 3600) // 60, b % 60
    return f"{hh:02d}_{mm:02d}_{ss:02d}"


def discover_raw():
    """Scan RAW_DIR for *.fits / *.fits.fz, index by (date_str, bucket_12s) -> {wl:int -> Path}."""
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


def build_groups_from_manifest(manifest_path: Path):
    """Read manifest CSV into {(date_str, tstr): {wl:int -> Path}}; skip missing files."""
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
                fp = Path(row["filepath"])
                if not fp.exists():
                    continue
                date_str = row["timestamp_utc"][:10]  # YYYY-MM-DD
                tstr = row["bucket_12s"]              # HH_MM_SS
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
        groups = build_groups_from_manifest(MANIFEST_PATH)
        if groups:
            print(f"[INFO] Using manifest {MANIFEST_PATH} (groups={len(groups)})")
            return groups
        print("[WARN] Manifest missing/empty; falling back to raw scan.")
    print("[INFO] Scanning data/raw/ (this may take a bit) …")
    return discover_raw()


def zarr_open_or_create():
    """Open (or create) a Zarr v2 DirectoryStore and return (root, ds)."""
    import zarr
    from zarr.storage import DirectoryStore
    import numcodecs

    ZARR_DIR.mkdir(parents=True, exist_ok=True)
    store = DirectoryStore(str(ZARR_PATH))
    root = zarr.group(store=store, overwrite=False)

    if DATASET in root:
        ds = root[DATASET]
    else:
        compressor = numcodecs.Blosc(
            cname="zstd",
            clevel=COMP_CLEVEL,
            shuffle=numcodecs.Blosc.BITSHUFFLE,
        )
        ds = root.create_dataset(
            DATASET,
            shape=(0, 6, 4096, 4096),   # start with 0 along time axis
            chunks=CHUNKS,
            dtype="f4",
            compressor=compressor,
        )

    # Initialize attrs if missing
    root.attrs.setdefault("layout", "(time, band, y, x)")
    root.attrs.setdefault("bands", BANDS)
    root.attrs.setdefault("cadence_s", CADENCE_S)
    root.attrs.setdefault("source", "AIA Level 1 (raw, uncalibrated)")
    root.attrs.setdefault("notes", "No exposure normalization; values are DN.")
    root.attrs.setdefault("created_at_utc", datetime.now(timezone.utc).isoformat())
    root.attrs.setdefault("manifest", str(MANIFEST_PATH))
    root.attrs.setdefault("done", json.dumps([]))
    root.attrs.setdefault("timestamps", json.dumps([]))

    return root, ds


def frame_key(date_str, tstr) -> str:
    return f"{date_str}/{tstr}"


def append_frame_to_zarr(date_str, tstr, filemap):
    """Load the 6 raw maps as float32 stack and append as one time step to Zarr."""
    import zarr
    root, ds = zarr_open_or_create()

    done = set(json.loads(root.attrs["done"]))
    key = frame_key(date_str, tstr)
    if key in done:
        return "skip"

    # Load maps in band order and basic shape check
    maps = [Map(str(filemap[wl])) for wl in sorted(BANDS)]
    shapes = {m.data.shape for m in maps}
    if len(shapes) != 1:
        return f"shape_mismatch:{shapes}"

    # Assemble float32 stack (raw, uncalibrated DN)
    bands_f32 = np.stack([m.data.astype(np.float32) for m in maps], axis=0)  # (6, 4096, 4096)

    # Append along time axis
    t = ds.shape[0]
    ds.resize(t + 1, 6, 4096, 4096)
    ds[t, :, :, :] = bands_f32

    # Update attrs (resumable bookkeeping)
    timestamps = json.loads(root.attrs.get("timestamps", "[]"))
    timestamps.append(str(maps[0].date))  # ISO string for the bucket representative
    done.add(key)
    root.attrs["timestamps"] = json.dumps(timestamps)
    root.attrs["done"] = json.dumps(sorted(done))
    return "written"


def main():
    if not RAW_DIR.exists():
        print(f"[ERROR] RAW_DIR not found: {RAW_DIR.resolve()}")
        sys.exit(1)

    groups = choose_groups()
    if not groups:
        print(f"[INFO] No raw FITS found.")
        return

    # keep only complete sets
    complete = {k: v for k, v in groups.items() if all(wl in v for wl in BANDS)}
    incomplete = len(groups) - len(complete)
    keys = sorted(complete.keys())

    # Initialize store (creates if not exists)
    _root, _ds = zarr_open_or_create()
    print(f"Zarr v2 store: {ZARR_PATH.resolve()}  dataset='{DATASET}' chunks={CHUNKS}")
    print(f"Timestamps: total={len(groups)} | complete={len(complete)} | incomplete(discarded)={incomplete}")

    written = 0
    skipped = 0
    failed = 0

    for (d, t) in tqdm(keys, desc="Appending to Zarr"):
        try:
            res = append_frame_to_zarr(d, t, complete[(d, t)])
            if res == "written":
                written += 1
            elif res == "skip":
                skipped += 1
            else:
                failed += 1
                print(f"[WARN] {d} {t}: {res}")
        except Exception as e:
            failed += 1
            print(f"[WARN] {d} {t}: exception {e}")

    # Final shape
    import zarr
    from zarr.storage import DirectoryStore
    root = zarr.group(store=DirectoryStore(str(ZARR_PATH)))
    ds = root[DATASET]

    print("\n=== Summary ===")
    print(f"Appended (written): {written}")
    print(f"Skipped (already done): {skipped}")
    print(f"Failed: {failed}")
    print(f"Discarded (incomplete): {incomplete}")
    print(f"Final dataset shape: {ds.shape}  chunks: {ds.chunks}")
    print(f"Zarr: {ZARR_PATH.resolve()}")


if __name__ == "__main__":
    main()
