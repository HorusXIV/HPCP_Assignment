#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build an HDF5 array from raw SDO/AIA FITS.

- Prefers data/raw_manifest.csv (fast); falls back to scanning data/raw/ (slower).
- Groups frames into 12 s buckets; requires complete 6-band sets (94,131,171,193,211,335 Å).
- Appends per-timestamp stacks (6, 4096, 4096) into a single HDF5 dataset:
    data/hdf5/aia_raw_6bands.h5
    dataset name: "aia" with shape (time, band, y, x)
- Resumable: keeps a "done" list (date/tstr keys) in HDF5 attrs and skips existing frames.

Deps: sunpy, astropy, numpy, h5py, tqdm
"""

import sys
import csv
import json
import warnings
from pathlib import Path
from datetime import timezone, datetime
import yaml

import numpy as np
import h5py
from astropy.io.fits.verify import VerifyWarning
from sunpy.map import Map
from tqdm import tqdm

warnings.simplefilter("ignore", VerifyWarning)

# ---------------- CONFIG ----------------

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

P = cfg["paths"]
RAW_DIR = Path(P["raw_dir"])
MANIFEST_PATH = Path(P["manifest"])

H5 = cfg["hdf5"]
HDF5_DIR = Path(P["hdf5_dir"])
HDF5_PATH = HDF5_DIR / H5["file_name"]
DATASET = H5["dataset"]
CHUNKS = tuple(H5["chunks"])
COMP = H5["compression"]
COMP_LEVEL = H5["compression_level"] if COMP == "gzip" else None

BANDS = cfg["wavelengths"]
CADENCE_S = cfg["cadence_s"]

PIPE = cfg.get("pipeline", {})
USE_MANIFEST = bool(PIPE.get("use_manifest", True))
# ----------------------------------------


def bucket_time_str(sunpy_time, step=CADENCE_S):
    dt = sunpy_time.datetime.replace(tzinfo=timezone.utc)
    sec = dt.hour * 3600 + dt.minute * 60 + dt.second
    b = (sec // step) * step
    hh, mm, ss = b // 3600, (b % 3600) // 60, b % 60
    return f"{hh:02d}_{mm:02d}_{ss:02d}"


def discover_raw():
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
            d = groups.setdefault((date_str, tstr), {})
            if wl not in d:
                d[wl] = fp
        except Exception:
            continue
    return groups


def build_groups_from_manifest(manifest_path: Path):
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
                date_str = row["timestamp_utc"][:10]
                tstr = row["bucket_12s"]
                d = groups.setdefault((date_str, tstr), {})
                if wl not in d:
                    d[wl] = fp
            except Exception:
                continue
    return groups


def choose_groups():
    if USE_MANIFEST:
        groups = build_groups_from_manifest(MANIFEST_PATH)
        if groups:
            print(f"[INFO] Using manifest {MANIFEST_PATH} (groups={len(groups)})")
            return groups
        print("[WARN] Manifest missing/empty; falling back to raw scan.")
    print("[INFO] Scanning data/raw/ (this may take a bit) …")
    return discover_raw()


def hdf5_open_or_create():
    HDF5_DIR.mkdir(parents=True, exist_ok=True)
    if HDF5_PATH.exists():
        f = h5py.File(HDF5_PATH, "r+")
    else:
        f = h5py.File(HDF5_PATH, "w")

    if DATASET in f:
        ds = f[DATASET]
    else:
        ds = f.create_dataset(
            DATASET,
            shape=(0, 6, 4096, 4096),
            maxshape=(None, 6, 4096, 4096),
            chunks=CHUNKS,
            dtype="f4",
            compression=COMP,
            compression_opts=COMP_LEVEL if COMP == "gzip" else None,
        )

    # initialize attrs if missing
    f.attrs.setdefault("layout", "(time, band, y, x)")
    f.attrs.setdefault("bands", BANDS)
    f.attrs.setdefault("cadence_s", CADENCE_S)
    f.attrs.setdefault("source", "AIA Level 1 (raw, uncalibrated)")
    f.attrs.setdefault("notes", "No exposure normalization; values are DN.")
    f.attrs.setdefault("created_at_utc", datetime.now(timezone.utc).isoformat())
    f.attrs.setdefault("manifest", str(MANIFEST_PATH))
    if "done" not in f.attrs:
        f.attrs["done"] = json.dumps([])
    if "timestamps" not in f.attrs:
        f.attrs["timestamps"] = json.dumps([])

    return f, ds


def frame_key(date_str, tstr):
    return f"{date_str}/{tstr}"


def append_frame_to_hdf5(date_str, tstr, filemap):
    f, ds = hdf5_open_or_create()

    done = set(json.loads(f.attrs["done"]))
    key = frame_key(date_str, tstr)
    if key in done:
        f.close()
        return "skip"

    maps = [Map(str(filemap[wl])) for wl in sorted(BANDS)]
    shapes = {m.data.shape for m in maps}
    if len(shapes) != 1:
        f.close()
        return f"shape_mismatch:{shapes}"

    bands_f32 = np.stack([m.data.astype(np.float32) for m in maps], axis=0)

    t = ds.shape[0]
    ds.resize(t + 1, axis=0)
    ds[t, :, :, :] = bands_f32

    timestamps = json.loads(f.attrs["timestamps"])
    timestamps.append(str(maps[0].date))
    done.add(key)
    f.attrs["timestamps"] = json.dumps(timestamps)
    f.attrs["done"] = json.dumps(sorted(done))

    f.close()
    return "written"


def main():
    if not RAW_DIR.exists():
        print(f"[ERROR] RAW_DIR not found: {RAW_DIR.resolve()}")
        sys.exit(1)

    groups = choose_groups()
    if not groups:
        print("[INFO] No raw FITS found.")
        return

    complete = {k: v for k, v in groups.items() if all(wl in v for wl in BANDS)}
    incomplete = len(groups) - len(complete)
    keys = sorted(complete.keys())

    # Initialize
    _f, _ds = hdf5_open_or_create()
    print(f"HDF5 store: {HDF5_PATH.resolve()}  dataset='{DATASET}' chunks={CHUNKS}")
    print(f"Timestamps: total={len(groups)} | complete={len(complete)} | incomplete(discarded)={incomplete}")
    _f.close()

    written = 0
    skipped = 0
    failed = 0

    for (d, t) in tqdm(keys, desc="Appending to HDF5"):
        try:
            res = append_frame_to_hdf5(d, t, complete[(d, t)])
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

    with h5py.File(HDF5_PATH, "r") as f:
        ds = f[DATASET]
        print("\n=== Summary ===")
        print(f"Appended (written): {written}")
        print(f"Skipped (already done): {skipped}")
        print(f"Failed: {failed}")
        print(f"Discarded (incomplete): {incomplete}")
        print(f"Final dataset shape: {ds.shape}  chunks: {ds.chunks}")
        print(f"HDF5: {HDF5_PATH.resolve()}")


if __name__ == "__main__":
    main()
