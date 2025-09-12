#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import hashlib
from pathlib import Path
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import astropy.units as u
from sunpy.net import Fido, attrs as a
from sunpy.map import Map
from parfive import Downloader
from tqdm import tqdm

# -------- CONFIG --------
import yaml

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

# paths
P = cfg["paths"]
RAW_DIR = Path(P["raw_dir"])
MANIFEST_PATH = Path(P["manifest"])

# If config provides a path template, use it; else default to raw/{file}
# (SunPy requires the literal "{file}" token in the path string)
PATH_TEMPLATE = P.get("raw_path_template", None)
if PATH_TEMPLATE is None:
    PATH_TEMPLATE = str(RAW_DIR / "{file}")

TIME_START = cfg["time_start"]
TIME_END   = cfg["time_end"]
CADENCE_S  = cfg["cadence_s"]
WAVELENGTHS = cfg["wavelengths"]

N = cfg["network"]
WINDOW_MINUTES       = N["window_minutes"]
PARALLEL_CONNECTIONS = N["parallel_connections"]
FILE_TIMEOUT_S       = N["file_timeout_s"]
RETRY_ATTEMPTS       = N["retry_attempts"]
BANDS_IN_PARALLEL    = int(N.get("bands_in_parallel", min(len(WAVELENGTHS), PARALLEL_CONNECTIONS)))

OUT_DIR = RAW_DIR     # downloader target
# ------------------------

def ensure_outdir():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Ensure SunPy puts files under OUT_DIR
    os.environ["SUNPY_DOWNLOAD_DIR"] = str(OUT_DIR.resolve())

def make_downloader():
    """Create a fresh Downloader for a thread (band)."""
    try:
        dl = Downloader(
            max_conn=PARALLEL_CONNECTIONS,
            progress=False,           # we'll show our own per-band progress
            retry_attempts=RETRY_ATTEMPTS,
            file_timeout=FILE_TIMEOUT_S,
        )
    except TypeError:
        # Compatibility with older parfive versions
        dl = Downloader(max_conn=PARALLEL_CONNECTIONS, progress=False)
        dl.retry_attempts = RETRY_ATTEMPTS
        dl.file_timeout = FILE_TIMEOUT_S
    return dl

def time_slices(start_str, end_str, minutes=10):
    tz = timezone.utc
    t0 = datetime.fromisoformat(start_str.replace(" ", "T")).replace(tzinfo=tz)
    t1 = datetime.fromisoformat(end_str.replace(" ", "T")).replace(tzinfo=tz)
    cur = t0
    step = timedelta(minutes=minutes)
    while cur < t1:
        nxt = min(cur + step, t1)
        yield cur, nxt
        cur = nxt

def download_band(wl: int) -> int:
    """Download one wavelength across all time windows with its own Downloader."""
    dl = make_downloader()
    saved = 0
    for t0, t1 in time_slices(TIME_START, TIME_END, WINDOW_MINUTES):
        q = Fido.search(
            a.Time(t0.strftime("%Y-%m-%d %H:%M:%S"),
                   t1.strftime("%Y-%m-%d %H:%M:%S")),
            a.Instrument("AIA"),
            a.Wavelength(wl * u.angstrom),
            a.Sample(CADENCE_S * u.second),
        )
        if len(q) == 0:
            continue
        try:
            # include {file} in path so files land under OUT_DIR
            results = Fido.fetch(q, downloader=dl, path=PATH_TEMPLATE)
            saved += len(results)
        except Exception as e:
            print(f"[WARN] fetch {wl} Å {t0}–{t1}: {e}")
    # finalize downloader (ensures any async tasks are finished in some parfive versions)
    try:
        _ = dl.results()
    except Exception:
        pass
    return saved

def bucket_time_str(sunpy_time, step=12):
    dt = sunpy_time.datetime.replace(tzinfo=timezone.utc)
    sec = dt.hour*3600 + dt.minute*60 + dt.second
    b = (sec // step) * step   # floor bucket to avoid collisions
    return f"{b//3600:02d}_{(b%3600)//60:02d}_{b%60:02d}"

def md5sum(path: Path, chunk=1024*1024) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()

def build_manifest():
    """Scan OUT_DIR and write data/raw_manifest.csv."""
    files = list(OUT_DIR.glob("*.fits")) + list(OUT_DIR.glob("*.fits.fz"))
    if not files:
        print(f"[INFO] No FITS found in {OUT_DIR.resolve()}, manifest will be empty.")
        # still create an empty file with header
        with open(MANIFEST_PATH, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "timestamp_utc", "bucket_12s", "wavelength", "filepath",
                "filesize_bytes", "hash_md5", "exptime", "naxis1", "naxis2"
            ])
            w.writeheader()
        print(f"Wrote empty manifest: {MANIFEST_PATH}")
        return

    rows = []
    for fp in tqdm(files, desc="Building manifest"):
        try:
            m = Map(str(fp))
            wl = int(m.meta.get("wavelnth", m.meta.get("WAVELNTH", -1)))
            if wl not in WAVELENGTHS:
                continue

            ts_iso = m.date.isot
            bucket = bucket_time_str(m.date, CADENCE_S)
            exptime = float(m.meta.get("exptime", m.meta.get("EXPTIME", 0.0)))
            nax1 = int(m.meta.get("naxis1", m.meta.get("NAXIS1", 0)))
            nax2 = int(m.meta.get("naxis2", m.meta.get("NAXIS2", 0)))
            sizeb = fp.stat().st_size

            try:
                relpath = str(fp.resolve().relative_to(Path.cwd()))
            except Exception:
                relpath = str(fp)

            digest = md5sum(fp)

            rows.append({
                "timestamp_utc": ts_iso,
                "bucket_12s": bucket,
                "wavelength": wl,
                "filepath": relpath,
                "filesize_bytes": sizeb,
                "hash_md5": digest,
                "exptime": exptime,
                "naxis1": nax1,
                "naxis2": nax2,
            })
        except Exception as e:
            print(f"[WARN] Skipping {fp.name}: {e}")
            continue

    rows.sort(key=lambda r: (r["timestamp_utc"], r["wavelength"]))

    with open(MANIFEST_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote manifest: {MANIFEST_PATH}  (rows: {len(rows)})")

def main():
    ensure_outdir()

    total_expected = int(
        (datetime.fromisoformat(TIME_END.replace(" ", "T")) -
         datetime.fromisoformat(TIME_START.replace(" ", "T"))
        ).total_seconds() // CADENCE_S
    )

    print(f"Saving raw FITS to: {OUT_DIR.resolve()}")
    print(f"Window: {TIME_START} → {TIME_END} | Cadence: {CADENCE_S}s | Expected per band: ≤{total_expected}")
    print(f"Path template: {PATH_TEMPLATE}")
    print(f"Parallel: {BANDS_IN_PARALLEL} bands × {PARALLEL_CONNECTIONS} connections each")

    per_band = {wl: 0 for wl in WAVELENGTHS}

    # Run bands in parallel, each with its own downloader
    with ThreadPoolExecutor(max_workers=BANDS_IN_PARALLEL) as ex:
        futures = {ex.submit(download_band, wl): wl for wl in WAVELENGTHS}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Bands (parallel)"):
            wl = futures[fut]
            try:
                per_band[wl] = int(fut.result())
            except Exception as e:
                print(f"[WARN] Band {wl} Å failed: {e}")

    print("\n=== Download Summary ===")
    for wl in sorted(per_band):
        print(f"{wl:>3} Å: saved {per_band[wl]} files (expected ≤{total_expected}; gaps/timeouts possible)")
    print(f"Raw folder: {OUT_DIR.resolve()}")

    # Build manifest from what's present on disk now
    build_manifest()

if __name__ == "__main__":
    main()
