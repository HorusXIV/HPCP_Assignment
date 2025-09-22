#!/usr/bin/env python3
"""
Fetch and unpack np32.zip into data/np32/.

Usage (from repo root):
  poetry run python data/fetch_np32.py
  poetry run python data/fetch_np32.py --force
  poetry run python data/fetch_np32.py --url https://.../np32.zip --sha256 <hex>

Env:
  ZENODO_TOKEN   : optional bearer token for restricted records
  NP32_URL       : override download URL
  NP32_SHA256    : override expected sha256
  NP32_META      : path to meta JSON (default: data/np32.meta.json)
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen, Request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
NP32_DIR = DATA_DIR / "np32"
ZIP_PATH = DATA_DIR / "np32.zip"
DEFAULT_META = DATA_DIR / "np32.meta.json"


def _hash_file(path: Path, algo: str = "sha256", chunk: int = 1024 * 1024) -> str:
    h = getattr(hashlib, algo)()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _human(n: int | None) -> str:
    if n is None:
        return "unknown"
    units = ["B","KB","MB","GB","TB"]
    v = float(n)
    for u in units:
        if v < 1024 or u == units[-1]:
            return f"{v:.1f} {u}"
        v /= 1024
    return f"{v:.1f} PB"


def download(url: str, dest: Path, token: str | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "hpcp-fetch/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = Request(url, headers=headers)
    with urlopen(req) as r, dest.open("wb") as f:
        total = getattr(r, "length", None)
        print(f"Downloading -> {dest} ({_human(total)})")
        read = 0
        chunk = 1024 * 1024
        while True:
            buf = r.read(chunk)
            if not buf:
                break
            f.write(buf)
            read += len(buf)
            if total:
                pct = 100.0 * read / total
                print(f"\r  {_human(read)} / {_human(total)} ({pct:5.1f}%)", end="")
        print("\nDone.")


def extract_zip(zip_path: Path, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path.name} -> {out_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)

    # Flatten if archive contains a top-level folder
    moved = 0
    for p in out_dir.rglob("*.npz"):
        if p.parent == out_dir:
            continue
        target = out_dir / p.name
        if not target.exists():
            p.replace(target)
            moved += 1
    if moved:
        # best-effort cleanup of now-empty dirs
        for d in sorted({p.parent for p in out_dir.rglob("*") if p.is_file()},
                        key=lambda d: len(d.parts), reverse=True):
            try:
                if d != out_dir:
                    d.rmdir()
            except OSError:
                pass

    count = len(list(out_dir.glob("*.npz")))
    print(f"Found {count} .npz files in {out_dir}")
    return count


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("NP32_URL"), help="Override download URL")
    ap.add_argument("--sha256", default=os.environ.get("NP32_SHA256"), help="Override SHA256 (hex)")
    ap.add_argument("--meta", default=os.environ.get("NP32_META", str(DEFAULT_META)), help="Path to meta JSON")
    ap.add_argument("--force", action="store_true", help="Re-download even if data/np32 has files")
    ap.add_argument("--keep-zip", action="store_true", help="Keep data/np32.zip after extraction")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    # Load meta
    meta_path = Path(args.meta)
    if not meta_path.exists():
        print(f"Meta file not found: {meta_path}. Provide --url/--sha256 manually.", file=sys.stderr)
        meta = {}
    else:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    url = args.url or meta.get("urls", {}).get("file_download")
    if not url:
        print("Missing download URL. Use --url or set NP32_URL.", file=sys.stderr)
        return 2

    sha256 = args.sha256 or (meta.get("checksums", {}).get("sha256") if meta.get("checksums") else None)
    md5 = meta.get("checksums", {}).get("md5") if meta.get("checksums") else None

    existing = list(NP32_DIR.glob("*.npz"))
    if existing and not args.force:
        print(f"{NP32_DIR} already contains {len(existing)} .npz files. Use --force to re-download.")
        return 0

    token = os.environ.get("ZENODO_TOKEN")

    # Download
    download(url, ZIP_PATH, token=token)

    # Verify integrity
    if sha256:
        print("Verifying SHA256…")
        got = _hash_file(ZIP_PATH, "sha256")
        if got.lower() != sha256.lower():
            print(f"SHA256 mismatch!\n  expected: {sha256}\n  got     : {got}", file=sys.stderr)
            return 1
        print("SHA256 OK.")
    elif md5:
        print("Verifying MD5… (consider adding SHA256)")
        got = _hash_file(ZIP_PATH, "md5")
        if got.lower() != md5.lower():
            print(f"MD5 mismatch!\n  expected: {md5}\n  got     : {got}", file=sys.stderr)
            return 1
        print("MD5 OK.")
    else:
        print("No checksum provided. Proceeding without verification.")

    # Extract
    n = extract_zip(ZIP_PATH, NP32_DIR)

    # Cleanup
    if not args.keep_zip:
        try:
            ZIP_PATH.unlink()
        except FileNotFoundError:
            pass

    print("All done.")
    return 0 if n > 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
