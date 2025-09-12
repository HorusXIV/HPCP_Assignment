#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import cv2
import numpy as np
from pathlib import Path
from sunpy.map import Map
from tqdm import tqdm
import yaml
import matplotlib.cm as cm  # fallback for older Matplotlib

# Colormap compat
try:
    import matplotlib.colormaps as cmaps
    COLORMAP = cmaps["magma"]         # >= 3.6
except Exception:
    COLORMAP = cm.get_cmap("magma")   # < 3.6

# -------- CONFIG --------
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

P = cfg["paths"]
RAW_DIR = Path(P["raw_dir"])
MANIFEST_PATH = Path(P["manifest"])
ROOT = Path(P.get("root", "data"))

WAVELENGTH = 211
FPS = 10
TARGET_W = TARGET_H = 1024

OUT_MP4 = ROOT / "preview_211A_1024.mp4"   # preferred (mp4v or imageio)
OUT_AVI = ROOT / "preview_211A_1024.avi"   # fallback for MJPG

P_LO, P_HI = 1.0, 99.0
SHOW_TIMESTAMP = True
# ------------------------


def load_manifest_rows(manifest_path: Path, wl: int):
    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found: {manifest_path}")
        return []
    rows = []
    with open(manifest_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                if int(row["wavelength"]) != wl:
                    continue
                fp = Path(row["filepath"])
                if not fp.exists():
                    fp = RAW_DIR / fp.name
                if not fp.exists():
                    continue
                rows.append((row["timestamp_utc"], fp))
            except Exception:
                continue
    rows.sort(key=lambda x: x[0])
    return rows


def normalize_to_uint8(arr: np.ndarray, p_lo=P_LO, p_hi=P_HI) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(a, (p_lo, p_hi))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(a.min(initial=0.0)), float(a.max(initial=1.0))
        if hi <= lo:
            hi = lo + 1.0
    norm = np.clip((a - lo) / (hi - lo), 0, 1)
    return (norm * 255).astype(np.uint8)


def apply_colormap(u8: np.ndarray) -> np.ndarray:
    rgb = (COLORMAP(u8 / 255.0)[..., :3] * 255).astype(np.uint8)
    return rgb


def put_timestamp_bgr(img_bgr: np.ndarray, text: str) -> None:
    org = (20, 40)
    cv2.putText(img_bgr, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img_bgr, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1, cv2.LINE_AA)


class Writer:
    """
    Resilient writer:
      - try OpenCV mp4v → .mp4
      - else try OpenCV MJPG → .avi
      - else use imageio (FFmpeg) H.264 → .mp4
    """
    def __init__(self, w, h, fps):
        self.w = w
        self.h = h
        self.fps = fps
        self.backend = None
        self.writer = None

        # 1) Try OpenCV mp4v
        OUT_MP4.parent.mkdir(parents=True, exist_ok=True)
        fourcc_mp4v = cv2.VideoWriter_fourcc(*"mp4v")
        w1 = cv2.VideoWriter(str(OUT_MP4), fourcc_mp4v, fps, (w, h), isColor=True)
        if w1.isOpened():
            self.backend = "opencv_mp4v"
            self.writer = w1
            self.path = OUT_MP4
            return

        # 2) Try OpenCV MJPG (AVI)
        OUT_AVI.parent.mkdir(parents=True, exist_ok=True)
        fourcc_mjpg = cv2.VideoWriter_fourcc(*"MJPG")
        w2 = cv2.VideoWriter(str(OUT_AVI), fourcc_mjpg, fps, (w, h), isColor=True)
        if w2.isOpened():
            self.backend = "opencv_mjpg"
            self.writer = w2
            self.path = OUT_AVI
            return

        # 3) ImageIO (FFmpeg) H.264
        try:
            import imageio.v3 as iio
            self.backend = "imageio_ffmpeg"
            self.iio = iio
            self.path = OUT_MP4
        except Exception as e:
            raise RuntimeError("No working video backend (OpenCV mp4v/MJPG nor imageio-ffmpeg).") from e

    def write(self, bgr_frame):
        if self.backend.startswith("opencv"):
            self.writer.write(bgr_frame)
        else:
            # imageio expects RGB
            if not hasattr(self, "_opened"):
                self.path.parent.mkdir(parents=True, exist_ok=True)
                # Use yuv420p for wide compatibility; crf ~23 default
                self._writer = self.iio.imopen(
                    self.path, "w",
                    plugin="pyav",                        # imageio-ffmpeg backend
                    fps=self.fps,
                    codec="libx264",
                    bitrate=None,
                    pix_fmt="yuv420p",
                )
                self._opened = True
            rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            self._writer.write(rgb)

    def release(self):
        if self.backend.startswith("opencv"):
            self.writer.release()
        else:
            if getattr(self, "_opened", False):
                self._writer.close()


def main():
    rows = load_manifest_rows(MANIFEST_PATH, WAVELENGTH)
    if not rows:
        print(f"[INFO] No rows for {WAVELENGTH} Å in manifest.")
        return

    # Prime dimensions from first frame
    ts0, fp0 = rows[0]
    m0 = Map(str(fp0))
    u8 = normalize_to_uint8(m0.data)
    rgb = apply_colormap(u8)
    # Downscale to 1024x1024
    rgb_small = cv2.resize(rgb, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
    h, w = rgb_small.shape[:2]

    writer = Writer(w, h, FPS)
    print(f"[INFO] Writing with backend: {writer.backend} → {writer.path.name}")

    # Write first frame
    bgr0 = cv2.cvtColor(rgb_small, cv2.COLOR_RGB2BGR)
    if SHOW_TIMESTAMP:
        ts_text = str(m0.date.isot) if hasattr(m0, "date") else ts0
        put_timestamp_bgr(bgr0, f"AIA 211 Å  {ts_text}")
    writer.write(bgr0)

    # Remaining frames
    for ts_iso, fp in tqdm(rows[1:], desc=f"211 Å → {writer.path.name}"):
        try:
            m = Map(str(fp))
            u8 = normalize_to_uint8(m.data)
            rgb = apply_colormap(u8)
            rgb_small = cv2.resize(rgb, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
            bgr = cv2.cvtColor(rgb_small, cv2.COLOR_RGB2BGR)
            if SHOW_TIMESTAMP:
                ts_text = str(m.date.isot) if hasattr(m, "date") else ts_iso
                put_timestamp_bgr(bgr, f"AIA 211 Å  {ts_text}")
            writer.write(bgr)
        except Exception as e:
            print(f"[WARN] Skipping {fp}: {e}")

    writer.release()
    print(f"[INFO] Saved: {writer.path.resolve()}  ({w}x{h}, {FPS} fps, frames={len(rows)})")


if __name__ == "__main__":
    main()
