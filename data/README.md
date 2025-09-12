# Data Pipeline – SDO/AIA Flare Dataset

This folder contains all data and conversion utilities used in the HPC assignment.  
It documents how raw AIA FITS images were acquired, organized, and transformed into formats suitable for analysis.

---

## 1. Acquisition

- **Source:** [SDO/AIA](https://sdo.gsfc.nasa.gov/data/) Level 1 (uncalibrated) FITS files.
- **Download script:** [`download.py`](../download.py)  
  Uses **SunPy** + **parfive** with a manifest-driven workflow.
- **Manifest:** [`raw_manifest.csv`](raw_manifest.csv)  
  Lists all raw files with metadata (UTC timestamp, wavelength, exposure time, size, MD5 checksum).  
  This is the authoritative record of what was downloaded.

**Wavelength bands used:** 94, 131, 171, 193, 211, 335 Å (6 bands)  
**Cadence:** 12 s  
**Time range originally downloaded:** 11:30 – 12:30 UTC, 2017-09-06 (covering the X9.3 flare)

---

## 2. Dataset reduction

While a full hour was initially downloaded, it was found computationally excessive for HPC DEM inversions.

- **Representative subset kept:**  
  Ten full 6-band frames between **12:00 and 12:04** (covering the flare peak at 12:02).
- **Rationale:**  
  - One frame suffices for benchmarking/evaluation.  
  - Ten frames retained to allow short time-series experiments.  
  - The rest of the hour is discarded to save compute/storage.

---

## 3. Formats

### Primary format: **NumPy NPZ**
- Location: `np32/` (float32, simplest to load).
- File naming: flat structure, `YYYYMMDD_HHMMSS.npz`.
- Contents:
  - `bands` → array `(6, 4096, 4096)`
  - `meta` → metadata dict (datetime, wavelengths, dtype, quantization info).
- Used as **input for HPC tasks**.
- **Download prebuilt stacks:** [np32 dataset (10 frames)]( https://filesender.switch.ch/filesender2/?s=download&token=9f6c8b51-4ebf-4015-b9b9-d12a10f52459)

### Other formats (unused, kept for reference in `unused/`)
- `raw_to_fits.py` – reconstruct grouped FITS  
- `raw_to_hdf5.py` – export to HDF5  
- `raw_to_movie.py` – quicklook MP4/AVI movie of a chosen band  
- `raw_to_zarr.py` – build Zarr v2 dataset  

---

## 4. Processing pipeline

```mermaid
flowchart TD
    A[Raw FITS (SDO/AIA)] -->|download.py| B[raw/ folder]
    B -->|raw_manifest.csv| C[Manifest]
    B -->|raw_to_np.py| D[np32/ NPZ stacks]
    C -->|guides grouping| D
    D -->|used for HPC tasks| E[DEM inversion & analysis]
```

---

## 5. Workflow

1. **Download**
   ```bash
   python download.py
   ```
   Saves raw FITS under `raw/` and writes `raw_manifest.csv`.

2. **Convert raw → NPZ**
   ```bash
   python raw_to_np.py
   ```
   - Reads `raw_manifest.csv` (or scans `raw/`).  
   - Buckets FITS into 12-second frames.  
   - Requires all 6 bands → discards incomplete sets.  
   - Writes per-frame compressed NPZ.

3. **Load in analysis**
   ```python
   import numpy as np
   arr = np.load("np32/20170906_120000.npz")
   bands = arr["bands"]   # (6, 4096, 4096)
   meta  = arr["meta"].item()
   ```

---

## 6. Folder structure

```
data/
│
├── raw/                # original FITS (as downloaded)
├── raw_manifest.csv    # manifest of raw FITS
├── np32/               # main NPZ stacks (float32)
├── np16/, np8/         # alternative quantized versions (unused)
├── unused/             # scripts not part of main workflow
│   ├── raw_to_fits.py
│   ├── raw_to_hdf5.py
│   ├── raw_to_movie.py
│   └── raw_to_zarr.py
│
├── download.py         # acquisition driver
├── raw_to_np.py        # FITS → NPZ converter (main preprocessing)
└── config.yaml         # shared configuration
```

---

## 7. Visualization

For a quick overview of the flare across all six AIA bands, we provide a composite figure in this folder:

- **`data/display_frames.ipynb`** — grid of 10 frames × 6 wavelengths, aligned in time.  
  Useful for a qualitative check of data quality, flare timing, and band morphology.

Example snippet from the figure:

---

## 8. Notes of interest

- **np32 stacks** are the only format guaranteed compatible with the DEM inversion codes (`dn2dem_pos`, etc.).
- Metadata embedded in `.npz` ensures reproducibility without relying on external files.
- Scripts are resumable:
  - Download resumes missing files.
  - Conversion skips already existing stacks.
- HPC load tests showed that even one full frame `(6, 4096, 4096)` is computationally heavy (minutes to hours on CPUs).  
  This motivated the dataset reduction.
