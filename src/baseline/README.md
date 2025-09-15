# Baseline Solver (`src/baseline`)

This directory contains the **reference DEM solver** used across the project. It wraps the vendor-provided implementation and exposes a minimal, stable Python interface that other modules (profiling, Dask tiling) rely on.

---

## Layout

```
src/
  baseline/
    vendor/
      dn2dem_pos.py        # Main solver: f,e,T_RESP,T_RESP_LOGT,TEMPS -> demmap,edemmap,elogt,chisq,dn_reg
      demmap_pos.py        # Lower-level DEM routines (used by profiling tools)
```

> The `vendor/` code is treated as a black box. We **do not** modify it; we adapt to it in thin wrappers elsewhere.

---

## Expected Interfaces

### 1) Vendor solver
- **`dn2dem_pos(f, e, T_RESP, T_RESP_LOGT, TEMPS, nmu=42)`**
  - `f`: `(H, W, 6)` float32 — DN values (per-pixel, per-band)
  - `e`: `(H, W, 6)` float32 — per-pixel DN uncertainties (e.g., `sqrt(f)+eps`)
  - `T_RESP`: `(n_logT, 6)` float32 — response matrix for the 6 bands
  - `T_RESP_LOGT`: `(n_logT,)` float32 — log10 temperature grid corresponding to rows of `T_RESP`
  - `TEMPS`: `(nt+1,)` float32 — temperature **bin edges** (Kelvin) used to discretize the DEM
  - Returns: `(demmap, edemmap, elogt, chisq, dn_reg)`
    - `demmap`: `(H, W, nt)` float32 — emissivity distribution per pixel
    - `elogt`: `(nt,)` float32 — log10 of temperature bin centers, or related abscissa
    - others are diagnostics

> **Important**: `T_RESP.shape[1]` **must match** the number of bands in the data (`6` in this assignment). The runtime error you saw — *"Tresp needs to be the same number of wavelengths/filters as the data."* — is raised when this is not the case.

### 2) Convenience wrapper (used by non-Dask code)
We provide a thin wrapper in `src/common/backend.py`:

```python
from src.common.backend import dn2dem
demmap, edemmap, logT_bins, chisq, dn_reg = dn2dem(frame_6hw, T_RESP, T_RESP_LOGT, TEMPS, nmu=42)
```

- Accepts `(6, H, W)` and internally reorders to `(H, W, 6)` and builds `e = sqrt(f) + 1e-6`.
- Ensures dtype/contiguity and clamps negative/NaNs.

The Dask path uses `src/dask/tiling.py` which calls the vendor function per tile via `_blk()`.

---

## Data & Responses

- Example data lives under `data/np32/*.npz` with key `bands` shaped `(6, H, W)`.
- Response matrices are loaded via **`src/common/dem_api.load_tresp()`** which returns a triplet:
  - `(T_RESP, T_RESP_LOGT, TEMPS)`
- If you don’t have real responses, `src/common/profiling.py` includes:
  - `prepare_synthetic_responses(...)` to generate a consistent synthetic set.

**Shape checklist** before calling the solver:
- `frame_6hw`: `(6, H, W)` (or `(H, W, 6)` in vendor form)
- `T_RESP`: `(n_logT, 6)`
- `T_RESP_LOGT`: `(n_logT,)`
- `TEMPS`: `(nt+1,)`

---

## Running the Baseline (Serial)

You can benchmark and profile the baseline (single process) using our profiling suite:

```bash
poetry run python -m src.baseline.main   --data-dir ./data/np32   --ext "*.npz"   --idx 0   --sizes 512,1024   --repeats 3   --outdir benchmark_out
```

> If you don’t have `src/baseline/main.py`, you can still reuse `src/common/profiling.py` utilities directly in a small script that loads one frame and calls `run_baseline_suite(...)`.

Artifacts include:
- `benchmark_out/baseline_wallclock.csv`
- `benchmark_out/profile_dn2dem_pos_*.{prof,txt}`
- Optional `lineprofile_*.{lprof,txt}` if `line_profiler` is installed
- `benchmark_out/env.json` with environment & shape snapshot

---

## Using with Dask

The Dask pipeline tiles the frame and runs the vendor solver per tile:

- It **broadcasts** the heavy constants (`T_RESP`, `T_RESP_LOGT`, `TEMPS`, `nmu`) to workers once via `client.run(_set_constants, ...)` to avoid bloating the task graph.
- Each tile uses the same vendor function, so behavior is identical to the serial path, modulo scheduling/partitioning.

CLI example (local cluster, CPU-bound workers):

```bash
poetry run python -m src.dask.main   --data-dir ./data/np32   --ext "*.npz"   --idx 0   --sizes 512,1024   --tile 128,128   --n-workers 4   --threads-per-worker 1   --no-processes   --memory-limit 6GB
```

---

## Troubleshooting

- **“Tresp needs to be the same number of wavelengths/filters as the data.”**
  - Make sure `T_RESP.shape[1] == 6` if your data has 6 bands.
  - Verify you’re loading the correct response set in `src/common/dem_api.load_tresp()`.

- **High unmanaged memory / memory pausing (Dask):**
  - Reduce tile size (e.g., `--tile 64,64`).
  - Lower `--n-workers` to 2–3 on memory-constrained machines.
  - Keep `threads-per-worker=1` and use `--no-processes` on Windows.
  - Ensure we **persist & reduce** (we compute `sum()` rather than materializing full arrays).

- **Graph too large warnings:**
  - We avoid shipping responses inside tasks; if you see *“Sending large graph …”* ensure you run with our latest `src/dask/runner.py` which caches constants via `_set_constants`.

---

## Reproducibility Notes

- We cap BLAS/OpenMP threads where appropriate (see `src/common/threads.py` and the Dask CLI flags).
- `benchmark_out/env.json` captures Python, NumPy, platform, and shapes.

---

## License Notes

The `vendor/` code may come with its own license terms. Keep it unmodified and attribute appropriately. Our wrappers and orchestration code follow the repository’s main license.
