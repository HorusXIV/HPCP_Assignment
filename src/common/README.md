# `src/common` — Shared Utilities (CPU/GPU ready)

This package hosts small, **stable building blocks** that higher-level code (baseline, Dask, GPU paths) rely on. It’s designed to be **boring and predictable** so you can swap implementations (e.g., GPU kernels) without touching call sites.

> TL;DR for GPU folks: start with [`gpu.py`](#gpuby), use the device-agnostic helpers in [`backend.py`](#backendpy), respect the **shape contracts** below, and keep the Dask tiling interface unchanged.

---

## Directory Map

- [`__init__.py`](#initpy) — package marker; keeps public surface tiny
- [`threads.py`](#threadspy) — BLAS/OpenMP thread caps, safe context manager
- [`backend.py`](#backendpy) — **device abstraction** (NumPy / CuPy) + CPU wrapper for vendor solver
- [`dem_api.py`](#dem_apipy) — single place to **load T-response matrices** and expose `dn2dem`
- [`gpu.py`](#gpuby) — GPU helpers (device selection, timing, pinned host buffers)
- [`nvtx.py`](#nvtxpy) — optional NVTX ranges (no-op if not installed)
- [`io.py`](#iopy) — rigid **I/O contract** for data files; stack/frame utilities
- [`responses.py`](#responsesspy) — synthetic T-response generator (for tests/dev)
- [`post.py`](#postpy) — post-processing: DEM → temperature maps
- [`profiling.py`](#profilingpy) — baseline profiling/bench harness + tiny cross-module `bench_row`

---

## Shape & Type Contracts (🔥 IMPORTANT)

These are relied upon by both the baseline and Dask runners.

- **Input frame to solver**: `(6, H, W)` `float32` (channel-first)
- **Vendor solver internal** expects `(H, W, 6)`; we convert inside wrappers
- **T-response tensors** from `load_tresp()`:
  - `T_RESP`: `(n_tresp, 6)` (must match **6 filters**)
  - `T_RESP_LOGT`: `(n_tresp,)` (log10 temperature grid)
  - `TEMPS`: `(nt+1,)` (Kelvin **bin edges**; `nt = len(TEMPS)-1`)
- **DEM output (per tile / full image)**: `(H, W, nt)` `float32`

If `T_RESP.shape[1] != 6`, the pipeline should fail early with a clear error. Keep that check in place.

---

## `threads.py`

Utilities to cap numerical libraries’ threads for reproducibility and to avoid oversubscription.

```python
from src.common.threads import early_env_caps, runtime_caps

early_env_caps(1)        # set env before importing numpy/scipy (driver process)
with runtime_caps(1):    # set caps during a critical section if threadpoolctl is available
    ...  # run bench
```

**Environment variables covered**: `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `BLIS_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `OMP_NUM_THREADS`, `NUMEXPR_NUM_THREADS`.

---

## `backend.py`

Device-agnostic helpers. Choose numpy (CPU) or cupy (GPU) at runtime.

```python
from src.common.backend import xp_for, to_device, to_host, has_cupy

xp = xp_for(None)    # -> numpy
if has_cupy():
    xp = xp_for(0)   # select GPU 0 and return cupy module
arr_dev = to_device(arr_cpu, device=0)
arr_cpu = to_host(arr_dev)
```

It also exposes a **CPU wrapper** for the vendor solver:

```python
from src.common.backend import dn2dem
demmap, edemmap, logT_bins, chisq, dn_reg = dn2dem(frame_6hw, T_RESP, T_RESP_LOGT, TEMPS, nmu=42)
```

The wrapper normalizes `(6,H,W)` → `(H,W,6)`, clips NaNs/infs, and uses `sqrt(DN)` as per baseline.

---

## `dem_api.py`

Single entry-point to load **T-response** constants and to expose the canonical `dn2dem` (re-export).

- `load_tresp()` → `(T_RESP, T_RESP_LOGT, TEMPS)`
- `dn2dem(...)` → calls the CPU baseline by default (GPU overrides can live here later)

> The Dask graph calls `load_tresp()` once on the driver and ships constants to workers via `client.run(_set_constants, ...)` to avoid graph bloat.

---

## `gpu.py`

Small helpers for CUDA when using CuPy:

- `available()` — can we import cupy?
- `set_device(idx)` — select device
- `sync()` — stream sync
- `cuda_event_timer()` — context manager for low-overhead GPU timing
- `pinned_empty(shape, dtype)` — **pinned host memory** (fast H2D/D2H)

```python
from src.common.gpu import available, set_device, cuda_event_timer, pinned_empty

if available():
    set_device(0)
    with cuda_event_timer() as t:
        ...  # GPU work
    print("seconds:", t.seconds)
```

---

## `nvtx.py`

Optional NVTX annotation. Safe to import on systems without NVTX — the context is a no-op.

```python
from src.common.nvtx import nvtx_range

with nvtx_range("dem-kernel"):
    ... # GPU code region
```

---

## `io.py`

Rigid, minimal I/O for `.npz` datasets:

- `default_files(ext="*.npz", directory=None) -> list[Path]` — discover inputs
- `load_np_stack(files, idx=-1, channels_last=False, dtype=None, contiguous=True)` → `(N,6,H,W)`
- `frame_for_solver(stack, i=0)` → `(H,W,6)` from either `(N,6,H,W)` or `(N,H,W,6)`

**Gotchas**

- Validates each file has `bands` with shape `(6,H,W)` and consistent spatial shape across all files.
- `idx` can be int, slice, list, or `-1` (all). Dask runner passes user-friendly strings which it parses to these forms.
- Returns C-contiguous arrays for vendor compatibility.

---

## `responses.py`

Synthetic response generator for tests:

```python
from src.common.responses import prepare_synthetic_responses
T_RESP, T_RESP_LOGT, TEMPS = prepare_synthetic_responses(nt=24, nf=6)
```

Handy for sanity checks without shipping large real response matrices.

---

## `post.py`

DERIVED maps from DEM:
- `dem_to_temp_maps(demmap, logT_bins) -> (mean_logT, peak_logT)`

```python
mean, peak = dem_to_temp_maps(demmap, logT_bins)
# shapes: (H,W) each; mean is log10(⟨T⟩), peak picks argmax bin
```

There’s also a Dask-friendly wrapper in `src/dask/post.py` that maps tiles lazily.

---

## `profiling.py`

Two roles:

1. Baseline profiling harness (serial):
   - `run_baseline_suite(...)` → writes env snapshot, wall-clock CSV/MD, cProfile, optional line_profiler
2. Cross-module minimal sink:
   - `set_bench_outdir(path)`
   - `bench_row(**kw)` → appends a single CSV row to `<outdir>/profiling_dask.csv`

The Dask runner uses `bench_row(...)` to record global wall-clock, tiling, and cluster settings.

---

## How Dask Uses `src/common`

- `io.default_files(...)` & `io.load_np_stack(...)` to load `(N,6,H,W)` then select one `(6,H,W)` frame (and optionally crop).
- `dem_api.load_tresp()` to fetch constants; broadcast to workers via `client.run(_set_constants, ...)`.
- `dask.map_blocks` calls into `tiling._blk` which uses `dem_api.dn2dem(...)`. **No large constants inside the task graph** → better memory behavior.
- Post-proc (optional) via `src/dask/post.py` → DEM to temperature maps lazily.

---

## Quickstart Snippets

**CPU-only baseline call**

```python
from src.common.io import default_files, load_np_stack
from src.common.dem_api import load_tresp, dn2dem

files = default_files(directory="data/np32")
stack = load_np_stack(files, idx=0).astype("float32")
frame = stack[0]  # (6,H,W)

T_RESP, T_RESP_LOGT, TEMPS = load_tresp()
demmap, edemmap, logT_bins, chisq, dn_reg = dn2dem(frame, T_RESP, T_RESP_LOGT, TEMPS, nmu=42)
```

**GPU extension (sketch)**

```python
from src.common.backend import xp_for, to_device, to_host
from src.common.gpu import available, set_device

if available():
    set_device(0)
    xp = xp_for(0)  # cupy
else:
    xp = xp_for(None)  # numpy

# Example: move inputs to GPU (if any GPU-side pre/post is needed)
frame_dev = to_device(frame, device=0)  # no-op on CPU
# ... run your custom kernels here ...
frame_host = to_host(frame_dev)
```

---

## Testing & CI Guidance

- Unit tests should **mock small arrays** and assert shape/value invariants.
- Synthetic responses (`responses.py`) keep tests light and deterministic.
- For GPU code, ensure CPU fallbacks exist (skip tests if `cupy` not available).
- Avoid importing heavyweight libs at module import time—keep imports inside functions for faster CLI startup.

---

## Contribution Rules for This Package

- Do **not** change function names or return shapes without updating:
  - `src/dask/runner.py`
  - `src/dask/tiling.py`
  - any baseline or notebooks that import these helpers
- Any new GPU/CPU kernels must preserve the public contracts and dtype (`float32` where practical).
- If you need additional constants, add them to `dem_api.load_tresp()` and keep the Dask broadcast pathway (`_set_constants`) in sync.

---

## FAQ

**Q: The pipeline yells “Tresp needs to be the same number of wavelengths/filters as the data.”**  
A: `T_RESP.shape[1]` must equal `6` for the provided AIA channels. Fix your responses or the dataset (or modify the solver to accept different `nf`).

**Q: Where do we plug in a GPU implementation of the solver?**  
A: Add a `dn2dem_gpu(...)` in `dem_api.py` or a separate module, and **choose** it based on a CLI flag/env var. Keep the signature and return shapes identical to `dn2dem`.

**Q: Any memory tips when running under Dask?**  
A: Reuse `dem_api.load_tresp()` once; broadcast via `client.run`. Keep per-task outputs small (e.g., sum/reductions during benchmarking) to avoid materializing giant arrays. Keep tiles around `128–256` on machines with ~6–8 GB per worker.

---

Happy hacking! 👩‍💻👨‍💻
