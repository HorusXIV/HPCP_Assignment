# Dask-based DEM Runner

This package provides a Dask-parallelized workflow for the HPCP assignment.

## Components

- **`main.py`** – CLI entry point. Parses args, builds client, runs `run_dask_suite`.
- **`runner.py`** – Core orchestration:
  - Loads data from `src.common.io`.
  - Loads responses from `src.common.dem_api.load_tresp`.
  - Broadcasts constants to workers via `_set_constants`.
  - Builds the Dask graph with `dem_map_blocks`.
- **`tiling.py`** – Maps tiles `(6,h,w) → (h,w,nt)` using vendor `dn2dem`.
- **`post.py`** – Reduces DEM maps to temperature diagnostics.

## CLI Usage

```bash
poetry run python -m src.dask.main --no-processes --n-workers 4   --threads-per-worker 1 --memory-limit 6GB --tile 128,128
```

### Common Flags

- `--tile h,w` → tile size (smaller = safer memory, larger = faster).
- `--n-workers` → number of workers (try 3–4 on 16 GB Windows).
- `--threads-per-worker 1` + `--no-processes` → best on Windows.
- `--memory-limit` → per-worker cap (e.g. `6GB`).
- `--use-synthetic` → use synthetic responses instead of real files.

## Tips

- Adjust Dask memory thresholds to avoid early warnings:
  ```powershell
  $env:DASK_DISTRIBUTED__WORKER__MEMORY__TARGET = "0.55"
  $env:DASK_DISTRIBUTED__WORKER__MEMORY__SPILL = "0.65"
  $env:DASK_DISTRIBUTED__WORKER__MEMORY__PAUSE = "0.80"
  $env:DASK_DISTRIBUTED__WORKER__MEMORY__TERMINATE = "0.98"
  ```
- Ensure `T_RESP.shape[1] == 6` (bands must match data).
- Expect a ~10–12 MiB graph size — this is normal.

## Outputs

Artifacts (benchmarks, profiles, etc.) are written under `benchmark_out/`.
