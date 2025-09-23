_# Hand‑off: Baseline & Dask Runners, Benchmarking, and Verification

**Owner (handover):** _Your Name_

This document gives the next maintainer everything needed to run, extend, and reuse the parts I implemented: the **baseline CPU runner**, the **Dask runner** (local & SLURM friendly), and the **benchmarking / verification plumbing**.

---

## 0) TL;DR Quickstart

```bash
# 1) Install lean deps
poetry install --without notebooks,data

# 2) Baseline CPU run (single machine)
poetry run hpcp-baseline --sizes 4096

# 3) Dask run (local cluster)
poetry run hpcp-dask --cluster-mode local --sizes 4096 --tile 512 --n-workers 4 --no-processes

# 4) (Optional) Verify against goldens
poetry run verify verify \
  --data-dir data/np32 \
  --golden-root data/golden \
  --sizes 1024 \
  --module cpu

# 5) Dev loop
poetry run ruff check --fix && poetry run ruff format
poetry run pytest -q
```

---

## 1) Code structure you’ll touch

```
src/
  baseline/
    cli.py                # Baseline CLI flags
    main.py               # Baseline entry; parses flags → runner
    runner.py             # Baseline execution + benchmarking hooks
    vendor/               # Upstream numerical kernels (do not lint/change)

  dask/
    main.py               # Dask entry; cluster bootstrapping & task selector
    runner.py             # Cluster setup (Local/SLURM), env, logging
    suite.py              # The Dask workload: build array → map tiles → profile
    tiles.py              # parse_hw / parse_tile / gen_tiles helpers (reusable)

  common/
    __init__.py           # Package exports for profiling/verification
    dataio/               # Data loading helpers (npz stack, metadata)
    profiling/
      io_helpers.py       # bench.csv writer + outdir management
      reporting.py        # run_<stamp>.md “run card”, JSON writers
      samplers.py         # optional GPU/CPU samplers (NVML guarded)
    verification/
      goldens.py          # create goldens
      verify.py           # compare arrays & metrics
      check.py            # verify helpers for datasets/JSON

  singlegpu/, multigpu/   # (stubs/placeholders if present)

data/
  fetch_np32.py           # Convenience downloader for the demo npz set
  raw_to_np.py            # Pipeline to build flat NPZ stacks from raw FITS
  ...
```

> **Note:** `src/baseline/vendor/` is treated as third‑party. It’s excluded from lint and tests don’t assert its internals.

---

## 2) Data: what the runners expect

- **Input**: flat `.npz` files (e.g., under `data/np32/`) with arrays and metadata. Use `poetry run python data/fetch_np32.py` to download the demo set if needed.
- **Selecting frames**: baseline/dask CLIs accept `--idx` (frame index) or iterate across files using glob `--ext` (defaults exist; see CLI tables below).

---

## 3) Baseline runner (CPU)

### Entry points
- Poetry script: `hpcp-baseline` → `src.baseline.main:main`
- Programmatic: `src.baseline.runner.run_benchmark(...)`

### Common flags (subset)
| Flag | Default | Notes |
|---|---|---|
| `--data-dir PATH` | — | Root folder containing `.npz` inputs |
| `--ext PATTERN` | `*.npz` | Glob pattern for inputs |
| `--idx` | `-1` | `-1` = last; `all` = run all; or an integer index |
| `--sizes H[,W]` | `2048,2048` | Accepts `H`, `H,W`, or `HxW`/`H×W` strings |
| `--tile Th[,Tw]` | `256,256` | Tile size per task |
| `--nmu` | `42` | Regularization parameter passed into the solver |
| `--device` | `cpu` | Keep as `cpu` for baseline |
| `--single-thread` | off | Force single‑threaded BLAS/OpenMP |
| `--blas-threads INT` | auto | Cap BLAS threads |
| `--outdir` | `baseline/benchmark_out` | (Legacy) output root, see *Artifacts* |
| `--verify / --no-verify` | on | Compare to goldens; see *Verification* |
| `--verify-sizes` | (same as `--sizes`) | Only verify selected sizes |
| `--golden-root` | `data/golden` | Root with `size/` subfolders & JSON/NPZ |
| `--chisq-mode` | `exact` | `exact | auto | skip` for χ² comparison |

### Behavior
1. Parse sizes & tiles → generate tile windows.
2. Load frame(s) → run **shared solver** per tile.
3. Profile the compute region; append a row to `bench.csv`.
4. Emit a **run card** markdown (`run_<stamp>.md`) with key metrics and environment snapshot JSON.
5. If `--verify`, compare against goldens and add `verify_ok` to the bench row.

---

## 4) Dask runner (local & SLURM)

### Entry points
- Poetry script: `hpcp-dask` → `src.dask.main:main`
- Programmatic: `src.dask.runner.run_dask_suite(...)` → `suite.run(...)` task

### Cluster modes
- `--cluster-mode local` (default): spins up a `LocalCluster`. Use `--n-workers` & `--threads-per-worker` or `--adapt-min/--adapt-max` for adaptive.
- `--cluster-mode slurm`: boots a `SLURMCluster` with sensible defaults (temp/log dirs, dashboard). You can pass queue/account options via flags or env.

### Common flags (subset)
| Flag | Default | Notes |
|---|---|---|
| `--cluster-mode` | `local` | `local` or `slurm` |
| `--n-workers` | auto | Fixed workers for `local` |
| `--threads-per-worker` | auto | Threads per worker |
| `--adapt-min/--adapt-max` | — | Enable adaptive scaling |
| `--data-dir` / `--ext` / `--idx` | — / `*.npz` / `-1` | As in baseline |
| `--sizes` / `--tile` | see defaults | Same parsing as baseline |
| `--verify` / `--golden-root` | off by default | Optional verification step |
| `--bench-root` | `./benchmarking/dask/` | Output root (timestamp subdir auto) |

### Behavior
1. Bring up the desired cluster; cap BLAS threads per worker to avoid oversubscription.
2. Build a chunked lazy array from NPZ → `map_blocks` the solver over tiles.
3. Profile compute and scheduler overheads; write `bench.csv` + run card.
4. Save Dask diagnostics: `task-stream.csv`, optional `dask-report.html`.

---

## 5) Artifacts & layout

```
benchmarking/
  baseline/
    <YYYYMMDD-HHMMSS>/
      bench.csv
      run_<stamp>.md
      env.json
      summary.json
  dask/
    <YYYYMMDD-HHMMSS>/
      bench.csv
      run_<stamp>.md
      task-stream.csv
      dask-report.html
```

- **bench.csv** (superset of columns): `stamp,mode,H,W,Th,Tw,frames,wall_s,verify_ok,notes`
- Run cards summarize inputs, timing, host/BLAS info, and verification status.

> The timestamped subfolder is created under the backend’s bench root; calls within the same second may reuse the same stamp (by design).

---

## 6) Verification (goldens)

Use the CLI exposed via `poetry run verify`.

- **Create**: `poetry run verify make-goldens --data-dir data/np32 --golden-root data/golden --sizes 512,1024`
- **Verify**: `poetry run verify verify --data-dir data/np32 --golden-root data/golden --sizes 1024 --module cpu`
- Runners accept `--verify` to compare run outputs to goldens and record `verify_ok`.

**Golden layout** (per size):
```
data/golden/
  512x512/
    baseline.npz     # reference arrays
    baseline.json    # reference metrics
  1024x1024/
    ...
```

---

## 7) Modules to reuse

### `src/dask/tiles.py`
- `parse_hw(arg) -> (H, W)`: accepts `None`, `int`, `"HxW"`, or `[H, W]`.
- `parse_tile(arg) -> (Th, Tw)`
- `gen_tiles(H, W, Th, Tw) -> List[(y0, y1, x0, x1)]`

### `src/common/profiling/io_helpers.py`
- `set_bench_outdir(path)` → sets the sink for `bench.csv`.
- `bench_row(**cols)` → append/update `bench.csv` (auto‑header).

### `src/common/profiling/reporting.py`
- `write_run_card_md(outdir, stamp, bench_row, env, notes=[])` → `Path`.
- `write_json(outdir, name, obj)` → `Path`.

### `src/common/verification/*`
- `write_goldens(...)` → persist NPZ/JSON baselines.
- `compare_to_golden(...)`, `verify_against_golden(...)` → booleans + metrics.

### `src/common/dataio`
- Helpers to load the NPZ stacks and metadata in a consistent shape/dtype.

> These modules are intentionally backend‑agnostic so new backends (e.g., GPU) can reuse them without re‑plumbing.

---

## 8) Testing & quality

- **Run tests**: `poetry run pytest -q`
- The suite covers: tiling, CLI glue, profiling writers, and a vendor **boundary** (mocked). We purposely do **not** unit‑test vendor internals.
- **Lint/format**: `poetry run ruff check --fix && poetry run ruff format`
- Ruff excludes `src/baseline/vendor/` (via `extend-exclude` or per‑file ignores).

---

## 9) CI behavior (GitHub Actions)

- Workflow file sets up a Python 3.12 slim container on a self‑hosted runner, installs via Poetry, runs **lint (non‑blocking)** and **pytest (blocking)**, with a 30‑min timeout and concurrency cancellation. See `.github/workflows/`.

---

## 10) Operational tips

- **BLAS threads**: tune `--blas-threads` (or `--single-thread`) to avoid oversubscription when Dask has many workers.
- **Tiles**: start with `--tile 256` and adjust based on cache/perf; `gen_tiles` ensures coverage of right/bottom edges.
- **Windows/MKL numerics**: if experimenting with vendor routines directly, prefer `float64` inputs and broader response matrices; our tests avoid asserting vendor math.

---

## 11) Extending the system

- **New backend** (e.g., GPU): mirror the baseline’s interface; reuse `tiles.py`, `io_helpers.py`, and `reporting.py`. Emit `bench.csv` + run card for consistent benchmarking.
- **More diagnostics**: drop additional JSON into the run folder (e.g., per‑tile timings) and link them from the run card.
- **Default verification**: wire `--verify` paths to call `verify_against_golden` after compute and include `verify_ok` in `bench.csv`.

---

## 12) Troubleshooting

- **No inputs found**: confirm `--data-dir` points to NPZs; `data/fetch_np32.py` can populate a demo dataset.
- **Dask dashboard**: for SLURM, ensure the scheduler dashboard port is reachable (ssh tunnel if needed).
- **Same timestamped folder reused**: `_ensure_timestamped_root` stamps to the second; two runs within one second may share a folder — expected._

