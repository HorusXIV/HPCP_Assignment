HPC Assignment — Accelerating DEMREG
---
# Contents

- [Introduction](#introduction)
- [Methodology](#methodology)
  - [Problem & Baseline Definition](#problem--baseline-definition)
  - [Dataset & Preparation](#dataset--preparation)
  - [Hardware & Environment](#hardware--environment)
  - [Evaluation Metrics](#evaluation-metrics)
  - [Benchmark Protocol](#benchmark-protocol)
- [Benchmarking Results](#benchmarking-results)
  - [Raw Wallclocks and Resources](#raw-wallclocks-and-resources)
  - [Raw Speedup vs. Baseline](#raw-speedup-vs-baseline)
  - [Dask Normalization (CPU-only)](#dask-normalization-cpu-only)
    - [Interpretation](#interpretation)
  - [GPU Methods (context)](#gpu-methods-context)
  - [Variance & Threats (recap)](#variance--threats-recap)
- [Baseline (CPU) Setup](#baseline-cpu-setup)
- [Accelerating with Dask](#accelerating-with-dask)
- [Accelerating with Numba and CuPy](#accelerating-with-numba-and-cupy)
  - [Initial Improvement of Baseline](#initial-improvement-of-baseline)
  - [Transition to CuPy](#transition-to-cupy)
  - [Profiling and Observations](#profiling-and-observations)
  - [Conclusion](#conclusion)
- [multiGPU Setup](#multigpu-setup)
  - [Baseline and First Principles](#baseline-and-first-principles)
  - [Multi‑GPU Distribution and Initial Improvements](#multi-gpu-distribution-and-initial-improvements)
  - [Profiling](#profiling)
  - [Batch Sizing](#batch-sizing)
  - [Overlapping Transfers and Compute & Memory Pooling](#overlapping-transfers-and-compute--memory-pooling)
  - [Experiment: Triple vs. Double Buffering](#experiment-triple-vs-double-buffering)
    - [Hypotheses](#hypotheses)
    - [Measurements](#measurements)
    - [Hypothesis test summary](#hypothesis-test-summary)
    - [Conclusion](#conclusion-1)
  - [Improving Memory Stability](#improving-memory-stability)
  - [Experiment 2: Memory‑Handling Logic](#experiment-2-memory-handling-logic)
    - [Hypotheses](#hypotheses-1)
    - [Measurements](#measurements-1)
    - [Hypothesis test summary](#hypothesis-test-summary-1)
    - [Conclusion](#conclusion-2)
  - [Wrapup and Restoring Vendor Parity](#wrapup-and-restoring-vendor-parity)
  - [Overlapping Transfers and Compute](#overlapping-transfers-and-compute)
- [Discussion](#discussion)
  - [Dask](#dask)
  - [Numba/CuPy](#numbacupy)
    - [Lessons Learned](#lessons-learned)
    - [Future Work](#future-work)
  - [multiGPU](#multigpu)
    - [Lessons Learned](#lessons-learned-1)
    - [Future Work](#future-work-1)

# Introduction

This project accelerates the [DEMREG Python codebase](https://github.com/ianan/demreg) (NumPy‑heavy scientific workload) with a focus on reducing end‑to‑end runtime on realistic solar imagery while preserving correctness and reproducibility. We structure the work as:

1. **Baseline (CPU / NumPy)** — establish wall‑clock reference and profiling on full‑resolution inputs.  
2. **Dask (CPU orchestration)** — parallelize CPU execution with task scheduling and chunked evaluation on the host; no GPU is used in this path.  
3. **Single‑GPU (CuPy)** — replace NumPy compute/memory with CuPy and vectorize hot paths for device execution.  
4. **Multi‑GPU (CuPy + MPI)** — distribute work across multiple GPUs with overlapping host↔device transfers and compute.

All reported results use a fixed dataset shared across the team for comparability: **10 complete AIA bandsets (6 bands) at 4096×4096** per timestamp, prepared from raw FITS via our scripts and flattened into NumPy stacks. We evaluate each approach on **wall‑time per image**, **throughput (images/s)**, and **speedup vs. the CPU baseline**, and we record the exact hardware and library versions for reproducibility. Subsequent sections provide methodology, implementation notes, and benchmarks following the order above.


# Methodology

## Problem & Baseline Definition
We solve Differential Emission Measure (DEM) maps per image from 6-band AIA stacks of shape `(6, H, W)`. One “image” denotes a full 6-band set at 4096×4096; evaluation runs one DEM solve per image and reports wall-clock time and throughput. The CPU baseline uses the vendor DEM routine behind a thin wrapper so the rest of the code can call a stable API for `(dem, edem, chisq, logT)` per tile (see `src/dask/solver.py`).

Our comparison order is **Baseline (CPU) → Dask (CPU-only) → Single-GPU (CuPy) → Multi-GPU (CuPy+MPI)**. Each variant keeps identical solver settings and dataset; only the execution strategy changes.

## Dataset & Preparation
All benchmarks use a fixed, shared dataset: **10 complete AIA bandsets (6 bands each) at 4096×4096**. Raw FITS frames are fetched in parallel per wavelength (SunPy + parfive) into a manifest; files are bucketed by time to avoid name collisions. We convert the raw frames into flat NumPy stacks (channels-first) and distribute them as NPZ bundles. The helper `fetch_np32.py` (and `data/fetch_np32.py`) unpacks and verifies the shared archive for team-wide reproducibility (see `download.py`, `raw_to_np.py`, `fetch_np32.py`).

Dask dataset utilities expect NPZ files with a `bands` array; builders produce lazy stacks shaped `(F, Hc, Wc, 6)` with tile-aligned chunks `(1, Th, Tw, 6)` to match the solver’s tiling (see `src/dask/main.py` and `src/dask/runner.py`).

**Note on frame selection:** for time reasons, the **Baseline (CPU)** and **Dask (CPU-only)** experiments were executed on **frame index 0 only**. Unless otherwise stated, **Single-GPU** and **Multi-GPU** results are reported over the full 10-frame set.

## Hardware & Environment
All experiments ran on the **FHNW SLURM cluster** (partition: `performance`). Below are the resources requested by each script; actual node models are recorded in run logs.
- **Baseline (CPU)** — `--cpus-per-task=16`, `--mem=80G`, `--time=02:00:00` (no GPU). Thread caps used: `OMP_NUM_THREADS=16`, `MKL_NUM_THREADS=16`, `OPENBLAS_NUM_THREADS=16`, `NUMEXPR_NUM_THREADS=16`.
- **Dask (CPU-only)** — `--cpus-per-task=32`, `--mem=96G`, `--time=04:00:00` (no GPU).
- **Single-GPU (CuPy)** — `--gres=gpu:1`, `--cpus-per-task=2`, `--mem=128G`, `--time=24:00:00`.
- **Multi-GPU (CuPy+MPI)** — `--nodes=1`, `--ntasks-per-node=4`, `--gpus-per-task=1`, `--cpus-per-task=3`, `--mem=32G`, `--time=24:00:00`, `--hint=nomultithread`.
(From `slurm_run_baseline.sh`, `slurm_run_Dask.sh`, `slurm_run_singleGPU.sh`, `slurm_run_multiGPU.sh`).

## Evaluation Metrics
We report:
- Wall-time per image and throughput (images/s) from a standardized benchmarking harness that writes CSV/JSONL/Markdown summaries per run (see `src/common/profiling/reporting.py`).
- Speedup vs. baseline (ratio on the same dataset / frame selection, as applicable).
- Optional resource notes: peak memory, GPU utilization via NVML sampler, and Dask task-stream snapshots when relevant (see `src/common/profiling/samplers.py`).

## Benchmark Protocol
To ensure apples-to-apples comparisons across all four variants:
1. **Fixed input**: the same NPZ stacks are used for all methods. Baseline and Dask operate on **frame 0**; Single- and Multi-GPU operate on **all 10 frames** unless noted.  
2. **Common parameters**: default tile size `256×256` and `nµ = 42` unless stated otherwise; controllable via CLI (see `src/dask/cli.py`, `src/dask/main.py`).  
3. **Repeats and warm-up**: run with repeats ≥1 and, if needed, ignore the first timing if initialization affects iteration 1 (supported by the Dask and single-GPU runners; see `src/dask/runner.py`, `runner.py`).  
4. **CPU Dask (no GPU)**: Dask orchestrates host-side tiles over the 6-band stacks using chunking aligned to tiles; capture task streams and wall clocks via the shared profiling API (see `src/dask/main.py`, `src/common/profiling/profiler.py`).  
5. **Single-GPU (CuPy)**: replace NumPy arrays and ops with CuPy, keep solver semantics, and record device + run metadata in the run folder (see `gpu.py`, `runner.py`, `src/common/profiling/reporting.py`).  
6. **Multi-GPU (CuPy+MPI)**: distribute per-image batches across devices, overlap H2D/D2H with compute via streams (double/triple buffering), and use an adaptive batch-sizing heuristic that probes free memory and backs off on OOM; NVTX ranges and Nsight timelines are optional but recommended for representative runs (see `mpi_manager.py`, `nvtx.py`, `src/multiGPU/gpu_kernels.py`).

All runs are written to method-specific folders with a run card (Markdown) and machine-readable artifacts for aggregation and plotting later (see `src/common/profiling/reporting.py`).

# Benchmarking Results

**Normalization choice:** Yes. Dask used 32 CPUs vs baseline 16; we report raw wallclocks and normalized views for fairness. Equal-core normalization scales Dask's wallclock by the core ratio (32→16). We also show CPU-seconds (wallclock × CPUs) to reflect total compute budget. GPU methods are reported as raw wallclocks and speedups.

## Raw Wallclocks and Resources

| Method | Wallclock (s per frame) | Std (s) | CPUs | GPUs |
|---|------------------------:|---:|---:|---:|
| Baseline (CPU) |                  2204.0 | 20.0 | 16 | 0 |
| Dask (CPU-only) |                   974.0 |  | 32 | 0 |
| Single-GPU (CuPy) |                    35.0 |  | 2 | 1 |
| Multi-GPU (CuPy+MPI) |                   28.99 |  | 12 | 4 |

## Raw Speedup vs. Baseline

| Method | Raw Speedup (×) |
|---|----------------:|
| Dask (CPU-only) |           2.26× |
| Single-GPU (CuPy) |          62.97× |
| Multi-GPU (CuPy+MPI) |          76.03× |  

## Dask Normalization (CPU-only)

- Equal-core normalization (scaled to 16 CPUs): **1948.0 s** (baseline 2204.0 s → 1.13× speedup).
- CPU-seconds (compute budget): Baseline **35264** vs Dask **31168** → **1.13×** reduction in CPU-seconds.

### Interpretation
- Raw Dask speedup vs baseline: ~2.26×, but it used 2× the CPU cores.
- Equalizing cores (16 vs 16) yields a modest **1.13×** speedup for Dask, indicating most gains come from parallelism rather than single-core efficiency.
- CPU-seconds show that Dask used ~88% of the baseline's total CPU budget while also finishing sooner (more efficient parallel utilization).

## GPU Methods (context)
- Single-GPU achieves **62.97×** speedup; Multi-GPU achieves **84.77×** vs baseline on wallclock. These were run on all frames unless noted; CPU paths were restricted to frame 0.

## Variance & Threats (recap)
- Baseline variance: ±20 s (0.9%). Dask and GPU runs listed here are single measurements; add repeats if available.
- Hardware allocation differences (CPU counts, GPU types) and frame selection differences (frame 0 vs all) can bias comparisons; hence both raw and normalized views.
- First-iteration effects, caching, and cluster contention may shift wallclocks by a few percent; our protocol discards warm-up when applicable.

## Baseline (CPU) Setup

**Objective.** Establish a clear, reproducible CPU reference and a thin solver wrapper that other paths can call identically.

**What runs.**
- Input: one 6-band AIA frame at 4096×4096 (frame index 0 of the shared NPZ set).
- Solver: vendor path via the existing wrapper for `(dem, edem, chisq, logT)` per tile.
- Tiling: default tile size 256×256, iterated over the full image; results stitched into full-frame outputs.
- Threads: BLAS/OMP threads capped to the SLURM allocation for stable timings.

**Key pieces.**
- CLI/entry-point and runner from the repo’s CPU path.
- Thread caps: `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`.
- Run cards and CSV/JSONL summaries are written to a timestamped results dir.

**How we ran it (frame 0).**
```
# 16 CPUs, no GPU; tile 256, nµ 42
python -m src.dask.main --frame 0 --tile 256 --nmu 42 --data <npz_dir>
```
This produces the wallclock used as the baseline reference in Benchmarking.

---

## Accelerating with Dask

**Goal.** Keep the CPU solver but expose parallelism across tiles using Dask; this path is CPU-only.

**Design.**
- Lazily open NPZ stacks and construct a tile-aligned array with chunks shaped `(1, Th, Tw, 6)`.
- Map a thin CPU solver over tiles; collect and stitch outputs back to full-frame arrays.
- Reuse the same CLI switches as baseline (tile size, nµ, frame selector) so results are comparable.

**Execution details.**
- Frame selection: for time reasons, runs were done on **frame 0** only (same as baseline).
- Resources: requested 32 CPUs (`--cpus-per-task=32`) to saturate host-side parallelism.
- Fairness: thread caps are set so BLAS/OMP do not over-subscribe beyond SLURM allocation.
- Instrumentation: task stream and wallclock via the shared profiling utilities; writes CSV/JSONL/MD per run.

**Graph sketch.**
```
npz -> (lazy open) -> rechunk to (1, Th, Tw, 6)
    -> map_blocks(solve_tile_cpu) -> stack
    -> write outputs + run card (CSV/JSONL/MD)
```

**Practical notes.**
- Tile size 256×256 balances per-task overhead and cache behavior; larger tiles reduce scheduler pressure but raise peak memory.
- Dask excels at exposing coarse CPU parallelism; it does not change the underlying algorithm, so per-tile compute cost remains similar to baseline.
- Because Dask used 32 CPUs vs baseline’s 16, the Benchmarking section reports both **raw** and **normalized** comparisons.

**How we ran it (frame 0).**
```
# 32 CPUs, no GPU; tile 256, nµ 42
python -m src.dask.main --frame 0 --tile 256 --nmu 42 --data <npz_dir> --repeats 1
```

## Accelerating with Numba and CuPy

### Initial Improvement of Baseline

I started from the baseline NumPy implementation, adding Numba decorators (`@jit`, `@cuda.jit`) to accelerate the most computationally expensive loops. The goal was to reduce Python’s interpreter overhead and offload basic operations to the GPU.  

However, after testing this initial setup, performance improvements were negligible, the processing time measured in at about 2h.  

The main reasons for this limited performance were:

- Serial execution across tiles rather than true parallelization  
- Low GPU utilization (typically <10%) due to small kernels and excessive synchronization  
- Frequent GPU to CPU memory transfers, negating any acceleration benefits  

In short, while Numba did accelerate the inner loops, the surrounding data movement and control flow dominated runtime.

### Transition to CuPy

To overcome the I/O bottlenecks, the implementation was refactored to use CuPy for memory management and array operations while still using Numba for the inner computational kernels.  

This hybrid approach, however, led to context conflicts between Numba and CuPy’s separate CUDA drivers, producing errors such as:

```
CUDA_ERROR_INVALID_CONTEXT
```

and frequent GPU crashes after multiple frames.  
Even when it ran, the computation time was still around 40 minutes per image, an improvement but far from optimal.

To fix this, all Numba kernels were removed and replaced with a pure CuPy implementation.  
The solver logic including DEM, EDEM, and Chi-squared was fully vectorized and executed using CuPy’s own GPU primitives (built on CUBLAS and CUB).  

This transition simplified memory management, removed kernel launch overheads, and enabled asynchronous CUDA stream–based parallel tile processing.  
After these changes, the runtime dropped dramatically to ~4.5 minutes per image on a single NVIDIA RTX A4500 GPU.


### Profiling and Observations

After implementing the CuPy-only path, profiling revealed:

- Good memory throughput, with most GPU memory being effectively used depending on the chosen tile size  
- Moderate GPU compute utilization (typically 30–50%), suggesting that while memory and I/O are efficient, the computation itself may not be fully saturating the GPU’s cores  
- Each frame is now processed independently, with GPU memory being freed after each image to avoid out-of-memory (OOM) issues  

The remaining performance bottleneck appears to be in the solver’s numerical complexity, which is already heavily optimized and not trivially parallelizable further.


### Conclusion

Using Numba for single-GPU acceleration of NumPy-heavy scientific code provides limited benefits, especially when data transfer and context management dominate runtime.  

By contrast, refactoring the implementation to use CuPy exclusively for both computation and memory management provided an order-of-magnitude speedup, reducing runtime from over 2 hours to about 4.5 minutes per image.  


## multiGPU Setup

In this Section I will describe the multiGPU implementation and the steps taken to improve the performance of the code. All experiments were run on the FHNW Slurm Cluster. Statistical Tests can be found in the Notebook [Stats4Report.ipynb](Notebooks/Stats4Report.ipynb).

### Baseline and First Principles

I began with the baseline NumPy implementation that was provided. As a first step, I replaced NumPy with CuPy and vectorized the hot paths to eliminate as many Python loops as possible. This converted the code to being GPU‑friendly. However, when I ran it on SLURM, GPU utilization was only around 14%, and overall it was even slower than the CPU version by a huge margin (9+ hours per image).

To run on the cluster, I created a Singularity container. This took significant time due to numerous dependencies and limited prior experience with Singularity. After matching CuPy and CUDA versions and installing additional libraries to avoid compilation errors, I was able to run the code on the cluster.

### Multi‑GPU Distribution and Initial Improvements

Next, I added MPI (using mpi4py) to distribute work across multiple GPUs. I chose a strategy where each image is partitioned across all available GPUs. This is feasible because the problem is embarrassingly parallel: each pixel is independent and does not influence any other.
This was straightforward to implement and balanced load reasonably well. Alternatives would have been to distribute disjoint subsets of images to each GPU, but if image sizes vary, this can lead to load imbalance. Another alternative would have been to distribute a subset of images per node and then split them among the GPUs on that node. I decided against this because the added complexity had uncertain benefits. With this multi‑GPU implementation and some minor code tweaks, I reduced runtime to about 2 hours per image.

### Profiling

At this point the obvious improvements were exhausted, so it was time to profile. I used NVIDIA Nsight Systems with NVTX annotations for timeline analysis. I profiled each GPU briefly but analyzed one in detail (usually GPU 0), assuming similar behavior. Additionally, I used `nvidia-smi` and `nvtop` on the compute nodes to monitor memory usage and GPU utilization in real time.

This led to two key insights: (1) even with the improvements so far, GPU utilization was still modest (around 30–35%), and (2) there were significant idle gaps between bursts of higher utilization.

### Batch Sizing

Based on these insights, I implemented the biggest improvement of the project: larger batches. Large batches dramatically increased throughput by minimizing kernel launch and transfer overhead.
I added an adaptive batch‑sizing heuristic: estimate free GPU memory, approximate batch footprint, attempt the largest safe batch, and back off on OOM. This achieved high memory utilization but still wasted time due to OOM retries. About one third of total GPU time in the worst cases. Overall, this brought the time down to about 4 minutes per image.

However, part of this behavior was due to a bug in my kernel logic, where I accidentally applied a term involving dn**2 twice. This made the computational complexity effectively O(n^2) instead of the intended O(n), and increased memory traffic. After fixing this bug (and a similar one affecting write operations), I was able to run with much larger batches, and the time dropped to about 35 seconds per image.

### Overlapping Transfers and Compute & Memory Pooling

After the batch work, I addressed pipeline idle gaps. I introduced CUDA streams with a double‑buffering pattern to overlap host‑to‑device (H2D) copies with compute. Additionally, I implemented a memory pool to avoid repeated allocation overhead. This reduced the gaps between kernels and improved effective GPU occupancy. The profile then showed ~205.3 s of pure compute within a 334 s run, even with profiler overhead and occasional OOM retries.

Extending the approach to triple buffering (H2D → compute → copy in separate streams) should smooth out the pipeline even more. To verify this, I implemented it and ran an experiment to compare double vs. triple buffering.

### Experiment: Triple vs. Double Buffering

Almost all previous changes yielded clear improvements, so the focused experiment was comparing double vs. triple buffering. I ran 3 rounds of 10 images each, measuring wall time and compute time per image. I discarded the first two rounds (warm‑up) and performed a one‑sided Welch’s t‑test (unequal variances) on the remaining data. All data were collected from NVIDIA Nsight Systems reports.
To avoid additional variability, I kept the batch size constant at 198'759 for both implementations. Another way to improve test reliability would have been to pin the SLURM job to a specific node, but I decided against it due to uncertainty about scheduling behavior on our cluster.

#### Hypotheses
* **$$H_0$$:** The mean time with triple buffering is **not lower** than with double buffering.
  ( $$\mu_\text{triple} \ge \mu_\text{double} $$ )
* **$$H_1$$:** The mean time with triple buffering is **lower** than with double buffering.
  ( $$\mu_\text{triple} < \mu_\text{double} $$ )

We define the significance level α = 0.05.

#### Measurements

Time in seconds per image:

|  # | Double Buffer – Time/image | Double Buffer – Compute  | Triple Buffer – Time/image | Triple Buffer – Compute |
| -: | -------------------------: | -----------------------: | -------------------------: | ----------------------: |
|  1 |                      24.68 |                   16.905 |                      24.26 |                  16.727 |
|  2 |                      24.68 |                   17.052 |                      24.29 |                  16.853 |
|  3 |                      24.92 |                   17.125 |                      24.40 |                  16.953 |
|  4 |                      25.09 |                   17.220 |                      24.68 |                  17.052 |
|  5 |                      25.38 |                   17.388 |                      24.98 |                  17.322 |
|  6 |                      25.45 |                   17.449 |                      24.93 |                  17.388 |
|  7 |                      25.45 |                   17.556 |                      25.28 |                  17.376 |
|  8 |                      25.47 |                   17.498 |                      25.00 |                  17.410 |
|  9 |                      24.63 |                   17.044 |                      24.64 |                  16.961 |
| 10 |                      24.49 |                   17.091 |                      24.44 |                  17.073 |
| 11 |                      24.54 |                   17.149 |                      24.69 |                  17.159 |
| 12 |                      24.65 |                   17.197 |                      24.61 |                  17.126 |
| 13 |                      24.97 |                   17.398 |                      24.98 |                  17.445 |
| 14 |                      25.10 |                   17.526 |                      25.06 |                  17.494 |
| 15 |                      25.25 |                   17.574 |                      24.85 |                  17.426 |
| 16 |                      25.20 |                   17.520 |                      24.99 |                  17.456 |
| 17 |                      25.03 |                   17.068 |                      24.37 |                  16.902 |
| 18 |                      25.23 |                   17.131 |                      24.49 |                  16.986 |
| 19 |                      25.36 |                   17.181 |                      24.54 |                  17.103 |
| 20 |                      25.23 |                   17.274 |                      24.59 |                  17.151 |
| 21 |                      25.35 |                   17.427 |                      24.82 |                  17.286 |
| 22 |                      25.51 |                   17.554 |                      24.84 |                  17.339 |
| 23 |                      25.65 |                   17.586 |                      25.42 |                  17.481 |
| 24 |                      25.61 |                   17.609 |                      25.09 |                  17.451 |

#### Hypothesis test summary

| Metric       | α (p-value threshold) | Welch’s t-test p-value | $$\mathbf{H_0}$$                                | Decision                |
| ------------ | --------------------: | ---------------------: | ----------------------------------------------: | ----------------------- |
| Wall Time    |                  0.05 |              0.0002424 | $$\mu_\text{triple} \ge \mu_\text{double} $$    | **Reject $$H_0$$**      |
| Compute Time |                  0.05 |                0.04911 | $$\mu_\text{triple} \ge \mu_\text{double} $$    | **Reject $$H_0$$**      |

#### Conclusion
The measurements show that the triple‑buffering implementation has a lower average wall time per image than the double‑buffering implementation. The t‑tests for both wall time and compute time yield p‑values below 0.05, leading to rejection of the null hypothesis. This indicates a statistically significant difference in both metrics, with triple buffering being faster; therefore, the triple‑buffered implementation is preferred.

### Improving Memory Stability

This was encouraging, but there was still a problem: out‑of‑memory (OOM) errors occurred frequently, especially with larger images or higher $$\mu$$‑grid points. Each OOM triggered a retry with a smaller batch (reducing the batch size by half each time), but this wasted time and hurt throughput.

There were multiple approaches to improve this (for example, statically reducing the number of $$\mu$$-grid points), but I wanted to keep vendor parity as high as possible. So I focused on improving the memory‑handling logic. Previously, after each OOM, the batch size was simply halved. The new approach is more robust: flush reclaimable pool blocks before sizing so the estimate reflects real free memory; set the target fraction of free memory to 0.7 by default (tunable via the `MULTIGPU_BATCH_MEM_FRAC` environment variable) to leave headroom for library overhead.


### Experiment 2: Memory‑Handling Logic

The test setup matches the previous experiment: 3 rounds of 10 images each, measuring wall time. I discarded the first two rounds (warm‑up) and performed a one‑sided Welch’s t‑test on the remaining data. Again, all data were collected from NVIDIA Nsight reports.

#### Hypotheses
* **$$H_0$$:** The newest memory handling is **not faster** than the old version.
  ( $$\mu_\text{new} \ge \mu_\text{old}$$ )
* **$$H_1$$:** The newest memory handling is **faster** (lower mean time).
  ( $$\mu_\text{new} < \mu_\text{old}$$ )

We define the significance level α = 0.05.

#### Measurements

Time in seconds per image. Even though this is the never version compared t the last test, I didn't fix the batch size, which led to a higher processing time per Image. Additionally, you can see that some servers take much longer than others.

|  # | Old Version |    New Version |
| -: | ----------: | -------------: |
|  1 |       29.27 |          22.92 |
|  2 |       29.50 |          23.65 |
|  3 |       28.20 |          25.18 |
|  4 |       28.91 |          25.60 |
|  5 |       29.06 |          26.28 |
|  6 |       28.91 |          26.81 |
|  7 |       28.61 |          26.60 |
|  8 |       29.27 |          22.14 |
|  9 |       28.00 |          23.30 |
| 10 |       28.80 |          23.48 |
| 11 |       28.19 |          25.45 |
| 12 |       28.70 |          25.57 |
| 13 |       28.57 |          25.46 |
| 14 |       27.60 |          23.76 |
| 15 |       28.76 |          22.74 |
| 16 |       29.85 |          23.10 |
| 17 |       23.36 |          24.02 |
| 18 |       25.01 |          22.74 |
| 19 |       25.08 |          23.10 |
| 20 |       26.14 |          24.02 |
| 21 |       26.21 |          25.47 |
| 22 |       26.21 |          26.04 |
| 23 |       25.29 |          26.30 |
| 24 |       25.40 |          26.17 |

#### Hypothesis test summary

| Metric           |                       Value |
| ---------------- | --------------------------: |
| α (significance) |                        0.05 |
| p-value          |    $$3.671\mathrm{e}{-08}$$ |
| H₀               | $$\mu_\text{new} \ge \mu_\text{old}$$ |
| Decision         |               **Reject $$H_0$$** |

#### Conclusion
The measurements show that the new memory‑handling logic has a lower average wall time per image than the old implementation. The t‑test yields a p‑value of $$3.671\mathrm{e}{-08}$$, which is far below the significance level of 0.05, leading to rejection of the null hypothesis. This indicates a statistically significant difference in wall time between the two implementations, with the new memory‑handling logic being faster.


### Wrapup and Restoring Vendor Parity
Following the performance work, a final sanity check against the additional sources revealed an implementation issue: my runs produced DEMs of around $$10^{80}$$. I found that the root cause was an ill-conditioned response matrix. The script had been using a trivial all-ones response matrix (K) as the response, which made the inversion nearly singular. The discrepancy principle then picked a very small $$\lambda$$ and amplified the noise, resulting in astronomically large DEM values.
To resolve this issue, I replaced the all-ones K with realistic, well-behaved synthetic responses, averaging them into bins and mapping them onto the DEM grid (logt, dlogt). This ensures that the solver recognises an (nt × nf) matrix that is consistent with the baseline/vendor pipeline. I also added a floor to the lambda to prevent extreme values.
I also wired in a dn2dem_pos wrapper and organised my code to more closely match the vendor structure, which will make it easier to read. However, gathering all the results at root rank adds some overhead, increasing the time per image to around 36 seconds.

### Overlapping Transfers and Compute
To reduce the total runtime, I implemented overlapping transfers and computed as much as possible. While the main script is saving the data, the next image is loaded and processed in the background.
Although this does not save any compute time, it helps to keep the GPU busy and reduces the overall wall time to an average of roughly 30 seconds.

# Discussion

## Dask

**What worked.** Dask delivered a solid wall‑clock reduction on the CPU path by exposing coarse parallelism over tiles. On the FHNW cluster we ran **frame 0** with `--cpus-per-task=32` (vs. 16 for the baseline), achieving **974 s** vs. **2204 s** for the baseline (≈ **2.26×** faster). Task scheduling + thread capping kept cores busy without BLAS oversubscription.

**Normalization matters.** Because Dask used 2× the CPU allocation, we also computed a like‑for‑like view:
- **Equal‑core normalization (32→16 CPUs):** ≈ **1948 s**, i.e., **1.13×** faster than baseline at the same core count.
- **CPU‑seconds (compute budget):** Dask ≈ **31,168** vs. baseline **35,264** → **1.13×** lower CPU‑time.

**Interpretation.** Most of Dask’s raw speedup came from using more cores, not per‑core efficiency. That’s expected: the solver cost per tile is essentially unchanged; Dask primarily improves **parallel utilization** and orchestration. Even so, the lower CPU‑seconds indicate some runtime overheads (I/O, Python control flow) were amortized better under Dask’s task graph.

**Bottlenecks & limits.**
- **Single‑node saturation.** With tile size 256×256, scheduler overhead and Python call boundaries become visible; pushing to smaller tiles hurts.
- **Memory bandwidth bound.** The CPU solve remains heavy on memory traffic per tile; faster nodes or larger LLC benefit more than extra threads beyond a point.
- **I/O cold starts.** First‑run effects (cache, page cache, import/JIT) can skew short runs; we mitigated by aligning Dask and baseline to the same **frame 0** and measuring with the same harness.

**When to use Dask here.** If GPUs are unavailable or oversubscribed, Dask offers a simple CPU‑only parallel path with reproducible gains, at the cost of higher node allocations. It pairs well with batch evaluations (many frames) where warm‑up is amortized.

**Future Dask work (lightweight).**
- Coarser chunking for lower scheduler pressure on big nodes.
- Pinning to a partition/node class with consistent CPU models.
- Optional local disk caching for repeated reads of the same stacks.
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia odio vitae vestibulum vestibulum. Cras venenatis euismod malesuada.

## Numba/CuPy

With the final CuPy-based implementation, the average compute time per image stabilized around 4.5 minutes (compared to the 2+ hours of the baseline)(**ADJUST IF NECESSARY**). GPU utilization reported by `nvidia-smi` typically ranged between 30–50%, which suggests that while memory throughput was good, the computation itself remained somewhat underutilized. This behavior is likely due to the per-tile vectorization strategy, tiles are processed in parallel on the GPU, synchronization points and host–device data transfers still introduce idle periods.

While the CuPy rewrite significantly improved runtime, it came with trade-offs in memory stability and transparency. Managing GPU memory explicitly (via CuPy’s memory pools) was crucial to prevent OOM errors on large frames. The use of CuPy over Numba proved beneficial: CuPy’s kernels leverage NVIDIA’s optimized CUDA libraries (CUBLAS, CUB), whereas Numba’s JIT-generated kernels, while flexible, introduced context instability when mixed with CuPy operations.

### Lessons Learned

1. Numba + CuPy interaction is fragile.  
   Mixing them can lead to invalid CUDA contexts, especially when both allocate memory on the same device. A single-GPU pipeline should use one GPU backend consistently.
2. CuPy excels at vectorized math, not orchestration.
   It’s ideal for replacing NumPy compute kernels but less efficient for managing complex GPU task pipelines (where PyTorch, Triton, or custom CUDA might be preferable).
3. Memory control is essential. 
   Freeing GPU memory after each frame (`cp.get_default_memory_pool().free_all_blocks()`) kept long runs stable and avoided OOM kills.

### Future Work

* Asynchronous tiling and batching: Implement true tile-level parallelism using CUDA streams or CuPy’s asynchronous APIs to improve kernel overlap.
* Adaptive tile sizing: Dynamically adjust tile size based on available GPU memory to balance throughput and stability.


## multiGPU

With the implemented improvements, the final compute time per image averaged 28.99 ± 1.74 seconds with 18.694 ± 0.677 seconds for GPU compute. At this point, I reached diminishing returns: GPU utilization during compute time is now around 90–100% (per `nvidia-smi`), and the timeline shows a steady stream of kernels with brief gaps for saving/loading data.
Occasional OOMs still happen but normally only once per image to get a good estimator for later batches. The adaptive batch sizing keeps memory usage high without frequent retries. Further improvements would likely require more complex changes, such as topology‑aware scheduling or algorithmic modifications. Given time constraints and the learning goals achieved, I did not pursue these further optimizations.

I also noticed strong dependence on the server/node. For example, running the final version on Server0092 with 4× NVIDIA RTX 2080 Ti was almost 4 seconds slower per image than on Server0101 with 4× NVIDIA RTX 3080 Ti. This variability complicates judging whether an observed improvement is real or noise, especially for effects in the 2–4 second range, which motivated the statistical testing above.

### Lessons Learned

1. Profiling reveals what matters. Nsight Systems with NVTX ranges shows behaviors you can’t see otherwise; live monitoring with `nvidia-smi`/`nvtop` helps target future work.
2. Batch size dominates. The biggest gains came from optimizing batch size. Larger batches amortize overhead and increase throughput but require careful memory management; otherwise, OOM retries waste time.
3. Overlap matters. Streams and buffering convert idle time into useful work. If you aim for full parallelism (including loading data while computing), beware race conditions and memory pressure. There is still some idle time while loading/saving data; removing it would likely require significantly more parallel I/O work (and Python may not be ideal for that).
4. Keep it simple. Many optimizations are possible, but simple changes often yield the largest benefits. I spent time implementing MPI, but batch‑size improvements had a much larger impact. Implementing adaptive batch sizing earlier would have saved time.

### Future Work

* Topology‑aware scheduling: distribute work based on GPU interconnects (e.g., NVLink vs. PCIe) and node topology to minimize transfer times.
* Asynchronous I/O: further overlap data loading/saving with computation using dedicated threads/processes or staged prefetching.
* Algorithmic levers: explore algorithmic changes that reduce memory footprint or computational complexity.
