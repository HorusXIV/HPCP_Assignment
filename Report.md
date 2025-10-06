HPC Assignment — Accelerating DEMREG
---
# Contents
- [Contents](#contents)
- [Introduction](#introduction)
- [Benchmarking](#benchmarking)
- [Implementation](#implementation)
  - [Accelerating with Dask](#accelerating-with-dask)
  - [Accelerating with Numba/CuPy](#accelerating-with-numba-and-cupy)
	- [Initial Improvement of Baseline](#initial-improvement-of-baseline)
	- [Transition to CuPy](#transition-to-cupy)
	- [Profiling and Observations](#profiling-and-observations)
	- [Conclusion](#conclusion)
  - [multiGPU Setup](#multigpu-setup)
    - [Baseline and First Principles](#baseline-and-first-principles)
    - [Multi‑GPU Distribution and Initial Improvements](#multigpu-distribution-and-initial-improvements)
    - [Profiling](#profiling)
    - [Batch Sizing](#batch-sizing)
    - [Overlapping Transfers and Compute \& Memory Pooling](#overlapping-transfers-and-compute--memory-pooling)
    - [Experiment: Triple vs. Double Buffering](#experiment-triple-vs-double-buffering)
      - [Hypotheses](#hypotheses)
      - [Measurements](#measurements)
      - [Hypothesis test summary](#hypothesis-test-summary)
      - [Conclusion](#conclusion-1)
    - [Improving Memory Stability](#improving-memory-stability)
    - [Experiment 2: Memory‑Handling Logic](#experiment-2-memoryhandling-logic)
      - [Hypotheses](#hypotheses-1)
      - [Measurements](#measurements-1)
      - [Hypothesis test summary](#hypothesis-test-summary-1)
      - [Conclusion](#conclusion-2)
- [Discussion](#discussion)
  - [Dask](#dask)
  - [Numba/CuPy](#numba/cupy)
    - [Lessons Learned](#lessons-learned)
    - [Future Work](#future-work)
  - [multiGPU](#multigpu)
    - [Lessons Learned](#lessons-learned-1)
    - [Future Work](#future-work-1)

# Introduction
This project is based on the [DEMREG codebase](https://github.com/ianan/demreg),  
a scientific application written in Python with heavy use of NumPy. The goal of the project is to utilize various solutions to improve runtime of the Demreg codebase.
Our approach is to initially benchmark the baseline implementation and then work on 3 approaches to speed up the runtime, the 3 implementations include:
- single GPU Numba/CuPy implementation
- single GPU Dask implementation
- multi GPU implementation utilizing CUDA/CuPy

# Methodology
For our Methodology we initially aim to benchmark the baseline code and improve the methods and calculations to utilize GPU Computational Power and Memory Throughput using the 3 previously mentioned implementations.

# Benchmarking
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia odio vitae vestibulum vestibulum. Cras venenatis euismod malesuada.

# Implementation
## Accelerating with Dask
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia odio vitae vestibulum vestibulum. Cras venenatis euismod malesuada.

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

### Baseline and First Principles

I began with the baseline NumPy implementation that was provided. As a first step, I replaced NumPy with CuPy and vectorized the hot paths to eliminate as many Python loops as possible. This converted the code from being loop‑bound to being GPU‑friendly. However, when I ran it on SLURM, GPU utilization was only around 14%, and overall it was even slower than the CPU version by a huge margin (9+ hours per image).

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

This was encouraging, but there was still a problem: out‑of‑memory (OOM) errors occurred frequently, especially with larger images or higher μ‑grid points. Each OOM triggered a retry with a smaller batch (reducing the batch size by half each time), but this wasted time and hurt throughput.

There were multiple approaches to improve this (for example, statically reducing the number of μ‑grid points), but I wanted to keep vendor parity as high as possible. So I focused on improving the memory‑handling logic. Previously, after each OOM, the batch size was simply halved. The new approach is more robust: flush reclaimable pool blocks before sizing so the estimate reflects real free memory; set the target fraction of free memory to 0.7 by default (tunable via the `MULTIGPU_BATCH_MEM_FRAC` environment variable) to leave headroom for library overhead.


### Experiment 2: Memory‑Handling Logic

The test setup matches the previous experiment: 3 rounds of 10 images each, measuring wall time. I discarded the first two rounds (warm‑up) and performed a one‑sided Welch’s t‑test on the remaining data. Again, all data were collected from NVIDIA Nsight reports.

#### Hypotheses
* **$$H_0$$:** The newest memory handling is **not faster** than the old version.
  ( $$\mu_\text{new} \ge \mu_\text{old}$$ )
* **$$H_1$$:** The newest memory handling is **faster** (lower mean time).
  ( $$\mu_\text{new} < \mu_\text{old}$$ )

We define the significance level α = 0.05.

#### Measurements

Time in seconds per image. Even though this is the never version rather than the last test, I didn't fix the batch size, which led to a higher processing time per Image. Additionally, you can see that some servers take much longer than others.

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

# Discussion

## Dask

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

With the implemented improvements, the final compute time per image averaged 24.58 ± 1.46 seconds. At this point, I reached diminishing returns: GPU utilization during compute time is now around 90–100% (per `nvidia-smi`), and the timeline shows a steady stream of kernels with brief gaps for saving/loading data.
Occasional OOMs still happen but are much rarer and GPU‑dependent. The adaptive batch sizing keeps memory usage high without frequent retries (at most 1× per image). Further improvements would likely require more complex changes, such as topology‑aware scheduling or algorithmic modifications. Given time constraints and the learning goals achieved, I did not pursue these further optimizations.

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


