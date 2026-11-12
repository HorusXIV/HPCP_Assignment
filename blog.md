# Accelerating a Solar Physics Solver Across GPUs

DEMREG reconstructs differential emission measure (DEM) maps from multi-band solar imagery: six-channel frames of the sun at 4096×4096, solved tile by tile into a picture of how much plasma is at each temperature along the line of sight. The [vendor implementation](https://github.com/ianan/demreg) is NumPy-heavy and correct, but slow. On our team's dataset, a single frame took 37 minutes on 16 CPU cores.

This is the story of getting that down to 29 seconds, one bottleneck at a time: CPU parallelism with Dask, a single GPU with CuPy, then four GPUs with CuPy and MPI. The baseline, Dask, and single-GPU stages were built by teammates; the multi-GPU stage, and most of what follows, was mine.

## The Baseline: 37 Minutes, 16 Cores

The starting point stayed close to the vendor code: same tiling (256×256), same solver settings, wrapped behind a thin API so every later variant could call it identically. On 16 CPU cores, one 4096×4096, six-band frame took 2204 seconds, give or take 20 seconds across runs. That number became the yardstick for everything after it.

## Scaling Out on CPU with Dask

The next step kept the same CPU solver but exposed it to Dask, chunking each frame into tile-aligned blocks and mapping the solver over them. On 32 cores, that ran in 974 seconds: a 2.26× speedup.

Except it used twice the CPU cores. Normalizing for that (scaling Dask's time to what 16 cores would need) drops the real gain to 1.13×. Most of the raw speedup came from throwing more hardware at the problem, not from Dask making each core more efficient. That's a useful number to report honestly: a 2.26× headline is misleading if it costs 2× the resources to get there.

## One GPU: Numba, CuPy, and a Turf War Between Two CUDA Contexts

The first attempt at GPU acceleration added Numba's `@jit` and `@cuda.jit` decorators to the hottest loops. It didn't help much: runtime stayed around 2 hours, GPU utilization sat below 10%, and the tiles still ran serially with constant round trips between host and device memory. Numba sped up the inner loop; everything around it stayed the bottleneck.

The next attempt kept Numba for compute but moved memory management to CuPy. That combination produced `CUDA_ERROR_INVALID_CONTEXT` crashes: Numba and CuPy each manage their own CUDA context, and mixing them on the same device is fragile. When it did run, it took about 40 minutes per image.

The fix was to drop Numba entirely and rewrite the solver in pure CuPy, vectorized end to end and built on CUBLAS and CUB. That took runtime to about 4.5 minutes per image, with GPU utilization in the 30-50% range: healthy memory throughput, with the solver's own numerical complexity as the remaining ceiling.

## Going Multi-GPU

### First Pass: Slower Than the CPU

I started from that CuPy implementation and pointed it at four GPUs, splitting each image's pixels evenly across them with MPI (mpi4py). The problem is embarrassingly parallel at the pixel level, so this was a natural first cut. It also needed a Singularity container to run on the FHNW SLURM cluster, matching CuPy and CUDA versions and chasing down a handful of missing libraries.

The first working version was worse than the CPU baseline: about 14% GPU utilization and 9+ hours per image. Getting the code onto four GPUs hadn't made it fast; it had just moved the same inefficiency onto more hardware. A round of cleanup and minor fixes brought that down to about 2 hours per image, still far from usable.

### Profiling Finds the Real Bottleneck

With the obvious fixes exhausted, I turned to NVIDIA Nsight Systems with NVTX annotations, checking each GPU briefly and one in detail, backed by live monitoring with `nvidia-smi` and `nvtop`. The timeline showed two things: GPU utilization was still only 30-35%, and there were long idle gaps between bursts of compute.

### The Biggest Lever: Batch Size

Small batches meant kernel launch and transfer overhead dominated actual compute. An adaptive batch-sizing heuristic fixed that: estimate free GPU memory, try the largest batch that should fit, and back off on out-of-memory errors. That alone brought runtime to about 4 minutes per image, though OOM retries still wasted up to a third of GPU time in the worst cases.

Digging into why memory pressure was so high turned up a bug: a term involving `dn**2` was applied twice in the kernel, silently turning an O(n) computation into O(n²), along with a similar mistake in the write path. Fixing both let batches grow much larger, and runtime dropped to about 35 seconds per image. That one bug fix moved the needle more than the multi-GPU distribution itself.

### Overlap, Memory Pooling, and Proving It With Statistics

Next came CUDA streams with double buffering, overlapping host-to-device transfers with compute, plus a memory pool to skip repeated allocation overhead. A profiled run showed about 205 seconds of pure compute inside a 334-second wall time, even accounting for profiler overhead and occasional OOM retries.

Triple buffering, adding a separate stream for the device-to-host copy, looked like it should smooth the pipeline further. Rather than assume it, I ran a controlled experiment: fixed batch size, three rounds of ten images, discard the first two rounds as warm-up, one-sided Welch's t-test on what's left. Triple buffering won on both wall time (p = 0.0002) and compute time (p = 0.049), a real if modest gain, not noise from a single lucky run.

The same rigor went into the batch-recovery logic. The original approach halved the batch size after every OOM; the replacement flushes reclaimable memory pool blocks before resizing and targets 70% of free memory instead of guessing. A second Welch's t-test confirmed it: p = 3.7e-8, decisively faster.

### A Silent Correctness Bug Behind Good Numbers

A sanity check against independent reference data caught something the timing numbers never would have: DEM values around 10^80. The cause was a placeholder response matrix, all ones instead of realistic per-wavelength responses, which made the inversion nearly singular. The solver's discrepancy principle responded by picking a tiny regularization parameter and amplifying noise into nonsense.

The fix was a proper response matrix built from realistic synthetic responses mapped onto the DEM grid, plus a floor on the regularization parameter to stop it from collapsing again. Restoring correctness cost some speed back, since results now get gathered at the root MPI rank, pushing runtime to about 36 seconds per image. A fast wrong answer isn't a result worth reporting.

### Closing the Gap

The last change overlapped I/O with compute: while one image saves, the next loads and starts processing. It doesn't save compute time, but it keeps the GPUs busy instead of idle, bringing the final average down to about 30 seconds per image, measured at 28.99 ± 1.74 seconds with GPU utilization in the 90-100% range during compute.

## The Full Picture

| Method | Wallclock (s/frame) | Speedup vs. baseline |
|---|---:|---:|
| Baseline (CPU, 16 cores) | 2204.0 | 1× |
| Dask (CPU, 32 cores) | 974.0 | 2.26× (1.13× at equal cores) |
| Single-GPU (CuPy) | 35.0 | 62.97× |
| Multi-GPU (CuPy + MPI, 4 GPUs) | 28.99 | 76.03× |

## What Actually Moved the Needle

A few things held up across the whole multi-GPU stretch:

- **Profile before optimizing.** Nsight Systems and `nvidia-smi` caught idle gaps and low utilization that were invisible from wall-clock time alone.
- **Batch size beat parallelism.** Distributing across four GPUs was necessary, but the adaptive batch-sizing fix was the single largest improvement in the whole project, bigger than going from one GPU to four.
- **Overlap turns idle time into throughput**, but only if you check for it. Both buffering changes went through a proper hypothesis test rather than a before/after eyeball comparison, because the effect sizes (2-4 seconds) were close to the noise floor of node-to-node hardware variance: the same code ran about 4 seconds slower per image on a node with RTX 2080 Tis than one with RTX 3080 Tis.
- **Simple changes beat clever ones.** Implementing MPI distribution took real effort; the batch-size heuristic took less and mattered more. Doing that first would have saved time.

## Where the Numbers Live

The full experiment data and the Welch's t-test analysis behind both buffering decisions sit in the project's stats notebook, for anyone who wants to check the raw measurements rather than take the p-values on faith.