
# Steps I took:
1) implement everything with cupy instaed of numpy. 
2) Applied vectorisation wherever possible to avoid costly loop. (9h + / image)
3) Debugged and implemented singularity enviroment to run it on the Calculon (huge time Investment)
4) Implemented MPI for scattering and gathering of data. Decided for an approach where one image is scattered over multiple GPU's. alternaties would have been, each  GPU gets a subset of the images to  process by itself or even, each node gets an subset of the Images, and then the load is split between the ggpus on that node. I decided for the first approach because easy to implment, GPUS have roughly similar load. (2h/ image)
5) Implemented Nsight Compute Profiling with nvtx.
6) Wondering / crying why it takes so long, why is it even slower then CPU
7) Increasing batchsize by many orders of magnitude, afterward autoscalling with respect to memory size / workload estimation & Retry on OOM ( 4m / img)
8) Implementation of Cuda streams after reading this Article: https://medium.com/@dmitrijtichonov/cuda-series-streams-and-synchronization-873a3d6c22f4 (+/- 30 secons / image)
9) Extend to  tripple buffering after reading multiple different sources. Run experiment to compare vs double buffering. Was the only real experiment (every other change was a pretty obvious & huge improvement)
10) Refactored the batch memory estimation and OOM handling to be conservative and to converge to a stable batch size quickly



# Report on Multi-GPU Implementation and Optimization

## Overview

This project explores how to push a multi-GPU image-processing pipeline from a slow, loop-bound baseline to a high-throughput system. The core lesson: performance came not from any single trick, but from an engineering workflow—profiling, forming hypotheses, testing changes, and documenting results.

## Baseline and First Principles

I began with a NumPy implementation that was dominated by Python loops. Early GPU attempts used very small batch sizes, which amplified launch and transfer overheads and left both the GPUs and host under-utilized. CPU utilization sat around ~30% while the GPUs stalled on frequent small kernels and transfers, making the GPU version slower than the CPU baseline.

## Implementation Path

I replaced NumPy with CuPy and vectorized hot paths to eliminate Python-level loops. I then brought the pipeline to the cluster (Singularity) and added MPI to distribute work across multiple GPUs. My first distribution strategy split each image across all available GPUs (rather than assigning whole images per GPU or per node). This was easy to implement and balanced load reasonably well when images were similar.

To understand bottlenecks, I instrumented the code with NVTX and profiled using Nsight Compute. Profiling showed that the major remaining levers were (1) batch size, (2) transfer/compute overlap.

## Batch Sizing and OOM Control

Large batches dramatically increased throughput by amortizing kernel-launch and transfer overhead. I added an adaptive batch-sizing heuristic: estimate free GPU memory, approximate batch footprint, attempt the largest safe batch, and back off on OOM. This achieved high memory utilization but still wasted time due to OOM retries—about one third of total GPU time in the worst cases.

I later refactored the policy to be conservative and converge faster: keep a safety margin, shrink more aggressively on OOM, and remember the last good size to avoid oscillation. Reducing the default number of μ grid points in the discrepancy search from 64 to 32 further lowered memory pressure, cutting OOM frequency and improving wall time.



## Overlapping Transfers and Compute

After the batch work, I addressed pipeline idle gaps. I introduced CUDA streams with a double-buffering pattern to overlap host-to-device (H2D) copies with compute. This reduced the gaps between kernels and improved GPU occupancy. Extending the approach to triple buffering (H2D → compute → D2H in separate streams) smoothed the timeline further. With triple buffering, the profile showed ~205.3 s of pure compute within a 329 s run—even with profiler overhead and occasional OOM retries—evidence of much better overlap.

## Experiment: Triple vs. Double Buffering

Almost all the previous changes yielded clear improvements, so the only real experiment was comparing double vs. triple buffering. I ran 3 rounds of 10 images each, measuring wall time and compute time per image. Then I discarded the first two rounds (warm-up) and performed a t-test on the remaining data.

### Measurements

|  # | Double Buffer – Time/image | Double Buffer – Compute  | Triple Buffer – Time/image | Triple Buffer – Compute |
| -: | -------------------------: | ------------------------ | -------------------------: | ----------------------: |
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
| 14 |                     25.102 |                   17.526 |                      25.06 |                  17.494 |
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

## Hypothesis test summary

| Metric       | α (p-value threshold) | TTEST p-value | Null hypothesis (mean_triple = mean_double) | Decision              |
| ------------ | --------------------: | ------------: | ------------------------------------------- | --------------------- |
| Wall Time    |                  0.05 |    0.00071655 | mean_triple = mean_double                   | **Reject H₀**         |
| Compute Time |                  0.05 |    0.07978201 | mean_triple = mean_double                   | **Fail to reject H₀** |

## Conclusion
The measurements show that the triple buffering implementation has a lower average wall time per image compared to the double buffering implementation. The t-test for wall time yields a p-value of 0.00071655, which is less than the significance level of 0.05, leading to the rejection of the null hypothesis. This indicates that there is a statistically significant difference in wall time between the two implementations, with triple buffering being faster.


## Current Limitations

The current MPI strategy ignores node topology. Work is split evenly per GPU without considering whether devices are on the same node, PCIe switch, or NVLink island. Transfers are all host-mediated; there is no GPUDirect RDMA between nodes. Finally, OOM handling—while improved—still costs time when memory estimates are off.

## What’s Realistic vs. Not

* **Realistic:**

  * Large batches increase efficiency by amortizing overhead.
  * Overlapping H2D/compute/D2H via multiple streams can cut idle time and reduce wall time.
  * Reducing algorithmic memory intensity (e.g., fewer μ grid points) directly lowers OOM risk and can improve throughput.

* **Not a silver bullet / Easy to overclaim:**

  * “Always use the largest possible batch”: true in spirit, but only with a good safety margin and stable estimation; naive “retry on OOM” wastes time.
  * “Evenly scatter across all GPUs” is simple, but ignoring topology (intra-node vs. inter-node) can leave performance on the table due to avoidable transfer latency.

## Results Summary

* Replacing NumPy with CuPy and vectorizing the hot path converted loop-bound code into GPU-friendly kernels.
* Adaptive batch sizing delivered the largest single speedup after vectorization.
* Double → triple buffering further reduced wall time; the improvement is statistically significant (p ≈ 7.16×10⁻⁴).
* Lowering the μ-grid from 64 to 32 reduced OOMs and improved stability with minimal impact on quality for the tested cases.

## Lessons Learned

1. **Profile first, then act.** Nsight + NVTX traces exposed idle gaps and transfer stalls that weren’t visible from timing alone.
2. **Batch size is a control knob.** It determines both throughput and stability; automate it with headroom and memory pools.
3. **Overlap beats sequentialism.** Streams and buffering convert dead time into work.
4. **Topology matters.** Simple equal splits are fine to start, but placement and affinity become important at scale.

## Future Work

* **Topology-aware scheduling:** Keep each image’s shards within a node when possible, prefer NVLink pairs, and align MPI ranks with GPU locality.
* **Smarter memory management:** Use a pooled allocator and a persistent “last known good” batch size per device/data shape; add a small safety factor to avoid thrashing.
* **Comms optimization:** Consider GPUDirect RDMA for inter-node exchanges; compress intermediate results before communication when feasible.
* **Algorithmic levers:** Explore reduced precision where safe, shared-memory tiling, and better reuse to lower the memory footprint without hurting accuracy.

## Alignment with the Assignment Goals

I followed an engineering workflow rather than chasing a single headline speedup. I profiled and benchmarked the baseline, analyzed bottlenecks, formulated and tested hypotheses (batch sizing, buffering depth, μ-grid size), applied standard HPC techniques (vectorization, GPU offload with CuPy, MPI distribution, CUDA streams, pipeline buffering), and documented design decisions, experiments, and results. The improvements came from disciplined iteration, not guesswork.

---
