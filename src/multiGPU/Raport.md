
# Report on Multi-GPU Implementation and Optimization

## Baseline and First Principles

I began with  the baseline NumPy implementation that was provided to us. As a first step I replaced NumPy with CuPy and vectorized the hot paths to eliminate as much Python loops as possible. This converted the code from being loop-bound to being GPU-friendly. However, when i let it run on the slurm, GPU utilisation was only arround 14% and overall it was even slower than the CPU version by hughe margin (9h + / image).

To let it run on the cluster, I created a singularity environment. This was a huge time investment because of the many dependencies and the fact that I had not much prior experience with singularity. After matching CuPy and CUDA versions, as well as installing some additional librarys to avoid compilation errors, I was able to run the code on the cluster.

## Multi-GPU Distribution & First Improvements

Next, I added MPI to distribute work across multiple GPUs. I decided for a strategy where each image is scattered across all available GPUs. This was easy to implement and balanced load reasonably well when images were similar. Alternatives would have been to distribute a subset of of images to each GPU, but if not all images are similar in size, this could lead to load imbalance. Another alternative would have been to distribute a subset of images to each node, and then split the work between the GPUs on that node. I decided against this approach because it would have been way more complex to implement. With this multi-GPU and some other minor tweaks in the code, I was able to bring the time down to about 2h / image.

## Profiling

Now I was on the limit of obvious improvements, so it was time to profile. I used Nsight Compute with NVTX annotations to get a detailed timeline of GPU activity. I then ran a profiling on each GPU, but because I expected them to behave similarly, I only analyzed one in detail (Usually GPU 0). Additionally, I used `nvidia-smi` & `nvtop` on the computing server to monitor memory usage and GPU utilization in real time.

This lead to two key insights: First, even with the improvements so far, GPU utilization was still low (around 30-35%), and there were significant breaks between this times of "high" utilisation where GPU-Utilisation dropped significantly. 

## Batch Sizing

Because of these Insights, I was able to implement the biggest Improvement of this whole Project: Large batches dramatically increased throughput by amortizing kernel-launch and transfer overhead. 
I added an adaptive batch-sizing heuristic: estimate free GPU memory, approximate batch footprint, attempt the largest safe batch, and back off on OOM. This achieved high memory utilization but still wasted time due to OOM retries—about one third of total GPU time in the worst cases. Overall this brought the time down to about 4m / image.

However this was due to a bug in my kernel logic, where I accidentally did some operations with dn ** 2 twice. This lead to a compute complexity of O(n^2) instead of the intended O(n), which made the algorithm much more memory intensive. After fixing this bug, I was able to run with much larger batches and the time dropped to about 35s / image.

## Overlapping Transfers and Compute & Memory pooling

After the batch work, I addressed pipeline idle gaps. I introduced CUDA streams with a double-buffering pattern to overlap host-to-device (H2D) copies with compute. Additionally, I implemented a memory pool to avoid repeated allocation overhead. This reduced the gaps between kernels and improved GPU occupancy. The profile the now showed ~205.3 s of pure compute within a 334 s run, even with profiler overhead and occasional OOM retries.

Extending the approach to triple buffering (H2D → compute → Copy in separate streams) should've smoothed out the pipeline even more. To verify this, I implemented it and ran an experiment to compare double vs. triple buffering.

### Experiment: Triple vs. Double Buffering

Almost all the previous changes yielded clear improvements, so the only real experiment was comparing double vs. triple buffering. I ran 3 rounds of 10 images each, measuring wall time and compute time per image. Then I discarded the first two rounds (warm-up) and performed a t-test on the remaining data. All the Data is collected from NVIDIA Nsight Compute reports.
To avoid additional variability, I kept the batch size constant at 198759 for both implementations. Another thing to improve test reliability would have been to fix the SLURM-Server allocation to a specific node, but because I don't really know how good this works on our cluster, I decided against it. 

#### Measurements

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

| Metric       | α (p-value threshold) | TTEST p-value | Null hypothesis (mean_triple = mean_double) | Decision              |
| ------------ | --------------------: | ------------: | ------------------------------------------- | --------------------- |
| Wall Time    |                  0.05 |    0.00071655 | mean_triple = mean_double                   | **Reject H₀**         |
| Compute Time |                  0.05 |    0.07978201 | mean_triple = mean_double                   | **Fail to reject H₀** |

#### Conclusion
The measurements show that the triple buffering implementation has a lower average wall time per image compared to the double buffering implementation. The t-test for wall time yields a p-value of 0.00071655, which is less than the significance level of 0.05, leading to the rejection of the null hypothesis. This indicates that there is a statistically significant difference in wall time between the two implementations, with triple buffering being faster.

However, the compute times between the two implementations are not significantly different, as indicated by the p-value of 0.07978201, which is greater than 0.05. Therefore, we fail to reject the null hypothesis for compute time, suggesting that the compute performance is similar for both buffering strategies.

## Improving Memory Stability

This all was great, but there still was a problem: Out-of-memory (OOM) errors still occurred frequently, especially with larger images or higher μ grid points. Each OOM triggered a retry with a smaller batch (reducing the batch size by half each time), but this wasted time and hurt throughput. 

There were multiple approaches to improve this, for example to statically reduce number of μ grid points, but I wanted to keep Vendor Parity as high as possible. So I focused on improving the memory handling logic, so I refactored the policy to be conservative and converge faster: keep a safety margin, shrink more aggressively on OOM, and remember the last good size to avoid oscillation.


### Experiment 2: Memory Handling Logic

The Test Setup is the same as for the Last experiment: 3 rounds of 10 images each, measuring wall time. Then I discarded the first two rounds (warm-up) and performed a One Sided, heteroskedastic t-test on the remaining data. Again, all the Data is collected from NVIDIA Nsight Compute reports.

#### Measurements

|  # | Old version | New version |
| -: | ----------: | ----------: |
|  1 |       29.27 |       25.62 |
|  2 |       29.50 |       26.19 |
|  3 |       28.20 |       25.94 |
|  4 |       28.91 |       26.32 |
|  5 |       29.06 |       26.04 |
|  6 |       28.91 |       26.32 |
|  7 |       28.61 |       26.14 |
|  8 |       29.27 |       27.06 |
|  9 |       28.00 |       21.17 |
| 10 |       28.80 |       23.14 |
| 11 |       28.19 |       22.65 |
| 12 |       28.70 |       23.19 |
| 13 |       28.57 |       23.41 |
| 14 |       27.60 |       23.96 |
| 15 |       28.76 |       23.34 |
| 16 |       29.85 |       22.91 |
| 17 |       23.36 |       23.88 |
| 18 |       25.01 |       23.89 |
| 19 |       25.08 |       22.98 |
| 20 |       26.14 |       27.67 |
| 21 |       26.21 |       23.01 |
| 22 |       26.21 |       23.52 |
| 23 |       25.29 |       23.75 |
| 24 |       25.40 |       23.58 |

#### Hypothesis test summary

| Metric           |                       Value |
| ---------------- | --------------------------: |
| α (significance) |                        0.05 |
| TTEST p-value    |                 3.58787e-08 |
| H₀               | Runtime(old) = Runtime(new) |
| H₁               | Runtime(old) < Runtime(new) |
| Decision         |               **Reject H₀** |

#### Conclusion
The measurements show that the new memory handling logic implementation has a lower average wall time per image compared to the old implementation. The t-test yields a p-value of 3.58787e-08, which is significantly less than the significance level of 0.05, leading to the rejection of the null hypothesis. This indicates that there is a statistically significant difference in wall time between the two implementations, with the new memory handling logic being faster.

## Summary

With all those improvements the final compute time per image was on average 24.40333333 ± 1.68542516 seconds. At this point I reach diminishing returns: GPU utilization is now around 90-100% during compute time(according to nvidia-smi), and the timeline shows a more or less steady stream of kernels only with gaps for saving / loading the Data. 
Occasional OOMs still happen but are rare, and the adaptive batch sizing keeps memory usage high without frequent retries (1x per Image). Further improvements would require more complex changes, such as topology-aware scheduling or algorithmic modifications. However, as I have already spent way more time than expected on this project and I learned already quite some things, I will not pursue these further possible optimizations.

I also noticed that it really depends on which server you let the program run. If I let the current ( and final version) run e.g. on Server0092, I'm almost 4 seconds slower per image than on Server0101. This of course had also some impact on telling if an improvement was real or just noise, especially at the point where improvements were only in the range of 2-4 seconds. That's where I also started to do the real experiments with statistical testing, as before it would have taken too much time to do this for every single change.

### Lessons Learned

1. **Profiling saves lives** Nsight + NVTX traces show things that you can't just see otherwise, also live monitoring with nvidia-smi / nvtop is very useful, to know where to improve in the future.
2. **Batch size on top** My Biggest improvements had all to do with optimizing batch size. Bigger batches amortize overhead and increase throughput, but require careful memory management, or else you have much wasted time on OOM retries.
3. **Overlap matters** Streams and buffering convert dead time into work. However if you want go full parallelism, including loading the data while still computing, you have to watch out for race conditions and memory failures. Even with the current approach, there is still some idle time while loading / saving the data, but getting this right would have required way more time spend on parallel computing, and frankly, I don't really know if Python is the right tool for this.
4. **KISS (Keep it simple, stupid)** Many optimizations are possible, but with the simplest you can gain the most. I spent a lot of time on implementing MPI, but in the end, this was not really necessary, because the batch size improvements were way more important. Also I could have saved a lot of time if I would have implemented the adaptive batch sizing earlier.

### Future Improvement possibilities

* **Topology-aware scheduling:** Distribute work based on GPU interconnects (e.g., NVLink vs. PCIe) to minimize data transfer times.
* **Asynchronous I/O:** Further overlap data loading/saving with computation using dedicated threads or processes.
* **Algorithmic levers:** Explore algorithmic changes that reduce memory footprint or computational complexity.
---