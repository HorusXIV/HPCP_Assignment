## Ground rules (to keep results comparable)

- Fix inputs & params: same STACK (or same tiles), same T_RESP/T_RESP_LOGT/TEMPS, same nmu, same dtype (pick float32 or float64 and stick to it).
- Warm-up once (JITs, caches, GPU clocks), then measure N=5 runs → report median.
- Thread caps for CPU baselines: OMP/OPENBLAS/MKL/NUMEXPR=1.
- Correctness check: after each variant, compare outputs to baseline, e.g. max_rel_err on demmap (or on mean/peak maps) with a small tolerance.
- Report both: (a) end-to-end time and (b) steady-state throughput DEMs/s = (H×W)/time.

## Metrics to report (all variants)

- Throughput: DEMs/s and time per DEM (μs/px).
- Speedup: vs single-core baseline.
- Parallel efficiency: speedup / #workers (Dask & multi-GPU).
- CPU/GPU util: avg utilization during the timed region.
- Transfer ratio (GPU): H2D+D2H time / total.
- Peak RSS / GPU mem: max memory used.

##A) Baseline CPU (vanilla dn2dem_pos)

Tools

- line_profiler (small crops): pinpoint hotspots (dem_pix, etc.).
- cProfile + SnakeViz (small/medium): call graph, cumulative time.
- py-spy or Scalene (medium/large): low-overhead sampling, memory.

What to capture

- For crops: 14×14 (serial path), 64×64, 256×256, and one large (e.g., 1024×1024).
- DEMs/s, hot functions, and where linear algebra costs appear.

## B) Dask parallelization (CPU scaling)

Tools

- Dask Dashboard (Task Stream, Graph, Workers).
- performance_report (single HTML per run).
- Optional: ResourceProfiler, CacheProfiler (from dask.diagnostics).

How to measure

- Tile your image (e.g., 256×256 tiles), map dn2dem_pos per tile (or per row block).
- Fix workers/threads (e.g., n_workers=physical_cores, threads_per_worker=1).
- Produce two studies:
  - Strong scaling: fixed global size, vary workers (1 → 2 → 4 → 8 …).
  - Weak scaling: fixed work per worker, grow workers & problem together.

What to capture

- End-to-end time and DEMs/s.
- Dask overhead fraction (from performance report), serialization time, spill occurrences.
- Speedup and efficiency vs number of workers.

*Minimal report code:*

```python
from dask.distributed import Client, performance_report
from datetime import datetime
with performance_report(filename=f"report_{datetime.now().isoformat()}.html"):
    # submit graph and wait; time with perf_counter around client.compute()
    ...
```

## C) Single-GPU SIMD

> Use CuPy/Numba/PyTorch for kernels; when you do, profile like this.

Tools

- Nsight Systems (nsys) = end-to-end timeline (kernels, memcpy, CPU).
- Nsight Compute (ncu) = per-kernel metrics (achieved occupancy, DRAM BW, FLOPs, roofline).
- NVTX ranges in Python to mark phases (preproc, H2D, kernels, D2H, post).

How to measure

- Time only the steady compute with synchronization:
    ```python
    start = cp.cuda.Event(); end = cp.cuda.Event()
    start.record()
    # kernels / ops ...
    end.record(); end.synchronize()
    ms = cp.cuda.get_elapsed_time(start, end)
    ```
- Also report end-to-end (including H2D/D2H).

Commands
- Systems (timeline):  
  `nsys profile -t cuda,nvtx,osrt -o runs/single_gpu python your_script.py`
- Compute (kernel deep dive):  
  `ncu --set full --target-processes all -o runs/single_gpu_ncu python your_script.py`

What to capture

- GPU util %, DRAM throughput, SM occupancy, kernel time share.
- Transfer/compute ratio; aim to overlap transfers if needed.

## D) Multi-GPU SIMD

Setup

- Dask-CUDA (or Ray) to shard tiles across GPUs. UCX for NVLink/IB if available.

Tools

- Nsight Systems again, but launched on the driver script; captures multiple GPUs.
- Dask dashboard (task placement per GPU, comms time).
- Optional: DCGM or nvidia-smi dmon for per-GPU util/mem/power.

What to capture

- Strong scaling over 1→N GPUs on a fixed global size.
- Efficiency vs GPUs, comms/serialization overheads, load balance.

## Reference harness (uniform across variants)

```python
import time, numpy as np

def time_dn2dem(frame_6hw, T_RESP, T_RESP_LOGT, TEMPS, nmu=42):
    f = np.moveaxis(frame_6hw, 0, -1).astype(np.float32, copy=False)
    f = np.clip(np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0), 0, None)
    e = np.sqrt(f) + 1e-6
    t0 = time.perf_counter()
    demmap, edemmap, logT_bins, chisq, dn_reg = dn2dem_pos(f, e, T_RESP, T_RESP_LOGT, TEMPS, nmu=nmu)
    dt = time.perf_counter() - t0
    dems = (f.shape[0]*f.shape[1]) / dt
    return dt, dems
```
Use the same wrapper for CPU, Dask (per tile aggregated), and GPU (wrap around the GPU call; for GPU also report device-timed kernels via events).

## Reporting template (fill once per run)

| Variant          | Size (H×W) | Tiles | Workers/GPUs | nmu | dtype | Time (s) | DEMs/s | Speedup | Eff. | CPU% / GPU% | H2D+D2H % | Peak Mem |
|------------------|------------|-------|--------------|-----|-------|----------|--------|---------|------|-------------|-----------|----------|
| CPU single-core  | 1024×1024  | –     | 1 / –        | 42  | f32   | …        | …      | 1.0×    | 100% | 100% / –    | –         | …        |
| Dask 8 workers   | 1024×1024  | 256²  | 8 / –        | 42  | f32   | …        | …      | …       | …    | 800% / –    | –         | …        |
| Single GPU       | 1024×1024  | 256²  | – / 1        | 42  | f32   | …        | …      | …       | –    | – / 95%     | 12%       | …        |
| 2 GPUs           | 2048×2048  | 256²  | – / 2        | 42  | f32   | …        | …      | …       | …    | – / 2×95%   | 15%       | …        |


Add one roofline screenshot from ncu (single GPU) and one Dask performance report (multi-CPU/GPU) to back up claims.