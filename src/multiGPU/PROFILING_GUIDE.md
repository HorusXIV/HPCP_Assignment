# Profiling and Performance Testing Guide

## Fixed Issues

### 1. Nsight Systems Profiling Error
**Problem:** The script was using `--tempdir` option which doesn't exist in nsys profile command.

**Solution:** 
- Removed invalid `--tempdir` option
- Use `TMPDIR` environment variable for temporary files
- Added comprehensive profiling options for GPU kernel analysis

## Running Performance Tests

### Basic Performance Test
```bash
# Run without profiling (fastest)
sbatch hpc/slurm_run_multiGPU.sh
```

### With Nsight Systems Profiling
```bash
# Enable profiling with enhanced GPU metrics
PROFILE=1 MULTIGPU_NVTX=1 sbatch hpc/slurm_run_multiGPU.sh
```

### Environment Variables for Testing

#### Performance Optimization Control
```bash
MULTIGPU_BATCH_SIZE=0           # 0=adaptive, >0=override
MULTIGPU_STABLE_PINV=1          # Enable stable pseudo-inverse for large images
MULTIGPU_KEEP_DEVICE=1          # Keep arrays on device
MULTIGPU_VECTOR_DISABLE=0       # Disable vectorized optimizations (testing)
```

#### Profiling Control
```bash
PROFILE=1                       # Enable Nsight Systems profiling
MULTIGPU_NVTX=1                # Enable detailed NVTX annotations
NSYS_OPTS="cuda,nvtx,osrt,cublas,cusolver"  # Profiling traces to collect
```

## Performance Comparison

### Before and After Optimization Test
1. **Baseline (create separate branch):**
   ```bash
   git checkout -b baseline
   git revert <commit-hash-of-optimizations>
   PROFILE=1 sbatch hpc/slurm_run_multiGPU.sh
   ```

2. **Optimized version:**
   ```bash
   git checkout f/improving_multiGPU
   PROFILE=1 MULTIGPU_NVTX=1 sbatch hpc/slurm_run_multiGPU.sh
   ```

3. **Compare results:**
   - Check processing times in logs
   - Compare `.nsys-rep` files in Nsight Systems GUI
   - Look for memory usage improvements

### Key Metrics to Monitor

#### In Logs
```bash
grep -E "(Processing|processed|GPU memory pool)" src/multiGPU/results/logs/*.out
```

#### In Nsight Systems
- **GPU Utilization:** Should increase with optimizations
- **Memory Usage:** Should show better patterns with memory pools
- **Kernel Launch Overhead:** Should decrease with kernel fusion
- **SVD Performance:** Should be significantly faster with economic SVD

### Expected Improvements

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| SVD Time | 100% | 60-80% | 20-40% faster |
| Memory Allocations | High fragmentation | Better reuse | Fewer allocations |
| GPU Utilization | Variable | More consistent | Better occupancy |
| Total Runtime | 100% | 40-70% | 30-60% faster |

## Troubleshooting

### Common Issues

1. **"nsys not found in container"**
   - Set `PROFILE=0` to disable profiling
   - Or install Nsight Systems in container

2. **Out of memory errors**
   - Reduce `MULTIGPU_BATCH_SIZE`
   - Set `MULTIGPU_STABLE_PINV=1` for large images

3. **Slow performance**
   - Check `MULTIGPU_VECTOR_DISABLE=0` (should be 0)
   - Verify GPU assignment in logs
   - Check memory pool utilization

### Debug Mode
```bash
# Enable all debugging
PROFILE=1 MULTIGPU_NVTX=1 MULTIGPU_STABLE_PINV=1 sbatch hpc/slurm_run_multiGPU.sh
```

## File Locations

- **Logs:** `src/multiGPU/results/logs/`
- **Profiling:** `src/multiGPU/results/nsys/`
- **Results:** `src/multiGPU/results/aggregate/`

## Next Steps

After validating Phase 1 improvements, proceed to Phase 2:
- CUDA streams implementation
- Asynchronous data transfers  
- Multi-stream processing for very large datasets