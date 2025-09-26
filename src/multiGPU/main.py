"""Entry point for launching multi-GPU DEM computations with explicit MPI.

This script demonstrates how to initialize MPI, bind GPUs to ranks and run a
GPU-accelerated kernel over input data. It intentionally keeps dependencies
light so it can be executed on development machines as a serial script.
"""
from __future__ import annotations

import argparse
import logging
import numpy as np
import os
import glob
import time
from src.common.nvtx import nvtx_range
from . import io as mio
from . import mpi_manager as mmpi
from . import gpu_kernels
from . import logging as mlog


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input-dir",
        default=None,
        help=(
            "directory to scan for .npz input files; rank 0 will enumerate "
            "and distribute them"
        ),
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "maximum number of spatial samples (pixels) to process"
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()

    with nvtx_range("INIT_MPI", color=0x4caf50):
        comm, rank, size = mmpi.init_mpi()
    # Setup results directory structure
    results_root = "src/multiGPU/results_Test"
    # Results will be written into the shared `aggregate` folder under
    # `results_root` (no per-rank subdirectories).
    if rank == 0:
        # ensure parent exists
        os.makedirs(results_root, exist_ok=True)

    # Initialize logging (creates logs/ under results_root)
    _ = mlog.setup_logging(results_root, rank=rank, size=size)
    log = logging.getLogger(__name__)
    log.info(f"Starting rank {rank}/{size-1}; results dir: {results_root}")

    # Map rank to GPU and bind environment (prefer per-node local-rank mapping)
    with nvtx_range("RANK_GPU_BIND", color=0x009688):
        local_rank, local_size, node = mmpi.get_local_rank_info(comm)
        gpu_assigned = mmpi.set_device_for_local_rank(comm)
    # Startup banner
    log.info(
        "MPI size=%d, node=%s, local_size=%d, gpu_assigned=%s",
        size,
        node,
        local_size,
        str(gpu_assigned),
    )

    # Decide input list: either single file or enumerate a directory

    if not args.input_dir:
        if rank == 0:
            raise RuntimeError(
                "Must specify --input-dir to process input files"
            )
    else:
        # Rank 0 enumerates files and distributes the list to all ranks
        with nvtx_range("ENUM_INPUTS", color=0x2196f3):
            if rank == 0:
                pattern = os.path.join(args.input_dir, "*.npz")
                all_inputs = sorted(glob.glob(pattern))
                if len(all_inputs) == 0:
                    raise RuntimeError(
                        f"No .npz files found in {args.input_dir}"
                    )
            else:
                all_inputs = None

        # Broadcast the list of inputs to all ranks. The serial stub will get
        # None which is handled below.
        with nvtx_range("BCAST_INPUTS", color=0x1976d2):
            if comm is not None:
                all_inputs = comm.bcast(all_inputs, root=0)
            else:
                # non-MPI fallback: enumerate locally
                pattern = os.path.join(args.input_dir, "*.npz")
                all_inputs = sorted(glob.glob(pattern))

        # We will iterate all inputs on every rank but let rank 0 load the
        # data and scatter rows to workers. This guarantees every MPI
        # collective (Scatterv/Gatherv) is invoked by all ranks and that
        # rank 0 can reliably save a full `dem_all` file for each input.
        n_files = len(all_inputs)
        if rank == 0:
            print(
                "Processing %d input files across %d ranks" % (n_files, size)
            )

    # All ranks iterate the global input list; rank 0 will load each file
    # and participate in the scatter/gather protocol so the full result
    # is created and saved for every input.
    # Ensure "all_inputs" is defined (it will be if --input-dir provided)
    if not args.input_dir:
        return

    for input_path in all_inputs:
        # High-level per-file range (covers full processing path for file)
        file_label = f"PROCESS_FILE:{os.path.basename(input_path)}"
        with nvtx_range(file_label, color=0xff9800):
            # Announce input start and measure elapsed time
            # Mark as general so it appears on console; restrict to rank 0
            if rank == 0:
                log.info(
                    f"Starting processing input {input_path}",
                    extra={"general": True},
                )
            try:
                # Rank 0 loads and prepares the arrays; other ranks wait to
                # participate in the collective scatterv. This ensures a
                # consistent collective ordering across ranks.
                # Initialize shared variables so they exist on all ranks
                # prior to broadcast
                counts = None
                dn2d = None
                edn2d = None
                n_samples = None
                if rank == 0:
                    with nvtx_range("LOAD_FILE", color=0x8e24aa):
                        data = mio.load_npz(input_path)
                    # Heuristics: prefer `dn`/`edn`; otherwise `bands` layout.
                    dn = data.get("dn", None)
                    edn = data.get("edn", None)
                    if dn is None:
                        if "bands" in data:
                            bands = data["bands"]
                            if bands.ndim != 3:
                                raise RuntimeError(
                                    "Unexpected `bands` shape; expected "
                                    "(nf, ny, nx)"
                                )
                            nf, ny, nx = bands.shape
                            n_pixels = ny * nx
                            max_samples = args.max_samples
                            bands_flat = bands.reshape(nf, n_pixels)
                            if (
                                max_samples is not None
                                and n_pixels > max_samples
                            ):
                                idx = np.linspace(
                                    0, n_pixels - 1, num=max_samples, dtype=int
                                )
                                dn2d = bands_flat[:, idx].T
                            else:
                                dn2d = bands_flat.T
                            edn2d = np.ones((dn2d.shape[0], nf), dtype=float)
                            dn = dn2d
                            edn = edn2d
                        else:
                            arrays = [v for v in data.values()]
                            if len(arrays) >= 2:
                                dn, edn = arrays[0], arrays[1]
                            else:
                                raise RuntimeError(
                                    "Input file missing required dn and edn "
                                    "arrays"
                                )

                    dn2d = mio.ensure_2d_dn(dn)
                    edn2d = mio.ensure_2d_dn(edn)
                    if edn2d.shape[0] != dn2d.shape[0]:
                        if edn2d.shape[0] == 1:
                            edn2d = np.repeat(edn2d, dn2d.shape[0], axis=0)
                        else:
                            raise RuntimeError(
                                "edn shape does not match dn and cannot be "
                                "broadcast"
                            )

                    print(
                        "dn2d.shape=%s, edn2d.shape=%s"
                        % (str(dn2d.shape), str(edn2d.shape))
                    )

                    n_samples = dn2d.shape[0]
                    counts = [
                        n_samples // size
                        + (1 if i < (n_samples % size) else 0)
                        for i in range(size)
                    ]
                else:
                    # non-root ranks start with placeholders; they will receive
                    # shapes/data via scatterv and counts broadcast.
                    dn2d = None
                    edn2d = None
                    n_samples = None
                    counts = None

                # Broadcast counts & scatter arrays (or local fallback)
                if comm is not None:
                    with nvtx_range("BCAST_COUNTS", color=0x1565c0):
                        counts = comm.bcast(counts, root=0)
                    with nvtx_range("BCAST_DTYPES", color=0x0d47a1):
                        if rank == 0:
                            dn_dtype_name = str(dn2d.dtype)
                            edn_dtype_name = str(edn2d.dtype)
                        else:
                            dn_dtype_name = None
                            edn_dtype_name = None
                        dn_dtype_name = comm.bcast(dn_dtype_name, root=0)
                        edn_dtype_name = comm.bcast(edn_dtype_name, root=0)
                        dn_dtype = np.dtype(dn_dtype_name)
                        edn_dtype = np.dtype(edn_dtype_name)
                    with nvtx_range("SCATTER_DN", color=0x43a047):
                        local_dn = mmpi.scatterv_array(
                            comm,
                            dn2d if rank == 0 else None,
                            counts,
                            dtype=dn_dtype,
                        )
                    with nvtx_range("SCATTER_EDN", color=0x2e7d32):
                        local_edn = mmpi.scatterv_array(
                            comm,
                            edn2d if rank == 0 else None,
                            counts,
                            dtype=edn_dtype,
                        )
                else:
                    local_dn = dn2d
                    local_edn = edn2d
                    counts = [n_samples]

                # build rmatrix like dn2dem_pos does (simple path)
                # for demonstration use a small synthetic rmatrix based on
                # filters
                nf = local_dn.shape[1]
                nt = 10
                # create log-temperature centers and widths
                logt = np.linspace(5.0, 7.0, nt)
                dlogt = np.full(nt, logt[1] - logt[0])
                tresp = np.ones((nt, nf))

                # Enforce CuPy presence and GPU assignment; fail loudly if
                # missing
                try:
                    # ensure an informative ImportError is raised via helper
                    mmpi._require_cupy()
                except Exception as e:
                    log.exception(
                        "Rank %d: CuPy requirement check failed: %s",
                        rank,
                        e,
                    )
                    raise
                if gpu_assigned is None or gpu_assigned < 0:
                    raise RuntimeError(
                        (
                            "No GPU assigned for multiGPU execution; ensure "
                            "CUDA_VISIBLE_DEVICES or N_GPUS is set and GPUs "
                            "are available."
                        )
                    )

                # Aggressive block sizing for large datasets to optimize
                # GPU utilization. Use adaptive sizing to minimize batches.
                env_block = int(os.environ.get("MULTIGPU_BATCH_SIZE", "0"))

                # Calculate optimal block size based on available GPU
                # memory and data size
                if local_dn.shape[0] <= 1024:
                    block = local_dn.shape[0]  # Single batch for small data
                elif local_dn.shape[0] <= 16384:  # 16K pixels
                    block = max(2048, local_dn.shape[0] // 4)  # ~4 batches max
                elif local_dn.shape[0] <= 1048576:  # 1M pixels
                    block = max(8192, local_dn.shape[0] // 8)  # ~8 batches max
                else:  # Large datasets (>1M pixels)
                    # ~16 batches max
                    block = max(32768, local_dn.shape[0] // 16)

                # Apply environment override only if it makes sense
                if env_block > 0:
                    # Only use env override if it's larger than the
                    # minimum efficient size
                    min_efficient = max(2048, local_dn.shape[0] // 32)
                    if env_block >= min_efficient:
                        block = min(env_block, local_dn.shape[0])
                    else:
                        log.warning(
                            "Rank %d: MULTIGPU_BATCH_SIZE=%d too small for %d "
                            "pixels, using %d",
                            rank,
                            env_block,
                            local_dn.shape[0],
                            block,
                        )
                
                # Ensure we don't exceed data size
                block = min(block, local_dn.shape[0])
                
                estimated_batches = (local_dn.shape[0] + block - 1) // block
                log.info(
                    "Rank %d: Processing %s pixels in %d batches of size %s",
                    rank,
                    f"{local_dn.shape[0]:,}",
                    estimated_batches,
                    f"{block:,}",
                )
                
                # Only log memory info for the first rank to avoid spam
                if rank == 0:
                    try:
                        import cupy as cp
                        free_mem, total_mem = cp.cuda.runtime.memGetInfo()
                        used_gb = (total_mem - free_mem) / 1024 ** 3
                        total_gb = total_mem / 1024 ** 3
                        used_pct = 100.0 * (total_mem - free_mem) / total_mem
                        log.info(
                            (
                                "GPU memory before processing: %.1f/%.1fGB "
                                "(%.1f%% used)"
                            ),
                            used_gb,
                            total_gb,
                            used_pct,
                        )
                    except Exception:
                        pass
                
                dem_local = np.zeros((local_dn.shape[0], nt))
                edem_local = np.zeros_like(dem_local)
                elogt_local = np.zeros_like(dem_local)
                chisq_local = np.zeros((local_dn.shape[0],))
                dn_reg_local = np.zeros((local_dn.shape[0], nf))

                with nvtx_range("GPU_COMPUTE", color=0xe65100):
                    for i in range(0, local_dn.shape[0], block):
                        i2 = min(local_dn.shape[0], i + block)
                        sub_dn = local_dn[i:i2]
                        sub_edn = local_edn[i:i2]
                        dem_b, edem_b, elogt_b, chisq_b, dnreg_b = (
                            gpu_kernels.demmap_pos(
                                sub_dn,
                                sub_edn,
                                tresp,
                                logt,
                                dlogt,
                                np.ones(nf),
                            )
                        )
                        dem_local[i:i2] = dem_b
                        edem_local[i:i2] = edem_b
                        elogt_local[i:i2] = elogt_b
                        chisq_local[i:i2] = chisq_b
                        dn_reg_local[i:i2] = dnreg_b

                    # End-of-file message and timing (rank 0 only to avoid
                    # spam)
                    if rank == 0:
                        log.info(
                            "Finished input %s (GPU)",
                            input_path,
                            extra={"general": True},
                        )

                # Gather results to root for final aggregation
                if comm is not None:
                    with nvtx_range("GATHER_DEM", color=0x6d4c41):
                        dem_all = mmpi.gatherv_array(
                            comm, dem_local, counts, root=0
                        )
                    with nvtx_range("POST_GATHER_BARRIER", color=0x5d4037):
                        try:
                            mmpi.barrier(comm)
                        except Exception as e:
                            log.exception(
                                "Rank %d: MPI barrier failed after gather: %s",
                                rank,
                                e,
                            )

                    if rank == 0:
                        # Basic sanity check: gathered rows must equal
                        # sum(counts)
                        expected = int(sum(counts))
                        # If gatherv returned None or a mismatched shape,
                        # attempt fallback reading per-rank local files.
                        if dem_all is None or dem_all.shape[0] != expected:
                            log.warning(
                                (
                                    "Gathered DEMs missing or size-"  # noqa: E501
                                    "mismatched; attempting fallback"
                                )
                            )
                            try:
                                parts = []
                                agg_dir = os.path.join(
                                    results_root, "aggregate"
                                )
                                basename = os.path.basename(input_path)
                                inbase = os.path.splitext(basename)[0]
                                for r in range(size):
                                    pfile = os.path.join(
                                        agg_dir,
                                        f"dem_local_r{r:03d}_{inbase}.npz",
                                    )
                                    if not os.path.exists(pfile):
                                        raise FileNotFoundError(
                                            f"Missing per-rank file: {pfile}"
                                        )
                                    with np.load(pfile) as d:
                                        parts.append(d["dem_local"])
                                dem_all = np.vstack(parts)
                                if dem_all.shape[0] != expected:
                                    raise RuntimeError(
                                        "Fallback aggregation produced %d "
                                        "rows; expected %d"
                                        % (dem_all.shape[0], expected)
                                    )
                                log.info(
                                    (
                                        "Fallback aggregation from per-rank "
                                        "files succeeded"
                                    )
                                )
                            except Exception:
                                log.exception(
                                    "Fallback aggregation failed; aborting "
                                    "save"
                                )
                                raise

                    if rank == 0 and dem_all is not None:
                        print(f"Computed total DEMs: {dem_all.shape[0]}")
                        # Save aggregated results to root results dir
                        # (per-input)
                        os.makedirs(
                            f"{results_root}/aggregate", exist_ok=True
                        )
                        inbase = os.path.splitext(
                            os.path.basename(input_path)
                        )[0]
                        final_path = os.path.join(
                            f"{results_root}/aggregate",
                            f"dem_all_{inbase}.npz",
                        )
                        # Allow disabling compression for faster shutdown
                        compress = os.environ.get(
                            "MULTIGPU_SAVE_COMPRESSED", "0"
                        ) == "1"
                        t0 = time.perf_counter()
                        with nvtx_range("SAVE_RESULTS", color=0x795548):
                            if compress:
                                np.savez_compressed(
                                    final_path, dem_all=dem_all
                                )
                            else:
                                np.savez(final_path, dem_all=dem_all)
                        dt = time.perf_counter() - t0
                        log.info(
                            (
                                "Saved aggregated DEMs to %s (shape=%s, "
                                "compressed=%s) in %.2fs"
                            ),
                            final_path,
                            tuple(dem_all.shape),
                            str(compress),
                            dt,
                        )

                else:
                    # Non-MPI fallback: we only have local results
                    print(f"Computed total DEMs: {dem_local.shape[0]}")
                    os.makedirs(f"{results_root}/aggregate", exist_ok=True)
                    inbase = os.path.splitext(
                        os.path.basename(input_path)
                    )[0]
                    final_path = os.path.join(
                        f"{results_root}/aggregate",
                        f"dem_all_{inbase}.npz",
                    )
                    # Save local array under same key name for consistency
                    compress = os.environ.get(
                        "MULTIGPU_SAVE_COMPRESSED", "0"
                    ) == "1"
                    t0 = time.perf_counter()
                    with nvtx_range("SAVE_RESULTS", color=0x795548):
                        if compress:
                            np.savez_compressed(final_path, dem_all=dem_local)
                        else:
                            np.savez(final_path, dem_all=dem_local)
                    dt = time.perf_counter() - t0
                    log.info(
                        (
                            "Saved local DEMs (serial) to %s (shape=%s, "
                            "compressed=%s) in %.2fs"
                        ),
                        final_path,
                        tuple(dem_local.shape),
                        str(compress),
                        dt,
                    )
                    log.info(
                        f"Saved local DEMs (serial mode) to {final_path}"
                    )

            except Exception:
                log.exception(
                    "Rank %d: exception while processing %s",
                    rank,
                    input_path,
                )
                raise

    # Final barrier to keep ranks in lock-step before shutdown
    if comm is not None:
        with nvtx_range("FINAL_BARRIER", color=0x9e9e9e):
            try:
                mmpi.barrier(comm)
            except Exception as e:
                logging.getLogger(__name__).warning(
                    "Final MPI barrier failed: %s", e
                )

    # Clean shutdown of logging to ensure all records are flushed
    with nvtx_range("SHUTDOWN", color=0x9e9e9e):
        mlog.shutdown_logging()


if __name__ == "__main__":
    main()
