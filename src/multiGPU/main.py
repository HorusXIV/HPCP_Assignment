"""Entry point for launching multi-GPU DEM computations with explicit MPI.

This script demonstrates how to initialize MPI, bind GPUs to ranks and run a
GPU-accelerated kernel over input data. It intentionally keeps dependencies
light so it can be executed on development machines as a serial script.
"""
from __future__ import annotations

import argparse
import logging
import numpy as np
from . import io as mio
from . import mpi_manager as mmpi
from . import gpu_kernels
from . import logging as mlog
from .checkpoint import CheckpointManager


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=None,
                   help="single input file to process (mutually exclusive with --input-dir)")
    p.add_argument("--input-dir", default=None,
                   help="directory to scan for .npz input files; rank 0 will enumerate and distribute them")
    p.add_argument("--max-samples", type=int, default=1000,
                   help="maximum number of spatial samples (pixels) to process for smoke tests")
    p.add_argument("--block-size", type=int, default=128,
                   help="number of pixels to process per GPU batch")
    return p.parse_args()


def main():
    args = parse_args()

    comm, rank, size = mmpi.init_mpi()
    # Setup results directory structure
    results_root = "src/multiGPU/results"
    # Each rank will have a per-rank subdir; root rank will host aggregates
    per_rank_dir = f"{results_root}/rank{rank:03d}"
    if rank == 0:
        # ensure parent exists
        import os

        os.makedirs(results_root, exist_ok=True)

    # Initialize logging (creates logs/ under results_root)
    _ = mlog.setup_logging(results_root, rank=rank, size=size, console=(rank == 0))
    log = logging.getLogger(__name__)
    log.info(f"Starting rank {rank}/{size}; results dir: {results_root}")

    # Create a checkpoint manager for this rank; rank 0 can also aggregate
    ck = CheckpointManager(outdir=per_rank_dir, keep=5, comm=comm, rank=rank)
    # Map rank to GPU and bind environment (prefer per-node local-rank mapping)
    local_rank, local_size, node = mmpi.get_local_rank_info(comm)
    gpu_assigned = mmpi.set_device_for_local_rank(comm)
    # Startup banner
    if rank == 0:
        print(f"MPI size={size}, node={node}, local_size={local_size}, gpu_assigned={gpu_assigned}")

    # Decide input list: either single file or enumerate a directory
    import os
    import glob

    if args.input_dir and args.input:
        if rank == 0:
            raise RuntimeError("Specify only one of --input or --input-dir")

    if args.input_dir:
        # Rank 0 enumerates files and distributes the list to all ranks
        if rank == 0:
            pattern = os.path.join(args.input_dir, "*.npz")
            all_inputs = sorted(glob.glob(pattern))
            if len(all_inputs) == 0:
                raise RuntimeError(f"No .npz files found in {args.input_dir}")
        else:
            all_inputs = None

        # Broadcast the list of inputs to all ranks (serial stub will just get None->handled)
        if comm is not None:
            all_inputs = comm.bcast(all_inputs, root=0)
        else:
            # non-MPI fallback: enumerate locally
            all_inputs = sorted(glob.glob(os.path.join(args.input_dir, "*.npz")))

        # Assign contiguous ranges of files to each rank for balanced I/O
        n_files = len(all_inputs)
        counts = [n_files // size + (1 if i < (n_files % size) else 0) for i in range(size)]
        displs = [sum(counts[:i]) for i in range(len(counts))]
        my_start = displs[rank]
        my_end = my_start + counts[rank]
        my_inputs = all_inputs[my_start:my_end]

        if rank == 0:
            print(f"Distributing {n_files} input files across {size} ranks: counts={counts}")

    else:
        # Single-input mode (existing behavior)
        if args.input is None:
            # default to the old hard-coded example if nothing provided
            args.input = "data/np32/20170906_12_00_12.npz"
        my_inputs = [args.input]

    # Announce assigned inputs on every rank (helps debug distribution)
    if rank == 0:
        print(f"Rank {rank} will process {len(my_inputs)} input(s): {my_inputs}")
    else:
        print(f"Rank {rank} will process {len(my_inputs)} input(s): {my_inputs}")

    # Process each assigned input file independently
    for input_path in my_inputs:
        # Announce input start and measure elapsed time
        log.info(f"Rank {rank}: starting processing input {input_path}")
        import time
        t0 = time.time()
        try:
            # Simple serial-friendly loader (per-file)
            data = mio.load_npz(input_path)

            # Heuristics for common file layouts: prefer `dn`/`edn`, otherwise `bands`
            dn = data.get("dn", None)
            edn = data.get("edn", None)
            if dn is None:
                # AIA-style archive: `bands` shaped (nf, ny, nx)
                if "bands" in data:
                    bands = data["bands"]
                    if bands.ndim != 3:
                        raise RuntimeError("Unexpected `bands` shape, expected (nf, ny, nx)")
                    nf, ny, nx = bands.shape
                    n_pixels = ny * nx
                    max_samples = args.max_samples
                    # reshape to (nf, n_pixels) and sample columns if necessary
                    bands_flat = bands.reshape(nf, n_pixels)
                    if max_samples is not None and n_pixels > max_samples:
                        idx = np.linspace(0, n_pixels - 1, num=max_samples, dtype=int)
                        dn2d = bands_flat[:, idx].T
                    else:
                        dn2d = bands_flat.T
                    # default edn: ones per filter, broadcast to pixels
                    edn2d = np.ones((dn2d.shape[0], nf), dtype=float)
                    dn = dn2d
                    edn = edn2d
                else:
                    # try generic first array(s)
                    arrays = [v for v in data.values()]
                    if len(arrays) >= 2:
                        dn, edn = arrays[0], arrays[1]
                    else:
                        if rank == 0:
                            raise RuntimeError("Input file does not contain dn and edn arrays")

            # Distribute dn rows across ranks using MPI scatter/gather semantics.
            # We'll fall back to using the baseline CPU demmap_pos on each rank for
            # correctness and incrementally port inner loops to GPU kernels.
            try:
                from mpi4py import MPI
            except Exception:
                MPI = None

            # Prepare data for scatter: flatten spatial dims to samples x filters
            dn2d = mio.ensure_2d_dn(dn)
            edn2d = mio.ensure_2d_dn(edn)

            # Ensure edn2d has same number of rows; if single row provided, broadcast
            if edn2d.shape[0] != dn2d.shape[0]:
                if edn2d.shape[0] == 1:
                    edn2d = np.repeat(edn2d, dn2d.shape[0], axis=0)
                else:
                    raise RuntimeError("edn array shape does not match dn and cannot be broadcast")

            if rank == 0:
                print(f"dn2d.shape={dn2d.shape}, edn2d.shape={edn2d.shape}")

            n_samples = dn2d.shape[0]
            # compute per-rank counts in a deterministic way
            if comm is not None:
                counts = [n_samples // size + (1 if i < (n_samples % size) else 0) for i in range(size)]
                local_dn = mmpi.scatterv_array(comm, dn2d if rank == 0 else None, counts, dtype=dn2d.dtype)
                local_edn = mmpi.scatterv_array(comm, edn2d if rank == 0 else None, counts, dtype=edn2d.dtype)
            else:
                counts = [n_samples]
                local_dn = dn2d
                local_edn = edn2d

            # Import CPU dem implementation (baseline) and run per-rank
            from ..baseline.vendor.demmap_pos import demmap_pos as demmap_pos_cpu

            # build rmatrix like dn2dem_pos does (simple path)
            # for demonstration use a small synthetic rmatrix based on filters
            nf = local_dn.shape[1]
            nt = 10
            # create log-temperature centers and widths
            logt = np.linspace(5.0, 7.0, nt)
            dlogt = np.full(nt, logt[1] - logt[0])
            tresp = np.ones((nt, nf))

            # Choose GPU-resident kernels when available and a device was assigned
            try:
                import cupy as cp
                gpu_ok = True
            except Exception:
                cp = None
                gpu_ok = False

            if gpu_ok and gpu_assigned is not None and gpu_assigned >= 0:
                # process local pixels in blocks to amortize GPU setup costs
                block = args.block_size
                dem_local = np.zeros((local_dn.shape[0], nt))
                edem_local = np.zeros_like(dem_local)
                elogt_local = np.zeros_like(dem_local)
                chisq_local = np.zeros((local_dn.shape[0],))
                dn_reg_local = np.zeros((local_dn.shape[0], nf))

                for i in range(0, local_dn.shape[0], block):
                    i2 = min(local_dn.shape[0], i + block)
                    sub_dn = local_dn[i:i2]
                    sub_edn = local_edn[i:i2]
                    dem_b, edem_b, elogt_b, chisq_b, dnreg_b = gpu_kernels.demmap_pos(
                        sub_dn, sub_edn, tresp, logt, dlogt, np.ones(nf)
                    )
                    dem_local[i:i2] = dem_b
                    edem_local[i:i2] = edem_b
                    elogt_local[i:i2] = elogt_b
                    chisq_local[i:i2] = chisq_b
                    dn_reg_local[i:i2] = dnreg_b
                log.info(f"Rank {rank}: processed {local_dn.shape[0]} pixels on GPU {gpu_assigned}")
                elapsed = time.time() - t0
                log.info(f"Rank {rank}: finished input {input_path} in {elapsed:.2f}s (GPU)")
            else:
                # CPU fallback
                dem_norm0 = np.ones((local_dn.shape[0], nt))
                dem_local, edem_local, elogt_local, chisq_local, dn_reg_local = demmap_pos_cpu(
                    local_dn, local_edn, tresp, logt, dlogt, np.ones(nf), dem_norm0=dem_norm0
                )
                log.info(f"Rank {rank}: processed {local_dn.shape[0]} pixels on CPU fallback")
                elapsed = time.time() - t0
                log.info(f"Rank {rank}: finished input {input_path} in {elapsed:.2f}s (CPU)")

            # Persist local per-rank results (atomic write) so we can recover
            # or post-process if gather fails. Each rank writes to its own dir.
            try:
                import os
                import tempfile

                os.makedirs(per_rank_dir, exist_ok=True)
                local_inbase = os.path.splitext(os.path.basename(input_path))[0]
                local_final = os.path.join(per_rank_dir, f"dem_local_{local_inbase}.npz")
                tmp_fd, tmp_local = tempfile.mkstemp(prefix=f"dem_local_{local_inbase}.", dir=per_rank_dir)
                try:
                    os.close(tmp_fd)
                    np.savez_compressed(tmp_local, dem_local=dem_local)
                    os.replace(tmp_local, local_final)
                    log.info(f"Rank {rank}: saved local results to {local_final}")
                except Exception:
                    log.exception(f"Rank {rank}: failed to save local results to {local_final}")
                    try:
                        if os.path.exists(tmp_local):
                            os.remove(tmp_local)
                    except Exception:
                        pass
                    raise
            except Exception:
                # best-effort: do not block on local save failures
                log.exception("Failed to persist per-rank local results")

            # Gather results to root for final aggregation
            if comm is not None:
                dem_all = mmpi.gatherv_array(comm, dem_local, counts, root=0)
                # Ensure all ranks reach the same point before root writes file
                try:
                    mmpi.barrier(comm)
                except Exception:
                    # barrier failure shouldn't prevent us from attempting to save,
                    # but log it for diagnostics
                    log.exception("MPI barrier failed after gather")

                if rank == 0:
                    # Basic sanity check: gathered rows must equal sum(counts)
                    expected = int(sum(counts))
                    # If gatherv returned None or a mismatched shape, attempt
                    # to recover by reading per-rank local files (shared FS)
                    if dem_all is None or dem_all.shape[0] != expected:
                        log.warning(
                            "Gathered DEMs missing or size-mismatched; attempting fallback"
                        )
                        # Try reading per-rank local outputs and concatenating
                        try:
                            parts = []
                            for r in range(size):
                                pdir = f"{results_root}/rank{r:03d}"
                                pfile = os.path.join(pdir, f"dem_local_{local_inbase}.npz")
                                if not os.path.exists(pfile):
                                    raise FileNotFoundError(f"Missing per-rank file: {pfile}")
                                with np.load(pfile) as d:
                                    parts.append(d["dem_local"])
                            dem_all = np.vstack(parts)
                            if dem_all.shape[0] != expected:
                                raise RuntimeError(
                                    f"Fallback aggregation produced {dem_all.shape[0]} rows, expected {expected}"
                                )
                            log.info("Fallback aggregation from per-rank files succeeded")
                        except Exception:
                            log.exception("Fallback aggregation failed; aborting save")
                            raise

                    print(f"Computed total DEMs: {dem_all.shape[0]}")
                    # Save aggregated results to root results dir (per-input)
                    import os
                    import tempfile

                    os.makedirs(f"{results_root}/aggregate", exist_ok=True)
                    inbase = os.path.splitext(os.path.basename(input_path))[0]
                    final_path = os.path.join(f"{results_root}/aggregate", f"dem_all_{inbase}.npz")

                    # Write atomically: write to a temp file in same dir then rename
                    tmp_fd, tmp_path = tempfile.mkstemp(prefix=f"dem_all_{inbase}.", dir=f"{results_root}/aggregate")
                    try:
                        os.close(tmp_fd)
                        np.savez_compressed(tmp_path, dem_all=dem_all)
                        # Atomic replace
                        os.replace(tmp_path, final_path)
                        log.info(f"Saved aggregated DEMs to {final_path}")
                    except Exception:
                        log.exception(f"Failed to save aggregated DEMs to {final_path}")
                        # Clean up temp file if present
                        try:
                            if os.path.exists(tmp_path):
                                os.remove(tmp_path)
                        except Exception:
                            pass
                        raise

                    # Optionally create a checkpoint manifest
                    ck_root = CheckpointManager(outdir=f"{results_root}/checkpoints", keep=10, comm=comm, rank=0)
                    ck_root.save({"dem_all_path": final_path, "counts": counts}, step=0, async_write=False)
            else:
                print(f"Computed total DEMs: {dem_local.shape[0]}")
                # Save local results for serial mode (per-input)
                import os

                os.makedirs(per_rank_dir, exist_ok=True)
                inbase = os.path.splitext(os.path.basename(input_path))[0]
                outpath = os.path.join(per_rank_dir, f"dem_local_{inbase}.npz")
                np.savez_compressed(outpath, dem_local=dem_local)
                log.info(f"Saved local DEMs to {outpath}")

        except Exception:
            log.exception(f"Rank {rank}: exception while processing {input_path}")
            raise

    # Save per-rank checkpoint of metadata (non-blocking)
    meta = {"rank": rank, "n_local": int(local_dn.shape[0]), "gpu_assigned": int(gpu_assigned or -1)}
    try:
        ck.save(meta, step=0, async_write=True)
    except Exception:
        log.exception("Failed to save per-rank checkpoint")

    # Clean shutdown of logging to ensure all records are flushed
    mlog.shutdown_logging()


if __name__ == "__main__":
    main()
