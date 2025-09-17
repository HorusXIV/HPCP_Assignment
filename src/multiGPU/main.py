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
    p.add_argument("--input", default="data/np32/20170906_12_00_12.npz")
    p.add_argument("--max-samples", type=int, default=1000,
                   help="maximum number of spatial samples (pixels) to process for smoke tests")
    p.add_argument("--block-size", type=int, default=128,
                   help="number of pixels to process per GPU batch")
    return p.parse_args()


def main():
    args = parse_args()

    comm, rank, size = mmpi.init_mpi()
    # Setup results directory structure
    results_root = "multiGPU/results"
    # Each rank will have a per-rank subdir; root rank will host aggregates
    per_rank_dir = f"{results_root}/rank{rank:03d}"
    if rank == 0:
        # ensure parent exists
        import os

        os.makedirs(results_root, exist_ok=True)

    # Initialize logging (creates logs/ under results_root)
    logger = mlog.setup_logging(results_root, rank=rank, size=size, console=(rank == 0))
    log = logging.getLogger(__name__)
    log.info(f"Starting rank {rank}/{size}; results dir: {results_root}")

    # Create a checkpoint manager for this rank; rank 0 can also aggregate
    ck = CheckpointManager(outdir=per_rank_dir, keep=5, comm=comm, rank=rank)
    # Map rank to GPU and bind environment (prefer per-node local-rank mapping)
    local_rank, local_size, node = mmpi.get_local_rank_info(comm)
    gpu_assigned = mmpi.set_device_for_local_rank(comm)
    if rank == 0:
        print(f"MPI size={size}, launching job with input={args.input}, node={node}, local_size={local_size}, gpu_assigned={gpu_assigned}")

    if rank == 0:
        print(f"MPI size={size}, launching job with input={args.input}")

    # Simple serial-friendly loader
    data = mio.load_npz(args.input)

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
        log.info(
            f"Rank {rank}: processed {local_dn.shape[0]} pixels on GPU {gpu_assigned}"
        )
    else:
        # CPU fallback
        dem_norm0 = np.ones((local_dn.shape[0], nt))
        dem_local, edem_local, elogt_local, chisq_local, dn_reg_local = demmap_pos_cpu(
            local_dn, local_edn, tresp, logt, dlogt, np.ones(nf), dem_norm0=dem_norm0
        )
        log.info(f"Rank {rank}: processed {local_dn.shape[0]} pixels on CPU fallback")

    # Gather results to root for final aggregation
    if comm is not None:
        dem_all = mmpi.gatherv_array(comm, dem_local, counts, root=0)
        if rank == 0:
            print(f"Computed total DEMs: {dem_all.shape[0]}")
            # Save aggregated results to root results dir
            import os

            os.makedirs(f"{results_root}/aggregate", exist_ok=True)
            outpath = os.path.join(f"{results_root}/aggregate", "dem_all.npz")
            np.savez_compressed(outpath, dem_all=dem_all)
            log.info(f"Saved aggregated DEMs to {outpath}")
            # Optionally create a checkpoint manifest
            ck_root = CheckpointManager(outdir=f"{results_root}/checkpoints", keep=10, comm=comm, rank=0)
            ck_root.save({"dem_all_path": outpath, "counts": counts}, step=0, async_write=False)
    else:
        print(f"Computed total DEMs: {dem_local.shape[0]}")
        # Save local results for serial mode
        import os

        os.makedirs(per_rank_dir, exist_ok=True)
        outpath = os.path.join(per_rank_dir, "dem_local.npz")
        np.savez_compressed(outpath, dem_local=dem_local)
        log.info(f"Saved local DEMs to {outpath}")

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
