"""Launch multi-GPU DEM computations with explicit MPI orchestration.

Flow:
    1. Initialize/broadcast MPI state and map ranks to GPUs.
    2. Rank 0 enumerates inputs and broadcasts the worklist.
    3. For each input, rank 0 loads and scatters rows to all ranks.
    4. All ranks compute once (internal batching), then gather DEMs.
    5. Rank 0 saves a single aggregated output per input.
"""

from __future__ import annotations

# pylint: disable=line-too-long

import argparse
import logging
import numpy as np
import os
import glob
import time
from src.common.nvtx import nvtx_range
from . import io as mio
from . import mpi_manager as mmpi
from . import kernels
from . import logging as mlog


def parse_args():
    """Parse CLI arguments for the multi-GPU entry point.

    Returns:
        argparse.Namespace: Parsed CLI options.
    """
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
        help=("maximum number of spatial samples (pixels) to process"),
    )
    return p.parse_args()


def main():
    """Program entrypoint when executed as ``python -m src.multiGPU.main``.

    Initializes MPI, logging, rank-to-GPU binding, handles I/O, and invokes
    the GPU kernels. Errors are allowed to propagate to ensure proper Slurm
    failure signaling.
    """
    args = parse_args()

    with nvtx_range("INIT_MPI", color=0x4CAF50):
        comm, rank, size = mmpi.init_mpi()

    results_root = "data/results_multiGPU"
    log_root = "src/multiGPU/logs"
    if rank == 0:
        os.makedirs(results_root, exist_ok=True)
    _ = mlog.setup_logging(log_root, rank=rank, size=size)
    log = logging.getLogger(__name__)
    log.info("Starting rank %d/%d; results dir: %s", rank, size - 1, results_root)
    try:
        from src.common.nvtx import nvtx_available as _nvtx_avail

        _nvtx_env = os.environ.get("MULTIGPU_NVTX", "0")
        _nvtx_on = _nvtx_avail()
        log.info("NVTX enabled=%s (env MULTIGPU_NVTX=%s)", str(_nvtx_on), _nvtx_env)
        if _nvtx_env == "1" and not _nvtx_on:
            log.warning(
                "NVTX requested but unavailable: install profiling extras (poetry install --with profiling) so the 'nvtx' package is present."
            )
    except Exception:
        pass

    # Map ranks to GPUs and validate the runtime environment
    with nvtx_range("RANK_GPU_BIND", color=0x009688):
        local_rank, local_size, node = mmpi.get_local_rank_info(comm)
        gpu_assigned = mmpi.set_device_for_local_rank(comm)
    log.info(
        "MPI size=%d, node=%s, local_size=%d, gpu_assigned=%s",
        size,
        node,
        local_size,
        str(gpu_assigned),
    )

    if os.environ.get("MULTIGPU_PREEMPT", "0") == "1":
        try:
            from . import preempt as _preempt

            def _on_preempt_save():
                mark_dir = os.path.join(log_root, "logs")
                os.makedirs(mark_dir, exist_ok=True)
                mark = os.path.join(mark_dir, f"preempt_rank{rank:03d}.txt")
                with open(mark, "a", encoding="utf-8") as f:
                    f.write("preempt received\n")

            _preempt.register_preempt_handlers(_on_preempt_save, comm=comm)
            log.info("Preemption handlers registered (rank %d)", rank)
        except Exception:
            log.warning("Failed to register preemption handlers", exc_info=True)

    if not args.input_dir:
        if rank == 0:
            raise RuntimeError("Must specify --input-dir to process input files")
        return

    with nvtx_range("ENUM_INPUTS", color=0x2196F3):
        if rank == 0:
            pattern = os.path.join(args.input_dir, "*.npz")
            all_inputs = sorted(glob.glob(pattern))
            if len(all_inputs) == 0:
                raise RuntimeError(f"No .npz files found in {args.input_dir}")
        else:
            all_inputs = None
    with nvtx_range("BCAST_INPUTS", color=0x1976D2):
        if comm is not None:
            all_inputs = comm.bcast(all_inputs, root=0)
        else:
            pattern = os.path.join(args.input_dir, "*.npz")
            all_inputs = sorted(glob.glob(pattern))

    if rank == 0:
        log.info(
            "Processing %d input files across %d ranks",
            len(all_inputs),
            size,
            extra={"general": True},
        )

    for input_path in all_inputs:
        file_label = f"PROCESS_FILE:{os.path.basename(input_path)}"
        with nvtx_range(file_label, color=0xFF9800):
            t_img_start = time.perf_counter()
            if rank == 0:
                log.info(
                    "Starting processing input %s",
                    os.path.basename(input_path),
                    extra={"general": True},
                )
            try:
                if rank == 0:
                    with nvtx_range("LOAD_FILE", color=0x8E24AA):
                        data = mio.load_npz(input_path)
                    # Prefer dn/edn, fall back to bands
                    dn = data.get("dn", None)
                    edn = data.get("edn", None)
                    if dn is None:
                        if "bands" in data:
                            bands = data["bands"]
                            if bands.ndim != 3:
                                raise RuntimeError(
                                    ("Unexpected `bands` shape; expected (nf, ny, nx)")
                                )
                            nf, ny, nx = bands.shape
                            n_pixels = ny * nx
                            bands_flat = bands.reshape(nf, n_pixels)
                            if (
                                args.max_samples is not None
                                and n_pixels > args.max_samples
                            ):
                                idx = np.linspace(
                                    0, n_pixels - 1, num=args.max_samples, dtype=int
                                )
                                dn2d = bands_flat[:, idx].T
                            else:
                                dn2d = bands_flat.T
                            edn2d = np.ones((dn2d.shape[0], nf), dtype=float)
                        else:
                            arrays = [v for v in data.values()]
                            if len(arrays) >= 2:
                                dn2d = mio.ensure_2d_dn(arrays[0])
                                edn2d = mio.ensure_2d_dn(arrays[1])
                            else:
                                raise RuntimeError(
                                    ("Input file missing required dn and edn arrays")
                                )
                    else:
                        dn2d = mio.ensure_2d_dn(dn)
                        edn2d = mio.ensure_2d_dn(edn)

                    if edn2d.shape[0] != dn2d.shape[0]:
                        if edn2d.shape[0] == 1:
                            edn2d = np.repeat(edn2d, dn2d.shape[0], axis=0)
                        else:
                            raise RuntimeError(
                                ("edn shape does not match dn and cannot be broadcast")
                            )

                    n_samples = int(dn2d.shape[0])
                    counts = [
                        n_samples // size + (1 if i < (n_samples % size) else 0)
                        for i in range(size)
                    ]
                    dn_dtype_name = str(dn2d.dtype)
                    edn_dtype_name = str(edn2d.dtype)
                else:
                    dn2d = None
                    edn2d = None
                    counts = None
                    dn_dtype_name = None
                    edn_dtype_name = None

                if comm is not None:
                    with nvtx_range("BCAST_COUNTS", color=0x1565C0):
                        counts = comm.bcast(counts, root=0)
                    with nvtx_range("BCAST_DTYPES", color=0x0D47A1):
                        dn_dtype_name = comm.bcast(dn_dtype_name, root=0)
                        edn_dtype_name = comm.bcast(edn_dtype_name, root=0)
                        dn_dtype = np.dtype(dn_dtype_name)
                        edn_dtype = np.dtype(edn_dtype_name)
                    with nvtx_range("SCATTER_DN", color=0x43A047):
                        local_dn = mmpi.scatterv_array(
                            comm, dn2d if rank == 0 else None, counts, dtype=dn_dtype
                        )
                    with nvtx_range("SCATTER_EDN", color=0x2E7D32):
                        local_edn = mmpi.scatterv_array(
                            comm, edn2d if rank == 0 else None, counts, dtype=edn_dtype
                        )
                else:
                    # Serial path
                    local_dn = dn2d
                    local_edn = edn2d
                    counts = [local_dn.shape[0]]

                if rank == 0:
                    try:
                        total_px = int(np.sum(counts)) if counts is not None else 0
                        log.info(
                            "Pixels per rank (total=%d): %s",
                            total_px,
                            counts,
                            extra={"general": True},
                        )
                    except Exception:
                        pass

                # Route computation through the multiGPU dn2dem_pos wrapper.
                # The wrapper will construct responses once internally for consistency.
                nf = int(local_dn.shape[1])
                nt = 10  # kept for logging/metrics parity only
                try:
                    verbose = mlog.verbose_enabled()
                except Exception:
                    log.error("Failed to query verbose logging state")
                    verbose = False
                if verbose:
                    na_rank = int(local_dn.shape[0])
                    nf_rank = int(nf)
                    nt_rank = int(nt)
                    nmu_rank = 42
                    plan = kernels.estimate_batch_plan(
                        na_rank, nf_rank, nt_rank, nmu_rank
                    )
                    bytes_per_img = int(plan.get("bytes_per_sample", 0) * nt_rank)
                    log.info(
                        "[metrics] rank=%d pixels=%d nf=%d nt=%d batch_size=%d num_batches=%d bytes_per_sample=%d est_batch_bytes=%d per_image_bytes~=%d free_bytes=%s",
                        rank,
                        na_rank,
                        nf_rank,
                        nt_rank,
                        int(plan.get("batch_size", 0)),
                        int(plan.get("num_batches", 0)),
                        int(plan.get("bytes_per_sample", 0)),
                        int(plan.get("est_batch_bytes", 0)),
                        int(bytes_per_img),
                        str(plan.get("free_bytes")),
                    )

                mmpi._require_cupy()
                if gpu_assigned is None or gpu_assigned < 0:
                    raise RuntimeError("No GPU assigned for multiGPU execution")

                from .dn2dem_pos import dn2dem_pos as _mg_dn2dem_pos

                with nvtx_range("GPU_COMPUTE", color=0xE65100):
                    (
                        dem_local,
                        edem_local,
                        elogt_local,
                        chisq_local,
                        dn_reg_local,
                    ) = _mg_dn2dem_pos(
                        local_dn,
                        local_edn,
                        # Responses built internally for synthetic runs
                        tresp=None,
                        tresp_logt=None,
                        temps=None,
                    )

                if comm is not None:
                    with nvtx_range("GATHER_DEM", color=0x6D4C41):
                        dem_all = mmpi.gatherv_array(comm, dem_local, counts, root=0)
                    with nvtx_range("POST_GATHER_BARRIER", color=0x5D4037):
                        try:
                            mmpi.barrier(comm)
                        except Exception as e:
                            log.exception(
                                "Rank %d: MPI barrier failed after gather: %s", rank, e
                            )
                else:
                    dem_all = dem_local

                if rank == 0 and dem_all is not None:
                    try:
                        dmin = float(np.min(dem_all))
                        dmax = float(np.max(dem_all))
                        dmean = float(np.mean(dem_all))
                        log.info(
                            "DEM summary: count=%d min=%.3e max=%.3e mean=%.3e",
                            dem_all.shape[0],
                            dmin,
                            dmax,
                            dmean,
                        )
                    except Exception:
                        pass
                    log.info(
                        f"Computed total DEMs: {dem_all.shape[0]}",
                        extra={"general": True},
                    )
                    out_dir = results_root
                    os.makedirs(out_dir, exist_ok=True)
                    inbase = os.path.splitext(os.path.basename(input_path))[0]
                    final_path = os.path.join(out_dir, f"dem_all_{inbase}.npz")
                    comp_e = os.environ.get("MULTIGPU_SAVE_COMPRESSED", "0")
                    compress = comp_e == "1"
                    t0 = time.perf_counter()
                    with nvtx_range("SAVE_RESULTS", color=0x795548):
                        if compress:
                            np.savez_compressed(final_path, dem_all=dem_all)
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
                    t_img = time.perf_counter() - t_img_start
                    log.info(
                        f"Finished processing input {os.path.basename(input_path)} in {t_img:.2f}s",
                        extra={"general": True},
                    )
            except Exception as e:
                log.exception(
                    "Rank %d: exception while processing %s: %s",
                    rank,
                    input_path,
                    e,
                )
                raise

    if comm is not None:
        with nvtx_range("FINAL_BARRIER", color=0x9E9E9E):
            try:
                mmpi.barrier(comm)
            except Exception as e:
                logging.getLogger(__name__).warning("Final MPI barrier failed: %s", e)

    with nvtx_range("SHUTDOWN", color=0x9E9E9E):
        mlog.shutdown_logging()


if __name__ == "__main__":
    main()
