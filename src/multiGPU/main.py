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

    # Full inter-file pipeline: pre-load first, then overlap steps across files
    def _load_and_partition(path):
        with nvtx_range("LOAD_FILE", color=0x8E24AA):
            data = mio.load_npz(path)
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
                if args.max_samples is not None and n_pixels > args.max_samples:
                    idx = np.linspace(0, n_pixels - 1, num=args.max_samples, dtype=int)
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
        _counts = [
            n_samples // size + (1 if i < (n_samples % size) else 0)
            for i in range(size)
        ]
        return dn2d, edn2d, _counts, str(dn2d.dtype), str(edn2d.dtype)

    pipeline = os.environ.get("MULTIGPU_PIPELINE_FILES", "1") == "1"

    # Pre-stage first file
    idx_file = 0
    pending_save_threads = []
    next_dn2d = None
    next_edn2d = None
    next_counts = None
    next_dtypes = (None, None)

    if pipeline and len(all_inputs) > 0 and rank == 0:
        dn2d0, edn2d0, counts0, dn_dtype_name0, edn_dtype_name0 = _load_and_partition(
            all_inputs[0]
        )
    else:
        dn2d0 = edn2d0 = counts0 = dn_dtype_name0 = edn_dtype_name0 = None

    while idx_file < len(all_inputs):
        input_path = all_inputs[idx_file]
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
                # Load/partition current file
                if rank == 0:
                    if pipeline and dn2d0 is not None:
                        dn2d, edn2d, counts = dn2d0, edn2d0, counts0
                        dn_dtype_name, edn_dtype_name = dn_dtype_name0, edn_dtype_name0
                    else:
                        dn2d, edn2d, counts, dn_dtype_name, edn_dtype_name = (
                            _load_and_partition(input_path)
                        )
                else:
                    dn2d = edn2d = counts = dn_dtype_name = edn_dtype_name = None

                if comm is not None:
                    with nvtx_range("BCAST_COUNTS", color=0x1565C0):
                        counts = comm.bcast(counts, root=0)
                    with nvtx_range("BCAST_DTYPES", color=0x0D47A1):
                        dn_dtype_name = comm.bcast(dn_dtype_name, root=0)
                        edn_dtype_name = comm.bcast(edn_dtype_name, root=0)
                        dn_dtype = np.dtype(dn_dtype_name)
                        edn_dtype = np.dtype(edn_dtype_name)
                    # Nonblocking scatters to overlap with GPU compute
                    with nvtx_range("SCATTER_DN", color=0x43A047):
                        local_dn, req_sd = mmpi.iscatterv_array(
                            comm, dn2d if rank == 0 else None, counts, dtype=dn_dtype
                        )
                    with nvtx_range("SCATTER_EDN", color=0x2E7D32):
                        local_edn, req_se = mmpi.iscatterv_array(
                            comm, edn2d if rank == 0 else None, counts, dtype=edn_dtype
                        )
                    # Ensure local recv complete before compute
                    if req_sd is not None:
                        req_sd.Wait()
                    if req_se is not None:
                        req_se.Wait()
                else:
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
                    # Optional downcast before gather to reduce network volume
                    # Controlled via env MULTIGPU_GATHER_DTYPE in {"float64","float32"}
                    gather_dtype_env = os.environ.get(
                        "MULTIGPU_GATHER_DTYPE", "float64"
                    ).lower()
                    if gather_dtype_env not in {"float64", "float32"}:
                        gather_dtype_env = "float64"
                    if gather_dtype_env == "float32":
                        dem_send = dem_local.astype(np.float32, copy=False)
                    else:
                        dem_send = dem_local
                    # Start nonblocking gather to allow overlap with local work on rank 0
                    with nvtx_range("GATHER_DEM", color=0x6D4C41):
                        dem_all, req = mmpi.igatherv_array(
                            comm, dem_send, counts, root=0
                        )
                    # No immediate barrier: we'll wait on req only on root before saving
                else:
                    dem_all = dem_local
                    req = None

                # While current is gathering on root, pre-stage next file's load/partition on root
                next_dn2d = next_edn2d = next_counts = None
                next_dtypes = (None, None)
                if pipeline and (idx_file + 1) < len(all_inputs) and rank == 0:
                    with nvtx_range("PRELOAD_NEXT", color=0x0097A7):
                        nd, ne, ncounts, ndt, edt = _load_and_partition(
                            all_inputs[idx_file + 1]
                        )
                        next_dn2d, next_edn2d, next_counts = nd, ne, ncounts
                        next_dtypes = (ndt, edt)

                if rank == 0 and (dem_all is not None or req is not None):
                    # Ensure gather completion on root
                    if req is not None:
                        with nvtx_range("WAIT_GATHER", color=0x5D4037):
                            try:
                                req.Wait()
                            except Exception as e:
                                log.exception("Rank 0: Igatherv wait failed: %s", e)
                    # At this point dem_all is populated on root
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
                    # Saving – optionally overlap by dispatching to a background thread
                    out_dir = results_root
                    os.makedirs(out_dir, exist_ok=True)
                    inbase = os.path.splitext(os.path.basename(input_path))[0]
                    final_path = os.path.join(out_dir, f"dem_all_{inbase}.npz")
                    comp_e = os.environ.get("MULTIGPU_SAVE_COMPRESSED", "0")
                    compress = comp_e == "1"
                    # Allow saving in a specific dtype to cut I/O size
                    save_dtype_env = os.environ.get("MULTIGPU_SAVE_DTYPE", None)
                    if save_dtype_env is not None:
                        try:
                            target_dtype = np.dtype(save_dtype_env)
                        except Exception:
                            target_dtype = dem_all.dtype
                    else:
                        if (
                            "gather_dtype_env" in locals()
                            and gather_dtype_env == "float32"
                        ):
                            target_dtype = np.dtype("float32")
                        else:
                            target_dtype = dem_all.dtype

                    if dem_all.dtype != target_dtype:
                        dem_to_save = dem_all.astype(target_dtype, copy=False)
                    else:
                        dem_to_save = dem_all

                    def _save_npz(path, arr, do_compress, shape, dtype_str):
                        t0_ = time.perf_counter()
                        with nvtx_range("SAVE_RESULTS", color=0x795548):
                            if do_compress:
                                np.savez_compressed(path, dem_all=arr)
                            else:
                                np.savez(path, dem_all=arr)
                        dt_ = time.perf_counter() - t0_
                        log.info(
                            (
                                "Saved aggregated DEMs to %s (shape=%s, "
                                "compressed=%s, dtype=%s) in %.2fs"
                            ),
                            path,
                            tuple(shape),
                            str(do_compress),
                            dtype_str,
                            dt_,
                        )

                    # Opt-in background saving via MULTIGPU_ASYNC_SAVE=1
                    async_save = os.environ.get("MULTIGPU_ASYNC_SAVE", "1") == "1"
                    if async_save:
                        try:
                            import threading

                            th = threading.Thread(
                                target=_save_npz,
                                args=(
                                    final_path,
                                    dem_to_save.copy(),  # ensure buffer owns memory
                                    compress,
                                    dem_to_save.shape,
                                    str(dem_to_save.dtype),
                                ),
                                daemon=True,
                            )
                            th.start()
                            pending_save_threads.append(th)
                        except Exception:
                            _save_npz(
                                final_path,
                                dem_to_save,
                                compress,
                                dem_to_save.shape,
                                str(dem_to_save.dtype),
                            )
                    else:
                        _save_npz(
                            final_path,
                            dem_to_save,
                            compress,
                            dem_to_save.shape,
                            str(dem_to_save.dtype),
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

        # Promote preloaded next into pipeline buffer for next loop
        if pipeline and rank == 0:
            dn2d0, edn2d0, counts0 = next_dn2d, next_edn2d, next_counts
            dn_dtype_name0, edn_dtype_name0 = next_dtypes
        idx_file += 1

    # Ensure any outstanding async saves are finished before exit
    for th in pending_save_threads:
        try:
            th.join()
        except Exception:
            pass

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
