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

    comm, rank, size = mmpi.init_mpi()
    # Setup results directory structure
    results_root = "src/multiGPU/results"
    # Results will be written into the shared `aggregate` folder under
    # `results_root` (no per-rank subdirectories).
    if rank == 0:
        # ensure parent exists
        import os

        os.makedirs(results_root, exist_ok=True)

    # Initialize logging (creates logs/ under results_root)
    _ = mlog.setup_logging(results_root, rank=rank, size=size)
    log = logging.getLogger(__name__)
    log.info(f"Starting rank {rank}/{size-1}; results dir: {results_root}")

    # Map rank to GPU and bind environment (prefer per-node local-rank mapping)
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
    import os
    import glob

    if not args.input_dir:
        if rank == 0:
            raise RuntimeError(
                "Must specify --input-dir to process input files"
            )
    else:
        # Rank 0 enumerates files and distributes the list to all ranks
        if rank == 0:
            pattern = os.path.join(args.input_dir, "*.npz")
            all_inputs = sorted(glob.glob(pattern))
            if len(all_inputs) == 0:
                raise RuntimeError(f"No .npz files found in {args.input_dir}")
        else:
            all_inputs = None

        # Broadcast the list of inputs to all ranks. The serial stub will get
        # None which is handled below.
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
    for input_path in all_inputs:
        # Announce input start and measure elapsed time
        log.info(f"Rank {rank}: starting processing input {input_path}")
        try:
            # Rank 0 loads and prepares the arrays; other ranks wait to
            # participate in the collective scatterv. This ensures a
            # consistent collective ordering across ranks.
            if rank == 0:
                data = mio.load_npz(input_path)

                # Heuristics: prefer `dn`/`edn`; otherwise `bands` layout.
                dn = data.get("dn", None)
                edn = data.get("edn", None)
                if dn is None:
                    if "bands" in data:
                        bands = data["bands"]
                        if bands.ndim != 3:
                            raise RuntimeError(
                                "Unexpected `bands` shape; expected (nf, ny, nx)"
                            )
                        nf, ny, nx = bands.shape
                        n_pixels = ny * nx
                        max_samples = args.max_samples
                        bands_flat = bands.reshape(nf, n_pixels)
                        if max_samples is not None and n_pixels > max_samples:
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
                                "Input file missing required dn and edn arrays"
                            )

                dn2d = mio.ensure_2d_dn(dn)
                edn2d = mio.ensure_2d_dn(edn)
                if edn2d.shape[0] != dn2d.shape[0]:
                    if edn2d.shape[0] == 1:
                        edn2d = np.repeat(edn2d, dn2d.shape[0], axis=0)
                    else:
                        raise RuntimeError(
                            "edn shape does not match dn and cannot be broadcast"
                        )

                if rank == 0:
                    print(
                        "dn2d.shape=%s, edn2d.shape=%s"
                        % (str(dn2d.shape), str(edn2d.shape))
                    )

                n_samples = dn2d.shape[0]
                counts = [
                    n_samples // size + (1 if i < (n_samples % size) else 0)
                    for i in range(size)
                ]
            else:
                # non-root ranks start with placeholders; they'll receive
                # their slices via the collective scatterv call below.
                dn2d = None
                edn2d = None
                n_samples = None
                counts = None

            # Broadcast counts so all ranks know local sizes for this input
            if comm is not None:
                counts = comm.bcast(counts, root=0)

                # Broadcast dtypes as strings so non-root ranks can pass a
                # valid dtype argument to scatterv_array. Reconstruct a
                # numpy.dtype locally on each rank.
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

                local_dn = mmpi.scatterv_array(
                    comm,
                    dn2d if rank == 0 else None,
                    counts,
                    dtype=dn_dtype,
                )
                local_edn = mmpi.scatterv_array(
                    comm,
                    edn2d if rank == 0 else None,
                    counts,
                    dtype=edn_dtype,
                )
            else:
                # serial fallback: local arrays are the full arrays
                local_dn = dn2d
                local_edn = edn2d
                counts = [n_samples]

            # build rmatrix like dn2dem_pos does (simple path)
            # for demonstration use a small synthetic rmatrix based on filters
            nf = local_dn.shape[1]
            nt = 10
            # create log-temperature centers and widths
            logt = np.linspace(5.0, 7.0, nt)
            dlogt = np.full(nt, logt[1] - logt[0])
            tresp = np.ones((nt, nf))

            # Enforce CuPy presence and GPU assignment; fail loudly if missing
            try:
                # ensure an informative ImportError is raised via the helper
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
                        "CUDA_VISIBLE_DEVICES or N_GPUS is set and GPUs are "
                        "available."
                    )
                )

            # process local pixels in blocks to amortize GPU setup costs
            block = 128
            dem_local = np.zeros((local_dn.shape[0], nt))
            edem_local = np.zeros_like(dem_local)
            elogt_local = np.zeros_like(dem_local)
            chisq_local = np.zeros((local_dn.shape[0],))
            dn_reg_local = np.zeros((local_dn.shape[0], nf))

            for i in range(0, local_dn.shape[0], block):
                i2 = min(local_dn.shape[0], i + block)
                sub_dn = local_dn[i:i2]
                sub_edn = local_edn[i:i2]
                dem_b, edem_b, elogt_b, chisq_b, dnreg_b = (
                    gpu_kernels.demmap_pos(
                        sub_dn, sub_edn, tresp, logt, dlogt, np.ones(nf)
                    )
                )
                dem_local[i:i2] = dem_b
                edem_local[i:i2] = edem_b
                elogt_local[i:i2] = elogt_b
                chisq_local[i:i2] = chisq_b
                dn_reg_local[i:i2] = dnreg_b

            log.info(
                "Rank %d: processed %d pixels on GPU %s",
                rank,
                local_dn.shape[0],
                str(gpu_assigned),
            )
            log.info(
                "Rank %d: finished input %s (GPU)",
                rank,
                input_path,
            )

            # Gather results to root for final aggregation
            if comm is not None:
                dem_all = mmpi.gatherv_array(comm, dem_local, counts, root=0)
                # Ensure all ranks reach the same point before root writes file
                try:
                    mmpi.barrier(comm)
                except Exception as e:
                    # barrier failure doesn't block saving; log it
                    log.exception(
                        "Rank %d: MPI barrier failed after gather: %s",
                        rank,
                        e,
                    )

                if rank == 0:
                    # Basic sanity check: gathered rows must equal sum(counts)
                    expected = int(sum(counts))
                    # If gatherv returned None or a mismatched shape, attempt
                    # to recover by reading per-rank local files (shared FS)
                    if dem_all is None or dem_all.shape[0] != expected:
                        log.warning(
                            (
                                "Gathered DEMs missing or size-mismatched; "
                                "attempting fallback"
                            )
                        )
                        # Try reading per-rank local outputs and concatenating
                        try:
                            parts = []
                            # Look for per-rank files in the aggregate folder.
                            # Filenames: dem_local_r{rank:03d}_{inbase}.npz
                            agg_dir = os.path.join(results_root, "aggregate")
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
                                    "Fallback aggregation produced %d rows; "
                                    "expected %d"
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
                                "Fallback aggregation failed; aborting save"
                            )
                            raise

                    print(f"Computed total DEMs: {dem_all.shape[0]}")
                    # Save aggregated results to root results dir (per-input)

                    os.makedirs(f"{results_root}/aggregate", exist_ok=True)
                    inbase = os.path.splitext(os.path.basename(input_path))[0]
                    final_path = os.path.join(
                        f"{results_root}/aggregate", f"dem_all_{inbase}.npz"
                    )

                    np.savez_compressed(final_path, dem_all=dem_all)

            else:
                print(f"Computed total DEMs: {dem_local.shape[0]}")
                import os

                os.makedirs(f"{results_root}/aggregate", exist_ok=True)
                inbase = os.path.splitext(os.path.basename(input_path))[0]
                final_path = os.path.join(
                    f"{results_root}/aggregate", f"dem_all_{inbase}.npz"
                )

                np.savez_compressed(final_path, dem_all=dem_all)

                log.info(f"Saved local DEMs to {final_path}")

        except Exception:
            log.exception(
                "Rank %d: exception while processing %s",
                rank,
                input_path,
            )
            raise

    # Clean shutdown of logging to ensure all records are flushed
    mlog.shutdown_logging()


if __name__ == "__main__":
    main()
