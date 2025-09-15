from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import dask
import dask.array as da
from dask.distributed import Client, LocalCluster

# Rigid IO & tiling contracts (as provided)
import src.common.io as io
from src.dask.tiling import dem_map_blocks, _set_constants

# Optional profiling hooks; tolerate absence
try:
    from src.common.profiling import bench_row, set_bench_outdir  # type: ignore
    from src.common.profiling import prepare_synthetic_responses  # for fallback
except Exception:  # pragma: no cover
    def set_bench_outdir(_outdir: str) -> None:  # type: ignore
        pass

    def bench_row(**_kw) -> None:  # type: ignore
        pass

    def prepare_synthetic_responses(
        logT_min=5.5, logT_max=7.5, n_tresp=200, nt=24, nf=6
    ):
        logT = np.linspace(logT_min, logT_max, n_tresp)
        centers = np.linspace(logT_min + 0.2, logT_max - 0.2, nf)
        width = 0.15
        T_RESP = np.exp(-0.5 * ((logT[:, None] - centers[None, :]) / width) ** 2) + 1e-30
        TEMPS = np.logspace(logT_min, logT_max, nt + 1)
        return T_RESP, logT, TEMPS


# ---------- memory tuning on workers ----------
def _tune_worker_memory(
    target: float = 0.55,
    spill: float = 0.65,
    pause: float = 0.80,
    terminate: float = 0.95,
):
    """
    Configure dask worker memory thresholds *on the worker*.

    target     -> start spilling/evicting to keep managed memory under this fraction
    spill      -> threshold to start spilling to disk
    pause      -> pause worker (stop taking new tasks)
    terminate  -> kill worker (hard safety)
    """
    import dask
    dask.config.set({
        "distributed.worker.memory.target": target,
        "distributed.worker.memory.spill": spill,
        "distributed.worker.memory.pause": pause,
        "distributed.worker.memory.terminate": terminate,
    })


# ---------- helpers ----------
def _parse_idx(idx: Union[str, int, slice, Sequence[int], None]) -> Union[int, slice, Sequence[int], None]:
    """
    Accept "-1", "0", "1,3,5", "10:20[:2]", or already-parsed values and
    return an int/slice/list usable by io.load_np_stack(..., idx=...).
    """
    if idx is None or isinstance(idx, (int, slice)) or isinstance(idx, (list, tuple)):
        return idx
    s = str(idx).strip()
    if not s:
        return None
    if ":" in s:
        bits = [b.strip() for b in s.split(":")]
        parts = [int(b) if b else None for b in bits]
        return slice(*parts)
    if "," in s:
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    return int(s)  # supports "-1"


def _target_hw_from_sizes(sizes: Tuple[int, ...], h0: int, w0: int) -> Tuple[int, int]:
    """
    Interpret the --sizes tuple for a single frame crop.
    - If two last numbers are H,W -> use those (clipped to data).
    - If only one number -> use symmetric LxL (clipped).
    - Otherwise fall back to existing H,W.
    """
    if not sizes:
        return h0, w0
    if len(sizes) >= 2:
        Ht, Wt = int(sizes[-2]), int(sizes[-1])
    else:
        Ht = Wt = int(sizes[-1])
    return min(Ht, h0), min(Wt, w0)


def _load_responses(nf: int = 6, nt_hint: int = 24):
    """
    Try to load real responses via src.common.dem_api.load_tresp(); on failure,
    fall back to synthetic responses (same shapes the tiler expects).
    """
    try:
        from src.common.dem_api import load_tresp  # type: ignore
        T_RESP, T_RESP_LOGT, TEMPS = load_tresp()
        return T_RESP, T_RESP_LOGT, TEMPS, int(len(TEMPS) - 1)
    except Exception:
        pass

    T_RESP, T_RESP_LOGT, TEMPS = prepare_synthetic_responses(nt=nt_hint, nf=nf)
    return T_RESP, T_RESP_LOGT, TEMPS, int(len(TEMPS) - 1)


# ---------- types ----------
@dataclass
class RunSummary:
    size: Tuple[int, int]
    tile: Tuple[int, int]
    wall_s: float
    n_workers: int
    threads_per_worker: int
    processes: bool


# ---------- local cluster helper ----------
def _start_cluster(
    n_workers: int,
    threads_per_worker: int,
    processes: bool,
    memory_limit: str | int | None,
    outdir: str,
) -> tuple[Client, LocalCluster]:
    """
    Start a local cluster with a per-run worker scratch dir so spilling is fast and isolated.
    """
    worker_space = Path(outdir) / "dask-worker-space"
    worker_space.mkdir(parents=True, exist_ok=True)

    # Make sure we don't inherit a too-aggressive fuse that can create giant tasks
    dask.config.set({"optimization.fuse.active": True})  # keep fusing but don't force it off

    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        processes=processes,
        memory_limit=memory_limit,
        dashboard_address=None,
        local_directory=str(worker_space),
    )
    client = Client(cluster)

    # Tweak the worker memory thresholds to start spilling earlier and pause before OOM
    client.run(_tune_worker_memory, target=0.55, spill=0.65, pause=0.80, terminate=0.95)

    return client, cluster


# ---------- main suite ----------
def run_dask_suite(
    *,
    # data selection / sizing
    use_synthetic: bool,                     # kept for interface parity; ignored (we load real files via io)
    data_dir: Optional[str],
    ext: str,
    idx: Union[str, int, slice, Sequence[int], None],
    sizes: Tuple[int, ...],
    # tiling
    tile: Optional[Tuple[int, int]] = None,
    tile_h: Optional[int] = None,
    tile_w: Optional[int] = None,
    # algorithm
    repeats: int = 3,
    nmu: int = 42,
    # cluster / scheduler
    scheduler: Optional[str] = None,
    n_workers: int = 4,
    threads_per_worker: int = 1,
    processes: bool = False,
    memory_limit: str | int | None = "8GB",
    # profiling outdir
    outdir: str = "benchmark_out",
) -> dict:
    """
    Build the dask graph once using src.dask.tiling.dem_map_blocks and benchmark it.

    Memory-safety measures:
      - per-worker memory thresholds tuned (spill earlier, pause before OOM)
      - per-run local_directory for fast spill
      - constants broadcast once with client.run(_set_constants, ...)
      - persist + reduction in the loop to avoid full materialization
    """
    set_bench_outdir(outdir)

    # normalize tile (smaller tiles lower peak; default 128x128 in CLI)
    if tile is None:
        tile = (int(tile_h or 128), int(tile_w or 128))
    th, tw = int(tile[0]), int(tile[1])

    # Client: connect or start local with memory tuning
    if scheduler:
        client = Client(scheduler)
        cluster = None
        # Even when connecting to an external scheduler, try to tune worker memory.
        # This is best-effort and will be a no-op if worker policies are locked down.
        try:
            client.run(_tune_worker_memory, target=0.55, spill=0.65, pause=0.80, terminate=0.95)
        except Exception:
            pass
    else:
        client, cluster = _start_cluster(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            processes=processes,
            memory_limit=memory_limit,
            outdir=outdir,
        )

    files_used: list[str] = []
    try:
        # ----------- load input frame (6, H, W) -----------
        files = io.default_files(ext=ext, directory=data_dir)
        idx_parsed = _parse_idx(idx)
        stack, paths = io.load_np_stack(
            files,
            idx=idx_parsed,
            dtype=np.float32,
            contiguous=True,
            return_paths=True,
        )  # (N,6,H,W)
        files_used = [str(p) for p in paths]

        if stack.ndim != 4 or stack.shape[1] != 6:
            raise ValueError(f"Expected (N,6,H,W) from load_np_stack, got {stack.shape}")

        frame_6hw = np.ascontiguousarray(stack[0])  # (6,H,W)
        H0, W0 = int(frame_6hw.shape[1]), int(frame_6hw.shape[2])

        # Optional crop according to --sizes
        Ht, Wt = _target_hw_from_sizes(sizes, H0, W0)
        if (Ht, Wt) != (H0, W0):
            frame_6hw = frame_6hw[:, :Ht, :Wt]
            H0, W0 = Ht, Wt

        # ----------- load & broadcast heavy constants -----------
        T_RESP, T_RESP_LOGT, TEMPS, nt = _load_responses(nf=6, nt_hint=24)
        client.run(_set_constants, T_RESP, T_RESP_LOGT, TEMPS, int(nmu))

        # ----------- build graph -----------
        # Use asarray=False to avoid copying the numpy into dask
        darr = da.from_array(frame_6hw, chunks=(6, th, tw), asarray=False)
        d_dem = dem_map_blocks(darr, nt=nt, tile_h=th, tile_w=tw)  # (H,W,nt)

        # small warm-up: touch a tiny slice; then drop it quickly
        _ = d_dem[: min(32, H0), : min(32, W0), :].sum().compute()

        # ----------- timed loop (persist + reduction) -----------
        t0 = time.perf_counter()
        for _ in range(int(repeats)):
            dd = d_dem.persist()
            # IMPORTANT: reduce immediately to avoid materializing all tiles in memory.
            # This keeps peak much lower and plays well with spilling thresholds.
            _ = dd.sum().compute()
            del dd
        wall = time.perf_counter() - t0

        bench_row(
            impl="dask",
            size=f"({H0},{W0})",
            tile=f"({th},{tw})",
            repeats=int(repeats),
            n_workers=int(n_workers),
            threads_per_worker=int(threads_per_worker),
            processes=int(bool(processes)),
            memory_limit=str(memory_limit),
            wall_s=round(float(wall), 4),
        )

        return {
            "size": (H0, W0),
            "tile": (th, tw),
            "wall_s": float(wall),
            "n_workers": int(n_workers),
            "threads_per_worker": int(threads_per_worker),
            "processes": bool(processes),
            "outdir": outdir,
            "files_used": files_used,
        }

    finally:
        try:
            client.close()
        except Exception:
            pass
        if 'cluster' in locals() and cluster is not None:
            try:
                cluster.close()
            except Exception:
                pass
