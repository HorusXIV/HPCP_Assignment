from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import dask.array as da
from dask.distributed import Client, LocalCluster

# rigid IO & tiling contracts
import src.common.io as io
from src.dask.tiling import dem_map_blocks, _set_constants

# profiling hooks
try:
    from src.common.profiling import bench_row, set_bench_outdir, flush_bench_csv  # type: ignore
except Exception:  # pragma: no cover
    def set_bench_outdir(_outdir: str) -> None:  # type: ignore
        pass
    def bench_row(**_kw) -> None:  # type: ignore
        pass
    def flush_bench_csv() -> None:  # type: ignore
        pass


# ---------- helpers ----------
def _parse_idx(idx: Union[str, int, slice, Sequence[int], None]) -> Union[int, slice, Sequence[int], None]:
    """
    Accepts "-1", "0", "1,3,5", "10:20", or already-parsed values and
    returns an int/slice/list usable by io.load_np_stack(..., idx=...).
    """
    if idx is None or isinstance(idx, (int, slice)) or isinstance(idx, (list, tuple)):
        return idx
    s = str(idx).strip()
    if not s:
        return None
    if ":" in s:
        # slice form a:b[:c]
        bits = [b.strip() for b in s.split(":")]
        parts = [int(b) if b else None for b in bits]
        return slice(*parts)
    if "," in s:
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    # plain integer (supports "-1")
    return int(s)


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


def _load_tresp_any():
    """
    Try several places for thermal response matrices.

    Returns
    -------
    (T_RESP, T_RESP_LOGT, TEMPS)
      T_RESP:         (n_logT, nf)
      T_RESP_LOGT:    (n_logT,)
      TEMPS:          (nt+1,)   temperature *edges* in Kelvin (len-1 == nt)
    """
    # 1) src.common.dem_api.load_tresp
    try:
        import src.common.dem_api as dem_api  # type: ignore
        if hasattr(dem_api, "load_tresp") and callable(getattr(dem_api, "load_tresp")):
            return dem_api.load_tresp()
    except Exception:
        pass

    # 2) src.common.responses.load_tresp
    try:
        import src.common.responses as resp  # type: ignore
        if hasattr(resp, "load_tresp") and callable(getattr(resp, "load_tresp")):
            return resp.load_tresp()
    except Exception:
        pass

    # 3) src.common.responses.prepare_synthetic_responses -> (T_RESP, logT, TEMPS)
    try:
        import src.common.responses as resp  # type: ignore
        if hasattr(resp, "prepare_synthetic_responses"):
            T_RESP, logT, TEMPS = resp.prepare_synthetic_responses()
            return T_RESP, np.asarray(logT), np.asarray(TEMPS)
    except Exception:
        pass

    raise ImportError(
        "Could not load response matrices. Expected one of:\n"
        " - src.common.dem_api.load_tresp()\n"
        " - src.common.responses.load_tresp()\n"
        " - src.common.responses.prepare_synthetic_responses()\n"
        "Please implement one of these loaders."
    )


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
    memory_limit: str,
    local_directory: str,
) -> tuple[Client, LocalCluster]:
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        processes=processes,
        memory_limit=memory_limit,
        dashboard_address=None,
        local_directory=local_directory,  # put worker scratch inside outdir
    )
    return Client(cluster), cluster


# ---------- main suite ----------
def run_dask_suite(
    *,
    # data selection / sizing
    use_synthetic: bool,                     # kept for interface parity; ignored (we load real files)
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
    memory_limit: str = "8GB",
    # profiling/artifacts
    outdir: str = "benchmark_out",
) -> dict:
    """
    Build the Dask graph once using src.dask.tiling.dem_map_blocks and benchmark it.

    Only depends on:
      - src.common.io.{default_files, load_np_stack}
      - src.dask.tiling.{dem_map_blocks, _set_constants}
      - response loader from _load_tresp_any()
    """
    outdir_p = Path(outdir).resolve()
    outdir_p.mkdir(parents=True, exist_ok=True)

    # profiling sink
    set_bench_outdir(outdir_p)

    # normalize tile
    if tile is None:
        tile = (int(tile_h or 256), int(tile_w or 256))
    th, tw = int(tile[0]), int(tile[1])

    # Client: connect or start local
    if scheduler:
        client = Client(scheduler)
        cluster = None
    else:
        client, cluster = _start_cluster(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            processes=processes,
            memory_limit=str(memory_limit),
            local_directory=str(outdir_p / "dask-worker-space"),
        )

    files_used: list[str] = []
    try:
        # ----------- load input frame (6, H, W) -----------
        files = io.default_files(ext=ext, directory=data_dir)
        idx_parsed = _parse_idx(idx)
        stack, paths = io.load_np_stack(files, idx=idx_parsed, dtype=np.float32, contiguous=True, return_paths=True)  # (N,6,H,W)
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
        T_RESP, T_RESP_LOGT, TEMPS = _load_tresp_any()
        if T_RESP.shape[1] != 6:
            # Friendly error mirroring what you saw from vendor code
            raise ValueError("Tresp needs to be the same number of wavelengths/filters as the data (nf=6).")
        nt = int(len(TEMPS) - 1)
        client.run(_set_constants, T_RESP, T_RESP_LOGT, TEMPS, int(nmu))

        # ----------- build graph -----------
        darr = da.from_array(frame_6hw, chunks=(6, th, tw), asarray=False)
        d_dem = dem_map_blocks(darr, nt=nt, tile_h=th, tile_w=tw)  # (H,W,nt)

        # small warm-up to catch issues early, minimal materialization
        _ = d_dem[: min(32, H0), : min(32, W0), :].sum().compute()

        # ----------- timed loop -----------
        t0 = time.perf_counter()
        for _ in range(int(repeats)):
            dd = d_dem.persist()
            _ = dd.sum().compute()  # low-peak reduction; avoids materializing full (H,W,nt)
        wall = time.perf_counter() - t0

        # ----------- write one bench row -----------
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
        flush_bench_csv()  # ensure file is present on disk right away

        # ----------- summary artifacts -----------
        (outdir_p / "RUN_SUMMARY.txt").write_text(
            f"Size=({H0}, {W0})  Tile=({th}, {tw})  Wall(s)={wall:.3f}  "
            f"Workers={n_workers}  TPW={threads_per_worker}  Processes={processes}\n",
            encoding="utf-8",
        )
        (outdir_p / "run_summary.json").write_text(
            json.dumps(
                {
                    "size": [H0, W0],
                    "tile": [th, tw],
                    "wall_s": round(float(wall), 6),
                    "n_workers": int(n_workers),
                    "threads_per_worker": int(threads_per_worker),
                    "processes": bool(processes),
                    "memory_limit": str(memory_limit),
                    "files_used": files_used,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        return {
            "outdir": str(outdir_p),
            "files_used": files_used,
            "size": (H0, W0),
            "tile": (th, tw),
            "wall_s": float(wall),
            "n_workers": int(n_workers),
            "threads_per_worker": int(threads_per_worker),
            "processes": bool(processes),
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
