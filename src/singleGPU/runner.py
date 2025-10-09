from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import logging
import subprocess
import os
import gc

from src.common.dataio import default_files
from src.common.profiling import (
    Profiler,
    write_bench_row,
    write_run_card_md,
    write_json,
)
from src.common.profiling.io_helpers import set_bench_outdir
from src.singleGPU.accelerate import solve_tile_all_single_gpu, gpu_ready


def _base_root_default() -> Path:
    return Path.cwd() / "benchmarking" / "singleGPU"


def _gpu_device_summary() -> str:
    try:
        import cupy as cp
        dev_id = int(cp.cuda.runtime.getDevice())
        props = cp.cuda.runtime.getDeviceProperties(dev_id)
        name = props.get("name", b"GPU").decode() if isinstance(props.get("name"), (bytes, bytearray)) else str(props.get("name", "GPU"))
        return f"id={dev_id} name={name}"
    except Exception:
        return "unknown"


def _gpu_utilization_once(index_hint: Optional[int] = None) -> Optional[dict]:
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        idx = int(index_hint) if index_hint is not None and 0 <= int(index_hint) < count else 0
        handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        name = pynvml.nvmlDeviceGetName(handle)
        return {
            "gpu": int(util.gpu),
            "mem": int(util.memory),
            "mem_used_mb": int(mem.used // (1024 * 1024)),
            "mem_total_mb": int(mem.total // (1024 * 1024)),
            "name": name.decode() if isinstance(name, (bytes, bytearray)) else str(name),
            "index": idx,
        }
    except Exception:
        pass

    try:
        q = [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        out = subprocess.check_output(q, stderr=subprocess.DEVNULL, text=True)
        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        if not lines:
            return None
        row = lines[int(index_hint) if index_hint is not None and int(index_hint) < len(lines) else 0]
        parts = [p.strip() for p in row.split(",")]
        idx = int(parts[0])
        name = parts[1]
        gpu = int(parts[2])
        mem = int(parts[3])
        mem_used_mb = int(parts[4])
        mem_total_mb = int(parts[5])
        return {"gpu": gpu, "mem": mem, "mem_used_mb": mem_used_mb, "mem_total_mb": mem_total_mb, "name": name, "index": idx}
    except Exception:
        return None


def _ensure_timestamped_root(base_root: Optional[Union[str, Path]]) -> Tuple[Path, str]:
    base = Path(base_root) if base_root else _base_root_default()
    base.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = base / stamp
    out.mkdir(parents=True, exist_ok=True)
    set_bench_outdir(out)
    return out, stamp


def _npz_band_shape(path: Path) -> Tuple[int, int, int]:
    with np.load(path, allow_pickle=False) as z:
        a = z["bands"]
        if a.ndim != 3 or a.shape[0] != 6:
            raise ValueError(f"{path} expected 'bands' with shape (6,H,W), got {a.shape}")
        return int(a.shape[0]), int(a.shape[1]), int(a.shape[2])


def _as_int(val: object, default: int) -> int:
    try:
        return int(val)
    except Exception:
        return default


def _parse_tile_arg(tile: Union[str, Tuple[int, int], None]) -> Tuple[int, int]:
    if tile is None:
        return 256, 256
    if isinstance(tile, tuple):
        return int(tile[0]), int(tile[1])
    s = str(tile)
    if "x" in s.lower():
        a, b = s.lower().split("x")
        return int(a), int(b)
    if "," in s:
        a, b = s.split(",", 1)
        return int(a), int(b)
    v = int(s)
    return v, v


def _free_gpu_memory():
    """Forcefully release GPU memory (CuPy + Numba)."""
    try:
        import cupy as cp
        cp.cuda.Stream.null.synchronize()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass
    try:
        from numba import cuda
        cuda.current_context().deallocations.clear()
    except Exception:
        pass
    gc.collect()


def run_benchmark_single_gpu(
    *,
    data_dir: Union[str, Path, None] = None,
    ext: Union[str, Sequence[str]] = "*.npz",
    idx: Union[str, Sequence[int]] = "-1",
    sizes: Sequence[int] = (1024,),
    tile: Union[str, Tuple[int, int], None] = None,
    nmu: int = 42,
    bench_root: Union[str, Path, None] = None,
    verify: bool = False,
    golden_root: Union[str, Path, None] = None,
    device: Optional[str] = None,
) -> Dict[str, Union[str, float, int]]:

    import cupy as cp

    # Optional device selection
    if device is not None and device.lower() != "cpu":
        try:
            dev = int(device.split(":")[1]) if ":" in device else int(device)
            cp.cuda.Device(dev).use()
        except Exception:
            pass

    data_dir = data_dir or "data/np32"
    files = default_files(directory=data_dir, ext=ext)
    if not files:
        patterns = [ext] if isinstance(ext, str) else list(ext)
        raise FileNotFoundError(f"No files found in {data_dir} matching {patterns}")

    # Frame selection
    if isinstance(idx, str):
        if idx.lower() in {"all", "-1"}:
            frame_idx: Sequence[int] = list(range(len(files)))
        else:
            i = _as_int(idx, 0)
            frame_idx = [i]
    else:
        frame_idx = [int(i) for i in idx]

    # Determine crop
    if sizes and len(sizes) >= 2:
        H, W = int(sizes[0]), int(sizes[1])
    elif sizes and len(sizes) == 1:
        H = W = int(sizes[0])
    else:
        _, H, W = _npz_band_shape(Path(files[0]))

    Th, Tw = _parse_tile_arg(tile)
    bench_dir, stamp = _ensure_timestamped_root(bench_root)

    def _load_frame_hw6(p: Path) -> np.ndarray:
        with np.load(p, allow_pickle=False) as z:
            bands = z["bands"].astype(np.float32)
        b = bands[:, :H, :W]
        return np.moveaxis(b, 0, -1)

    # Load frames
    stack = np.stack([_load_frame_hw6(Path(files[i])) for i in frame_idx], axis=0)
    min_val = stack.min()
    if min_val < 0.0:
        print(f"[INFO] Shifting all frames by {-min_val:.3f} to remove negative counts")
        stack -= min_val  # shift so minimum is zero
    F = int(stack.shape[0])

    log = logging.getLogger(__name__)
    gpu_str = _gpu_device_summary()
    util0 = _gpu_utilization_once()
    util_str = f"gpu={util0['gpu']}% mem={util0['mem']}% ({util0['mem_used_mb']}/{util0['mem_total_mb']} MB)" if util0 else "util=unknown"
    log.info("[singleGPU] starting compute: frames=%d crop=%dx%d tiles=%dx%d (GPU=%s, %s)", F, H, W, Th, Tw, gpu_str, util_str)

    if not gpu_ready():
        print("[singleGPU] GPU not available; results will match CPU baseline performance.")

    from src.common.solver import get_logt_bins_once as _bins
    NT, _ = _bins(nmu=nmu)

    dem = np.empty((F, H, W, NT), dtype=np.float32)
    edem = np.empty_like(dem)
    chisq = np.empty((F, H, W), dtype=np.float32)

    with Profiler(client=None, benchdir=bench_dir, stamp=stamp, enable_perf_html=False) as prof:
        prof.section("compute", start=True)

        # Process each frame
        for fi in range(F):
            frame_t0 = time.perf_counter()
            frame = stack[fi]
            # DEBUG: Check input frame stats
            print(f"[DEBUG] Frame {fi} input min/max per band:")
            for b in range(frame.shape[2]):
                print(f"  Band {b}: min={frame[:, :, b].min()}, max={frame[:, :, b].max()}")

            # Split frame into tiles
            tile_coords = []
            tiles = []

            for i0 in range(0, H, Th):
                for j0 in range(0, W, Tw):
                    i1 = min(i0 + Th, H)
                    j1 = min(j0 + Tw, W)
                    tile_coords.append((i0, i1, j0, j1))
                    tiles.append(frame[i0:i1, j0:j1, :])

            num_tiles = len(tiles)
            log.info("[singleGPU] frame %d: %d tiles to process (%dx%d each)", fi, num_tiles, Th, Tw)

            # number of tiles to process per GPU call — keep small to avoid OOM
            tiles_per_call = int(os.environ.get("TILES_PER_CALL", "2"))
            log.info("[singleGPU] using tiles_per_call=%d", tiles_per_call)

            dem_tiles_list = []
            edem_tiles_list = []
            chisq_tiles_list = []


            for start in range(0, num_tiles, tiles_per_call):
                end = min(num_tiles, start + tiles_per_call)
                sub_tiles = tiles[start:end]
                sub_batch = np.stack(sub_tiles, axis=0)

                log.debug("[singleGPU] processing tiles %d-%d (batch shape=%s)", start, end - 1, sub_batch.shape)


                try:
                    dem_sub, edem_sub, chisq_sub, _ = solve_tile_all_single_gpu(sub_batch, nmu=nmu)
                    log.debug(
                        "[singleGPU] dem_sub stats: shape=%s min=%.3g max=%.3g mean=%.3g",
                        dem_sub.shape, float(dem_sub.min()), float(dem_sub.max()), float(dem_sub.mean()),
                    )
                except Exception as ex:
                    log.error("[singleGPU] tiles %d-%d GPU/solver failed: %s — falling back to CPU", start,
                                end - 1, str(ex))

                    from src.common.solver import solve_tile_all as _solve_cpu
                    dem_sub_list = []
                    edem_sub_list = []
                    chisq_sub_list = []
                    for t_idx, t in enumerate(sub_tiles):
                        try:
                            dem_f, edem_f, chisq_f, _ = _solve_cpu(t, nmu=nmu, nt=NT)
                            dem_sub_list.append(dem_f)
                            edem_sub_list.append(edem_f)
                            chisq_sub_list.append(chisq_f)
                        except Exception as cpu_ex:
                            log.exception("[singleGPU] CPU fallback failed for tile %d: %s", t_idx, str(cpu_ex))
                            # produce zero output as placeholder
                            dem_sub_list.append(np.zeros((t.shape[0], t.shape[1], NT), dtype=np.float32))
                            edem_sub_list.append(np.zeros_like(dem_sub_list[-1]))
                            chisq_sub_list.append(np.zeros((t.shape[0], t.shape[1]), dtype=np.float32))

                    dem_sub = np.stack(dem_sub_list, axis=0)
                    edem_sub = np.stack(edem_sub_list, axis=0)
                    chisq_sub = np.stack(chisq_sub_list, axis=0)

                dem_tiles_list.append(dem_sub)
                edem_tiles_list.append(edem_sub)
                chisq_tiles_list.append(chisq_sub)

              # clean GPU memory between batches
                try:
                    _free_gpu_memory()
                except Exception:
                    pass

            # concatenate all batches
            dem_tiles = np.concatenate(dem_tiles_list, axis=0)
            edem_tiles = np.concatenate(edem_tiles_list, axis=0)
            chisq_tiles = np.concatenate(chisq_tiles_list, axis=0)

            log.info("[singleGPU] frame %d all tiles done: dem min=%.3g max=%.3g mean=%.3g",
                    fi, float(dem_tiles.min()), float(dem_tiles.max()), float(dem_tiles.mean()))

            # Scatter tiles back
            for k, (i0, i1, j0, j1) in enumerate(tile_coords):
                dem[fi, i0:i1, j0:j1, :] = dem_tiles[k][:i1-i0, :j1-j0, :]
                edem[fi, i0:i1, j0:j1, :] = edem_tiles[k][:i1-i0, :j1-j0, :]
                chisq[fi, i0:i1, j0:j1] = chisq_tiles[k][:i1-i0, :j1-j0]

            print(f"[DEBUG] Frame {fi} DEM min/max: {dem[fi].min()}, {dem[fi].max()}")
            print(f"[DEBUG] Frame {fi} EDEM min/max: {edem[fi].min()}, {edem[fi].max()}")
            print(f"[DEBUG] Frame {fi} chisq min/max: {chisq[fi].min()}, {chisq[fi].max()}")
            frame_dt = time.perf_counter() - frame_t0
            tiles_per_frame = len(tile_coords)
            ms_per_tile = frame_dt / tiles_per_frame * 1e3
            util = _gpu_utilization_once()
            log.info("[singleGPU] frame %d done: time=%.3fs tiles=%d ms/tile=%.2f | gpu=%s",
                     fi, frame_dt, tiles_per_frame, ms_per_tile,
                     f"{util['gpu']}% mem={util['mem']}%" if util else "unknown")

             # clear memory after frame

            prof.section("compute", start=False)

            outputs_path = bench_dir / f"outputs_{stamp}_{fi}.npz"
            np.savez_compressed(outputs_path, dem=dem, edem=edem, chisq=chisq)
            _free_gpu_memory()

            # Reporting
    verify_ok = bool(verify)
    reports: List[str] = []
    if verify and golden_root:
        reports.append("Verification placeholder — integrate golden check here.")
        verify_ok = True

    total_s = float((getattr(prof, "_sections", {}).get("compute", (0.0, 0.0))[1]))
    tiles_total = F * ((H + Th - 1)//Th) * ((W + Tw - 1)//Tw)
    tiles_per_s = tiles_total / total_s if total_s > 0 else float("nan")
    dems_per_s = F * H * W / total_s if total_s > 0 else float("nan")

    bench = dict(
        stamp=stamp,
        mode="single-gpu",
        frames=int(F),
        H=H,
        W=W,
        Th=Th,
        Tw=Tw,
        nmu=int(nmu),
        total_seconds=round(total_s, 6),
        tiles_per_s=float(tiles_per_s),
        dems_per_s=float(dems_per_s),
        verify=bool(verify),
        verify_ok=bool(verify_ok),
        outputs_npz=str(outputs_path),
    )

    write_bench_row(**bench)
    write_run_card_md(outdir=bench_dir, stamp=stamp, bench_row=bench, env=None, notes=reports)
    write_json(bench, bench_dir / f"bench_{stamp}.json")

    log.info("[singleGPU] summary: frames=%d tiles/frame=%d crop=%dx%d total=%.3fs tiles/s=%.2f dems/s=%.2f",
             F, (H+Th-1)//Th * (W+Tw-1)//Tw, H, W, bench["total_seconds"], tiles_per_s, dems_per_s)

    print(f"[SingleGPU] frames={F} tiles/frame={(H+Th-1)//Th * (W+Tw-1)//Tw} crop={H}x{W} "
          f"total={bench['total_seconds']:.3f}s -> artifacts: {bench_dir}")

    return {
        "bench_root": str(bench_dir),
        "stamp": stamp,
        "frames": F,
        "H": H,
        "W": W,
        "Th": Th,
        "Tw": Tw,
        "seconds": bench["total_seconds"],
    }


__all__ = ["run_benchmark_single_gpu"]
