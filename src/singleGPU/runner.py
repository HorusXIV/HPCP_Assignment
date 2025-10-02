from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import logging
import subprocess
import os

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
        import cupy as cp  # type: ignore
        dev_id = int(cp.cuda.runtime.getDevice())
        props = cp.cuda.runtime.getDeviceProperties(dev_id)
        name = props.get("name", b"GPU").decode() if isinstance(props.get("name"), (bytes, bytearray)) else str(props.get("name", "GPU"))
        return f"id={dev_id} name={name}"
    except Exception:
        return "unknown"


def _gpu_utilization_once(index_hint: Optional[int] = None) -> Optional[dict]:
    # Prefer NVML if available
    try:
        import pynvml  # type: ignore
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

    # Fallback: nvidia-smi parsing
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

    # Optionally set CUPY device based on --device argument (e.g. 'cuda:0' or '0')
    if device is not None and device.lower() != "cpu":
        try:
            import cupy as cp  # type: ignore

            dev = 0
            if isinstance(device, str) and device.lower().startswith("cuda:"):
                dev = int(device.split(":", 1)[1])
            else:
                try:
                    dev = int(device)
                except Exception:
                    dev = int(cp.cuda.runtime.getDevice())
            cp.cuda.Device(dev).use()
        except Exception:
            # If cupy not available or device selection fails, continue and let gpu_ready report
            pass

    data_dir = data_dir or "data/np32"
    files = default_files(directory=data_dir, ext=ext)
    if not files:
        patterns = [ext] if isinstance(ext, str) else list(ext)
        raise FileNotFoundError(f"No files found in {data_dir} matching {patterns}")

    # Select frames
    if isinstance(idx, str):
        if idx.lower() in {"all", "-1"}:
            frame_idx: Sequence[int] = list(range(len(files)))
        else:
            i = _as_int(idx, 0)
            if not (0 <= i < len(files)):
                raise IndexError(f"--idx {i} out of range (0..{len(files) - 1})")
            frame_idx = [i]
    else:
        frame_idx = [int(i) for i in idx]
        for i in frame_idx:
            if not (0 <= i < len(files)):
                raise IndexError(f"--idx {i} out of range (0..{len(files) - 1})")

    # Determine crop size
    if sizes and len(sizes) >= 2:
        H, W = int(sizes[0]), int(sizes[1])
    elif sizes and len(sizes) == 1:
        H = W = int(sizes[0])
    else:
        _, H0, W0 = _npz_band_shape(Path(files[0]))
        H, W = H0, W0

    Th, Tw = _parse_tile_arg(tile)

    bench_dir, stamp = _ensure_timestamped_root(bench_root)

    def _load_frame_hw6(p: Path, H: int, W: int) -> np.ndarray:
        with np.load(p, allow_pickle=False) as z:
            bands = z["bands"].astype(np.float32)
        b = bands[:, :H, :W]
        return np.moveaxis(b, 0, -1)

    frames = [_load_frame_hw6(Path(files[i]), H, W) for i in frame_idx]
    stack = np.stack(frames, axis=0)  # (F, H, W, 6)
    F = int(stack.shape[0])

    # Initial GPU check & logging
    log = logging.getLogger(__name__)
    gpu_str = _gpu_device_summary()
    util0 = _gpu_utilization_once()
    util_str = (
        f"gpu={util0['gpu']}% mem={util0['mem']}% ({util0['mem_used_mb']}/{util0['mem_total_mb']} MB)"
        if util0 else "util=unknown"
    )
    log.info(
        "[singleGPU] starting compute: frames=%d crop=%dx%d tiles=%dx%d (GPU=%s, %s)",
        F, H, W, Th, Tw, gpu_str, util_str,
    )

    if not gpu_ready():
        print("[singleGPU] GPU not available; results will match CPU baseline performance.")

    from src.common.solver import get_logt_bins_once as _bins
    NT, _ = _bins(nmu=nmu)

    with Profiler(client=None, benchdir=bench_dir, stamp=stamp, enable_perf_html=False) as prof:
        prof.section("compute", start=True)

        # Prepare output arrays
        dem = np.empty((F, H, W, NT), dtype=np.float32)
        edem = np.empty_like(dem)
        chisq = np.empty((F, H, W), dtype=np.float32)

        # Try batch GPU path first (pass entire stack). If it fails, fall back to per-frame GPU or CPU.
        frame_t0 = time.perf_counter()
        try:
            # solve_tile_all_single_gpu accepts (F,H,W,6) or (H,W,6) and will use CuPy if available.
            _dem, _edem, _chisq, logT_centers = solve_tile_all_single_gpu(stack, nmu=nmu)
            # If returned single-frame shapes (shouldn't for batched input) coerce
            if _dem.ndim == 3 and _dem.shape[0] == F:
                dem[:, :, :, :] = _dem
                edem[:, :, :, :] = _edem
                chisq[:, :, :] = _chisq
            elif _dem.ndim == 3 and _dem.shape[0] != F:
                # Unexpected shape — try to broadcast / sanity-check
                raise RuntimeError("GPU returned unexpected shape for dem when processing full stack")
            else:
                # If vendor returned (pixels,NT) flattened data, attempt reshape
                try:
                    dem[:, :, :, :] = _dem.reshape((F, H, W, NT))
                    edem[:, :, :, :] = _edem.reshape((F, H, W, NT))
                    chisq[:, :, :] = _chisq.reshape((F, H, W))
                except Exception as re:
                    raise RuntimeError("Unable to reshape GPU outputs to (F,H,W,NT)") from re
        except Exception as e:
            # Log and fallback: try per-frame GPU calls or CPU solver if necessary.
            log.error("[singleGPU] batch GPU processing failed: %s", str(e))
            # Try per-frame GPU processing (calls same solver with (H,W,6))
            for fi in range(F):
                frame_t0_f = time.perf_counter()
                frame = stack[fi]
                try:
                    dem_f, edem_f, chisq_f, _ = solve_tile_all_single_gpu(frame, nmu=nmu)
                    dem[fi] = dem_f
                    edem[fi] = edem_f
                    chisq[fi] = chisq_f
                except Exception as e2:
                    # As last resort use CPU solver per-frame
                    log.error("[singleGPU] frame %d GPU failed: %s, falling back to CPU", fi, str(e2))
                    from src.common.solver import solve_tile_all as _solve_cpu
                    dem_f, edem_f, chisq_f, _ = _solve_cpu(frame, nmu=nmu, nt=NT)
                    dem[fi] = dem_f
                    edem[fi] = edem_f
                    chisq[fi] = chisq_f
                frame_dt_f = time.perf_counter() - frame_t0_f
                # per-frame logging
                tiles_per_frame = int(np.ceil(H / Th) * np.ceil(W / Tw))
                ms_per_tile = (frame_dt_f / tiles_per_frame) * 1e3
                util = _gpu_utilization_once()
                if util:
                    log.info(
                        "[singleGPU] frame %d done: time=%.3fs tiles/frame=%d ms/tile=%.2f | gpu=%d%% mem=%d%% (%d/%d MB)",
                        fi, frame_dt_f, tiles_per_frame, ms_per_tile,
                        util["gpu"], util["mem"], util["mem_used_mb"], util["mem_total_mb"]
                    )
                else:
                    log.info("[singleGPU] frame %d done: time=%.3fs tiles/frame=%d ms/tile=%.2f",
                             fi, frame_dt_f, tiles_per_frame, ms_per_tile)

        frame_dt = time.perf_counter() - frame_t0

        # Logical tiles for reporting
        tiles_per_frame = int(np.ceil(H / Th) * np.ceil(W / Tw))
        tiles_total = tiles_per_frame * F
        ms_per_tile = (frame_dt / tiles_total) * 1e3 if tiles_total > 0 else float("nan")
        tiles_per_s = tiles_total / frame_dt if frame_dt > 0 else float("nan")

        util1 = _gpu_utilization_once()
        if util1:
            log.info(
                "[singleGPU] finished: frames=%d total_tiles=%d time=%.3fs tiles/s=%.2f ms/tile=%.2f | gpu=%d%% mem=%d%% (%d/%d MB)",
                F, tiles_total, frame_dt, tiles_per_s, ms_per_tile,
                util1["gpu"], util1["mem"], util1["mem_used_mb"], util1["mem_total_mb"]
            )
        else:
            log.info(
                "[singleGPU] finished: frames=%d total_tiles=%d time=%.3fs tiles/s=%.2f ms/tile=%.2f",
                F, tiles_total, frame_dt, tiles_per_s, ms_per_tile
            )

        prof.section("compute", start=False)

    # ---------------- persist outputs ----------------
    outputs_path = bench_dir / f"outputs_{stamp}.npz"
    np.savez_compressed(outputs_path, dem=dem, edem=edem, chisq=chisq)

    verify_ok = bool(verify)
    reports: List[str] = []
    if verify and golden_root:
        reports.append("Verification placeholder — integrate golden check here.")
        verify_ok = True

    tiles_per_frame = int(np.ceil(H / Th) * np.ceil(W / Tw))
    tiles_total = tiles_per_frame * F
    total_s = float((getattr(prof, "_sections", {}).get("compute", (0.0, 0.0))[1]))
    tiles_per_s = tiles_total / total_s if total_s > 0 else float("nan")
    dems_per_s = (F * H * W) / total_s if total_s > 0 else float("nan")

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

    log.info(
        "[singleGPU] summary: frames=%d tiles/frame=%d crop=%dx%d total=%.3fs tiles/s=%.2f dems/s=%.2f",
        F,
        tiles_per_frame,
        H,
        W,
        bench["total_seconds"],
        tiles_per_s,
        dems_per_s,
    )
    print(
        f"[SingleGPU] frames={F} tiles/frame={tiles_per_frame} crop={H}x{W} "
        f"total={bench['total_seconds']:.3f}s -> artifacts: {bench_dir}"
    )

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
