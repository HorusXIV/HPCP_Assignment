from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
import logging
import subprocess

import numpy as np

from src.common.dataio import default_files
from src.common.profiling import Profiler, write_bench_row, write_run_card_md, write_json
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
    # fallback
    try:
        q = ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total",
             "--format=csv,noheader,nounits"]
        out = subprocess.check_output(q, stderr=subprocess.DEVNULL, text=True)
        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        if not lines:
            return None
        row = lines[int(index_hint) if index_hint else 0]
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

    log = logging.getLogger(__name__)

    # Set device
    if device is not None and device.lower() != "cpu":
        try:
            import cupy as cp  # type: ignore
            dev = int(device.split(":")[1]) if ":" in device else int(device)
            cp.cuda.Device(dev).use()
        except Exception:
            pass

    data_dir = data_dir or "data/np32"
    files = default_files(directory=data_dir, ext=ext)
    if not files:
        raise FileNotFoundError(f"No files found in {data_dir} matching {ext}")

    # Frame selection
    if isinstance(idx, str) and idx.lower() in {"all", "-1"}:
        frame_idx = list(range(len(files)))
    else:
        frame_idx = [int(i) for i in idx] if isinstance(idx, (list, tuple)) else [int(idx)]
    
    # Crop sizes
    if len(sizes) >= 2:
        H, W = int(sizes[0]), int(sizes[1])
    elif len(sizes) == 1:
        H = W = int(sizes[0])
    else:
        _, H, W = _npz_band_shape(Path(files[0]))

    Th, Tw = _parse_tile_arg(tile)
    bench_dir, stamp = _ensure_timestamped_root(bench_root)

    # Load frames and stack
    def _load_frame(p: Path) -> np.ndarray:
        with np.load(p, allow_pickle=False) as z:
            b = z["bands"].astype(np.float32)[:, :H, :W]
        return np.moveaxis(b, 0, -1)  # H,W,6

    stack = [_load_frame(Path(files[i])) for i in frame_idx]
    F = len(stack)

    if not gpu_ready():
        print("[singleGPU] GPU not available; results will match CPU baseline performance.")

    from src.common.solver import get_logt_bins_once
    NT, _ = get_logt_bins_once(nmu=nmu)

    # Output arrays
    dem = np.empty((F, H, W, NT), dtype=np.float32)
    edem = np.empty_like(dem)
    chisq = np.empty((F, H, W), dtype=np.float32)

    gpu_str = _gpu_device_summary()
    util0 = _gpu_utilization_once()
    util_str = f"gpu={util0['gpu']}% mem={util0['mem']}%" if util0 else "unknown"
    log.info("[singleGPU] starting compute: frames=%d crop=%dx%d tiles=%dx%d (GPU=%s, util=%s)",
             F, H, W, Th, Tw, gpu_str, util_str)

    with Profiler(client=None, benchdir=bench_dir, stamp=stamp, enable_perf_html=False) as prof:
        prof.section("compute", start=True)

        for fi, frame in enumerate(stack):
            t0 = time.perf_counter()
            try:
                # Single-frame GPU solver
                dem_f, edem_f, chisq_f, _ = solve_tile_all_single_gpu(frame, nmu=nmu)
            except Exception as e:
                log.error("[singleGPU] frame %d GPU failed: %s, falling back to CPU", fi, e)
                from src.common.solver import solve_tile_all as _solve_cpu
                dem_f, edem_f, chisq_f, _ = _solve_cpu(frame[None, :, :, :], nmu=nmu)

            dem[fi] = dem_f[0]
            edem[fi] = edem_f[0]
            chisq[fi] = chisq_f[0]

            dt = time.perf_counter() - t0
            tiles_per_frame = int(np.ceil(H / Th) * np.ceil(W / Tw))
            ms_per_tile = dt / tiles_per_frame * 1e3
            tiles_per_s = tiles_per_frame / dt if dt > 0 else float("nan")
            util = _gpu_utilization_once()
            if util:
                log.info("[singleGPU] frame %d done: time=%.3fs tiles/s=%.2f ms/tile=%.2f | gpu=%d%% mem=%d%%",
                         fi, dt, tiles_per_s, ms_per_tile, util["gpu"], util["mem"])
            else:
                log.info("[singleGPU] frame %d done: time=%.3fs tiles/s=%.2f ms/tile=%.2f",
                         fi, dt, tiles_per_s, ms_per_tile)

        prof.section("compute", start=False)

    # Save outputs
    outputs_path = bench_dir / f"outputs_{stamp}.npz"
    np.savez_compressed(outputs_path, dem=dem, edem=edem, chisq=chisq)

    # Verification (placeholder)
    verify_ok = bool(verify)
    reports: List[str] = []
    if verify and golden_root:
        reports.append("Verification placeholder — integrate golden check here.")
        verify_ok = True

    total_s = float(getattr(prof, "_sections", {}).get("compute", (0.0, 0.0))[1])
    tiles_total = int(np.ceil(H / Th) * np.ceil(W / Tw)) * F
    tiles_per_s = tiles_total / total_s if total_s > 0 else float("nan")
    dems_per_s = (F * H * W) / total_s if total_s > 0 else float("nan")

    bench = dict(
        stamp=stamp,
        mode="single-gpu",
        frames=F,
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
             F, int(np.ceil(H / Th) * np.ceil(W / Tw)), H, W, bench["total_seconds"], tiles_per_s, dems_per_s)
    print(f"[SingleGPU] frames={F} tiles/frame={int(np.ceil(H / Th) * np.ceil(W / Tw))} crop={H}x{W} "
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
