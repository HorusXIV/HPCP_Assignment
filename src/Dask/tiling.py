from __future__ import annotations

import numpy as np
import dask.array as da

from src.common.dem_api import dn2dem

# Per-worker constant cache. We fill this via client.run() in runner.py
_CONST = None  # tuple: (T_RESP, T_RESP_LOGT, TEMPS, nmu)


def _blk(block_6hw: np.ndarray, nt: int) -> np.ndarray:
    """
    Map a single tile (6, h, w) -> (h, w, nt).
    Reads heavy constants from the per-worker cache to avoid graph bloat.
    """
    global _CONST
    if _CONST is None:
        raise RuntimeError(
            "Per-worker constants not set. "
            "runner.py should call client.run(_set_constants, ...) before computing."
        )

    T_RESP, T_RESP_LOGT, TEMPS, nmu = _CONST

    # Ensure clean, compact memory for vendor code
    frame_6hw = np.ascontiguousarray(block_6hw, dtype=np.float32)

    # dn2dem returns: demmap, edemmap, elogt, chisq, dn_reg
    demmap, _, _, _, _ = dn2dem(frame_6hw, T_RESP, T_RESP_LOGT, TEMPS, nmu=nmu)

    # demmap expected shape: (h, w, nt); ensure dtype is float32 to reduce memory
    return np.asarray(demmap, dtype=np.float32)


def dem_map_blocks(
    frame_6hw: np.ndarray | da.Array,
    nt: int,
    *,
    tile_h: int = 256,
    tile_w: int = 256,
) -> da.Array:
    """
    Tile a (6, H, W) frame and run dn2dem per tile using dask.map_blocks.
    Output is (H, W, nt). Heavy constants are read from per-worker cache.
    """
    if isinstance(frame_6hw, np.ndarray):
        darr = da.from_array(frame_6hw, chunks=(6, tile_h, tile_w), asarray=False)
    else:
        # Ensure chunks along H, W; channel chunk should be 6
        ch0, hch, wch = frame_6hw.chunks
        if ch0 != (6,):
            frame_6hw = frame_6hw.rechunk({0: 6})
        darr = frame_6hw.rechunk({1: tile_h, 2: tile_w})

    # Map each tile (6, h, w) -> (h, w, nt)
    out = da.map_blocks(
        _blk,
        darr,
        nt=nt,
        dtype=np.float32,
        chunks=(tile_h, tile_w, nt),
    )

    return out


def _set_constants(T_RESP, T_RESP_LOGT, TEMPS, nmu: int) -> None:
    """
    Called on each worker via client.run(...) to cache heavy constants.
    """
    global _CONST
    _CONST = (T_RESP, T_RESP_LOGT, TEMPS, nmu)
