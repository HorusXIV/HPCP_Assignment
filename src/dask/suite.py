# src/dask/suite.py
# Dask pipeline for DEMREG on AIA tiles (channels-last tiles from dataio)
# Change summary:
# - Accept tiles shaped (Th, Tw, 6) from src.common.dataio
# - Run solve_tile ONCE per tile: (6, Th, Tw) -> (Th, Tw, nt), (Th, Tw, nt), (Th, Tw)
# - Pack outputs into a single array and slice to avoid 3× recompute & 3× graph size

from __future__ import annotations

import numpy as np
import dask.array as da

from .solver import solve_tile


def _solve_all_wrapper(block: np.ndarray, nmu: int, nt: int) -> np.ndarray:
    """
    Run the expensive DEMREG solve ONCE for a single tile and pack outputs.

    Parameters
    ----------
    block : np.ndarray
        Tile array with shape (Th, Tw, 6) [channels-last], float32 recommended.
    nmu : int
        Mu grid size / solver parameter.
    nt : int
        Temperature grid size (DEM bins).

    Returns
    -------
    out : np.ndarray
        Packed outputs with shape (Th, Tw, 2*nt + 1):
          out[..., 0:nt]     = dem
          out[..., nt:2*nt]  = edem
          out[..., -1]       = chisq
    """
    if block.ndim != 3 or block.shape[-1] != 6:
        raise ValueError(f"Expected tile (Th,Tw,6); got {block.shape}")

    dem, edem, chisq, _meta = solve_tile(block, nmu=nmu, nt=nt)  # dem: (Th,Tw,nt)

    Th, Tw, NT = dem.shape
    if NT != nt:
        raise ValueError(f"solve_tile returned nt={NT}, expected nt={nt}")

    out = np.empty((Th, Tw, 2 * nt + 1), dtype=np.float32)
    out[..., 0:nt] = dem.astype(np.float32, copy=False)
    out[..., nt:2 * nt] = edem.astype(np.float32, copy=False)
    out[..., -1] = chisq.astype(np.float32, copy=False)
    return out


def build_graph(
    frame_da: da.Array,
    *,
    nmu: int,
    nt: int,
) -> tuple[da.Array, da.Array, da.Array]:
    """
    Build the Dask graph that computes DEM, eDEM, and chi-squared ONCE per tile.

    Parameters
    ----------
    frame_da : dask.array.Array
        Single-frame array with shape (H, W, 6), chunked as (Th, Tw, 6).
        (Produced by src.common.dataio.build_lazy_npz_stack(...)[0])
    nmu : int
        Mu grid.
    nt : int
        Temperature bins.

    Returns
    -------
    dem_lazy, edem_lazy, chisq_lazy : dask.array.Array
        Lazy arrays shaped:
          dem   : (H, W, nt)
          edem  : (H, W, nt)
          chisq : (H, W)
    """
    if frame_da.ndim != 3 or frame_da.shape[-1] != 6:
        raise ValueError(f"Expected (H,W,6) frame; got {frame_da.shape}")

    H_chunks, W_chunks, C_chunks = frame_da.chunks
    if not (len(C_chunks) == 1 and C_chunks[0] == 6):
        raise ValueError(f"Last axis must be single chunk of 6 bands, got {C_chunks}")

    # Each input chunk is (Th, Tw, 6); output chunk per tile is (Th, Tw, 2*nt+1)
    packed_chunks = (H_chunks, W_chunks, (2 * nt + 1,))

    combined = da.map_blocks(
        _solve_all_wrapper,
        frame_da,
        nmu,
        nt,
        dtype=np.float32,
        chunks=packed_chunks,
    )  # (H, W, 2*nt+1)

    dem_lazy = combined[..., 0:nt]          # (H, W, nt)
    edem_lazy = combined[..., nt : 2 * nt]  # (H, W, nt)
    chisq_lazy = combined[..., -1]          # (H, W)

    return dem_lazy, edem_lazy, chisq_lazy
