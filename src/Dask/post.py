# src/Dask/post.py
from __future__ import annotations
import dask.array as da
import numpy as np
from src.common.post import dem_to_temp_maps

def dem_to_temp_maps_blocks(dem_dask: da.Array, logT_bins: np.ndarray):
    """
    dem_dask: (N, H, W, nt) -> returns two Dask arrays:
      mean_logT (N,H,W), peak_logT (N,H,W)
    """
    nt = dem_dask.shape[-1]
    logT_bins = np.asarray(logT_bins)
    def _blk(dem_np: np.ndarray, bins: np.ndarray):
        # dem_np: (1,H,W,nt) -> (H,W) x2
        mean, peak = dem_to_temp_maps(dem_np[0], bins)
        return mean[None, ...], peak[None, ...]  # re-add N=1

    mean = da.map_blocks(
        lambda x, b: _blk(x, b)[0], dem_dask, logT_bins,
        dtype=np.float32, chunks=(1, dem_dask.chunks[1][0], dem_dask.chunks[2][0])
    )
    peak = da.map_blocks(
        lambda x, b: _blk(x, b)[1], dem_dask, logT_bins,
        dtype=np.float32, chunks=(1, dem_dask.chunks[1][0], dem_dask.chunks[2][0])
    )
    return mean, peak
