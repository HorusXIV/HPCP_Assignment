import numpy as np

def dem_to_temp_maps(demmap: np.ndarray, logT_bins: np.ndarray):
    dem = np.clip(demmap, 0, None).astype(np.float32, copy=False)  # (H,W,nt)
    EM = dem.sum(axis=2)
    valid = EM > 0
    imax = dem.argmax(axis=2).astype(np.intp)
    logT_bins = np.asarray(logT_bins).reshape(-1)
    peak = np.where(valid, np.take(logT_bins, imax), np.nan)
    T_centers = (10.0**logT_bins).astype(np.float32)
    num = np.einsum('ijk,k->ij', dem, T_centers, optimize=True)
    mean = np.full(EM.shape, np.nan, dtype=np.float32)
    np.divide(num, EM, out=mean, where=valid)
    mean = np.log10(mean, out=mean, where=valid)
    return mean, peak
