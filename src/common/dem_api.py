import numpy as np
from src.baseline.vendor.dn2dem_pos import dn2dem_pos as _dn2dem_pos

def dn2dem(frame_6hw: np.ndarray, T_RESP, T_RESP_LOGT, TEMPS, nmu=42):
    """Thin wrapper around the provided solver.
       frame_6hw: (6,H,W) -> returns (demmap, edemmap, logT_bins, chisq, dn_reg)
    """
    f = np.moveaxis(frame_6hw, 0, -1).astype(np.float32, copy=False)  # (H,W,6)
    f = np.clip(np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0), 0, None)
    edn = np.sqrt(f, dtype=np.float32) + 1e-6
    return _dn2dem_pos(f, edn, T_RESP, T_RESP_LOGT, TEMPS, nmu=nmu)