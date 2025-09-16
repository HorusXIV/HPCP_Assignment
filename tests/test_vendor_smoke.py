import numpy as np
from src.baseline.vendor.dn2dem_pos import dn2dem_pos


def test_vendor_smoke():
    # tiny synthetic inputs
    H = W = 8; nf = 6; n_tresp = 50; nt = 16
    rng = np.random.default_rng(0)
    f = rng.random((H, W, nf)).astype(np.float32)
    e = np.sqrt(f) + 1e-6
    logT = np.linspace(5.5, 7.5, n_tresp)
    T_RESP = np.exp(-0.5*((logT[:,None]-np.linspace(5.7,7.3,nf)[None,:])/0.2)**2) + 1e-30
    TEMPS = np.logspace(5.5, 7.5, nt+1)

    demmap, edemmap, logT_bins, chisq, dn_reg = dn2dem_pos(f, e, T_RESP, logT, TEMPS, nmu=42)
    assert demmap.shape[:2] == (H, W)
    assert demmap.shape[2] == len(logT_bins)
