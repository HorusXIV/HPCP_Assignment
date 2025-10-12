# src/common/solver_utils.py
from __future__ import annotations
import numpy as np

# (keep your existing helpers: validate_input, build_error_model, get_logt_bins, ...)
# If you already added get_bins / temps_edges_from_logt earlier, leave them as-is.

def get_logt_bins(nt: int = 50, logt_min: float = 5.5, logt_max: float = 7.5):
    """Return (nt, log10(T) centers)."""
    logt = np.linspace(logt_min, logt_max, nt, dtype=np.float32)
    return nt, logt

def logt_edges_from_centers(logt: np.ndarray) -> np.ndarray:
    """Return log10(T) edges (nt+1) from centers (nt)."""
    logt = np.asarray(logt, dtype=np.float64)
    nt = logt.size
    d = float(np.mean(np.diff(logt))) if nt > 1 else 0.1
    edges = np.empty(nt + 1, dtype=np.float64)
    edges[0] = logt[0] - 0.5 * d
    for i in range(nt):
        edges[i + 1] = edges[i] + d
    return edges.astype(np.float32)

def temps_edges_from_logt(logt: np.ndarray) -> np.ndarray:
    """From log10(T) centers (nt), return Kelvin edges (nt+1)."""
    return (10.0 ** logt_edges_from_centers(logt)).astype(np.float32)

def get_bins(nt: int = 50, logt_min: float = 5.5, logt_max: float = 7.5):
    """Convenience: (logt_centers (nt,), temps_edges_K (nt+1,))."""
    _, logt = get_logt_bins(nt=nt, logt_min=logt_min, logt_max=logt_max)
    temps = temps_edges_from_logt(logt)
    return logt, temps

# ---------------- Synthetic responses ----------------

def synthesize_tresp(
    logt_centers: np.ndarray,
    nf: int = 6,
    *,
    model: str = "gaussian",
    width: float = 0.20,
    normalize: str = "l1",
    include_jacobian: bool = True, # <- include ΔlogT by default
) -> np.ndarray:
    """
    Build a synthetic temperature response matrix R of shape (nt, nf)
    aligned with the provided log10(T) centers.

    - model='gaussian': nf broad Gaussians across the logT range
    - width: sigma in log10(T) units
    - normalize:
        'l1'     -> each column integrates to 1 over logT with ΔlogT
        'colmax' -> each column max = 1 (less robust for DEM scale)
        'none'   -> no normalization (rarely useful)
    - include_jacobian: multiply by ΔlogT per bin before normalization
    """
    logt = np.asarray(logt_centers, dtype=float)
    nt = logt.size
    lo, hi = float(logt.min()), float(logt.max())

    # gaussian beams centered across interior
    mu = np.linspace(lo + 0.1*(hi-lo), hi - 0.1*(hi-lo), nf)
    sigma = float(width)

    R = np.empty((nt, nf), dtype=float)
    if model == "gaussian":
        for j in range(nf):
            R[:, j] = np.exp(-0.5 * ((logt - mu[j]) / sigma)**2)
    else:
        raise ValueError(f"Unknown synthetic response model: {model}")

    # ΔlogT as bin widths (length nt)
    edges = logt_edges_from_centers(logt)        # (nt+1,)
    dlogt = np.diff(edges)                       # (nt,)

    if include_jacobian:
        R *= dlogt[:, None]                      # integrate over logT

    if normalize == "l1":
        col_sums = R.sum(axis=0, keepdims=True)  # integral per channel
        R /= (col_sums + 1e-12)
    elif normalize == "colmax":
        R /= (R.max(axis=0, keepdims=True) + 1e-12)
    elif normalize == "none":
        pass
    else:
        raise ValueError(f"Unknown normalize='{normalize}'")

    return R.astype(np.float32, copy=False)

def get_synthetic_calibration(
    nt: int = 50,
    nf: int = 6,
    *,
    logt_min: float = 5.5,
    logt_max: float = 7.5,
    **kwargs,
):
    """
    Convenience: return (tresp (nt,nf), tresp_logt (nt,), temps_edges_K (nt+1,))
    using the synthetic generator above.
    kwargs are passed to synthesize_tresp (e.g., width=0.15, include_jacobian=False).
    """
    logt, temps = get_bins(nt=nt, logt_min=logt_min, logt_max=logt_max)
    tresp = synthesize_tresp(logt, nf=nf, **kwargs)
    return tresp, logt.astype(np.float32, copy=False), temps.astype(np.float32, copy=False)
