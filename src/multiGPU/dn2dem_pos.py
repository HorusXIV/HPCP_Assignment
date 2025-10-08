"""GPU-aware wrapper mirroring the vendor's ``dn2dem_pos`` behavior.

High-level flow:
- Prepare the temperature grid in log10 space (``logT``), the per-bin widths
    (``dlogT``), and the instrument response matrix ``R(T)``.
- Convert between DEM (differential emission measure; per unit log-temperature)
    and EMD (emission measure in bins) consistently by applying or removing the
    bin-width factor when required.
- Normalize inputs to 2D, dispatch a batched CuPy-based solver, and reshape the
    results back to the original spatial layout.

Notes
-----
- A CUDA-capable environment is required for GPU acceleration. If no CUDA
    device is visible at runtime, the underlying kernel raises a ``RuntimeError``.
- Behavior mirrors ``src/baseline/vendor/dn2dem_pos.py`` to facilitate
    cross-validation and reproducibility.
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from .kernels import demmap_pos  # GPU kernel
from src.common.nvtx import nvtx_range


def _prepare_rmatrix_and_axes(
    tresp: np.ndarray,
    tresp_logt: np.ndarray,
    temps: np.ndarray,
    emd_int: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Construct response matrix and temperature axes consistent with vendor.

    The vendor routine expects instrument responses as a function of
    temperature sampled at specific log10(T) locations. We interpolate each
    filter's response to the target ``logt`` grid and, when solving for DEM
    (rather than EMD), include the bin-width factor from the change of
    variables between T and log10(T).

    Args:
        tresp: Array of shape ``(n_resp_samples, nf)`` with per-filter
            temperature responses R(T). Values must be positive; nonpositive
            entries are replaced by the smallest positive value in that filter
            to avoid taking log of nonpositive numbers during interpolation.
        tresp_logt: Monotonic array of shape ``(n_resp_samples,)`` giving the
            log10(T) coordinate of each row in ``tresp``.
        temps: Monotonic array of temperature bin edges in linear T (Kelvin).
            A regular grid in log10(T) is computed from these edges.
        emd_int: If True, interpret the unknown as EMD (per-bin EM); if False,
            interpret it as DEM (per unit logT). DEM requires multiplying
            responses by the bin factor below.

    Returns:
        Tuple of ``(rmatrix, logt, dlogt, sclf)`` where:
        - rmatrix: Interpolated response matrix of shape ``(nt, nf)``;
          multiplied by a numeric scale factor ``sclf`` for stability and by
          a bin factor when solving for DEM.
        - logt: Log10(T) bin centers of shape ``(nt,)``.
        - dlogt: Per-bin widths in log10(T) of shape ``(nt,)``.
        - sclf: Scalar numeric scaling (``1e15``) applied to ``rmatrix`` to
          keep values in a numerically comfortable range.
    """
    temps = np.asarray(temps)
    # Compute logt bin centers and widths in log10 space
    dlogt = np.log10(temps[1:]) - np.log10(temps[:-1])
    nt = int(dlogt.shape[0])
    logt = np.array(
        [np.log10(temps[0]) + dlogt[i] * (float(i) + 0.5) for i in range(nt)],
        dtype=float,
    )

    tresp = np.asarray(tresp, dtype=float)
    tresp_logt = np.asarray(tresp_logt, dtype=float)
    nf = int(tresp.shape[1])

    # Clean responses: replace nonpositive entries with the per-filter
    # minimum of positive entries to keep log10 well-defined
    truse = np.zeros_like(tresp, dtype=float)
    for i in range(nf):
        tcol = tresp[:, i]
        pos = tcol > 0
        if not np.any(pos):
            # Degenerate column; keep zeros to avoid div-by-zero and rely on solver guards
            truse[:, i] = 0.0
        else:
            truse[pos, i] = tcol[pos]
            truse[~pos, i] = float(np.min(tcol[pos]))

    # Interpolate the logarithm of the response in log10 space (smooths
    # multiplicative variations), then exponentiate to recover linear values
    tr = np.zeros((nt, nf), dtype=float)
    for i in range(nf):
        # interp in log10 of response, then convert back
        tr[:, i] = 10.0 ** np.interp(
            logt, tresp_logt, np.log10(np.maximum(truse[:, i], 1e-300))
        )

    # Prepare response matrix for DEM vs EMD
    rmatrix = np.zeros((nt, nf), dtype=float)
    if emd_int:
        for i in range(nf):
            rmatrix[:, i] = tr[:, i]
    else:
        # DEM requires a bin factor due to d(T) -> d(log10 T) change of
        # variables. In linear T, dT = (T * ln 10) dlog10 T, hence the factor
        dlogTfac = (10.0**logt) * np.log(10.0**dlogt)
        for i in range(nf):
            rmatrix[:, i] = tr[:, i] * dlogTfac

    # Numeric scale-up to keep values in comfortable ranges; undone on return
    sclf = 1.0e15
    rmatrix *= sclf
    return rmatrix, logt, dlogt, sclf


def _shape_to_2d(arr: np.ndarray) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """Flatten an array to 2D ``(n_samples, nf)`` and return original shape.

    Args:
        arr: Input array whose last axis is interpreted as filters ``nf``;
            leading axes (if any) are flattened into ``n_samples``.

    Returns:
        Tuple ``(flat, orig_shape)`` where ``flat`` has shape
        ``(n_samples, nf)`` and ``orig_shape`` is the input shape for later
        restoration.
    """
    a = np.asarray(arr)
    if a.ndim == 1:
        return a.reshape(1, -1), a.shape
    if a.ndim == 2:
        return a.reshape(-1, a.shape[-1]), a.shape
    # assume last axis is nf
    return a.reshape(-1, a.shape[-1]), a.shape


def _reshape_from_2d(
    arr2d: np.ndarray, orig_shape: Tuple[int, ...], nt: int
) -> np.ndarray:
    """Restore a ``(n_samples, nt)`` array to the original leading layout.

    Args:
        arr2d: 2D array with shape ``(n_samples, nt)``.
        orig_shape: Original shape of the input data prior to flattening.
        nt: Number of temperature bins in the output per sample.

    Returns:
        Array with shape ``(*orig_shape[:-1], nt)`` where the spatial layout
        matches the original input and the last axis is ``nt``.
    """
    if len(orig_shape) == 1:
        # nf -> nt (keep 1D for truly vector inputs)
        return arr2d.reshape(nt)
    if len(orig_shape) == 2:
        nx, _nf = orig_shape
        # For 2D inputs, keep 2D outputs (nx, nt) so MPI gather and callers
        # expecting (n_samples, nt) continue to work.
        return arr2d.reshape(nx, nt)
    # (..., nf) -> (..., nt) without squeezing singleton spatial dims
    return arr2d.reshape(*orig_shape[:-1], nt)


def dn2dem_pos(
    dn_in: np.ndarray,
    edn_in: np.ndarray,
    tresp: Optional[np.ndarray] = None,
    tresp_logt: Optional[np.ndarray] = None,
    temps: Optional[np.ndarray] = None,
    reg_tweak: float = 1.0,
    max_iter: int = 10,
    gloci: int = 0,
    rgt_fact: float = 1.5,
    dem_norm0: Optional[np.ndarray] = None,
    nmu: int = 40,
    warn: bool = False,
    emd_int: bool = False,
    emd_ret: bool = False,
    l_emd: bool = False,
    non_pos: bool = False,
    rscl: bool = False,
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    """Reconstruct DEM/EMD for many samples on GPU with vendor-compatible I/O.

    This wrapper normalizes shapes, prepares instrument responses and
    temperature axes, and calls the CuPy-accelerated kernel
    ``kernels.demmap_pos``. Outputs are reshaped back to match the input's
    spatial layout.

    Args:
        dn_in: Data numbers (counts) of shape ``(..., nf)`` where the last
            axis enumerates filters/bands. Leading axes are flattened into
            samples.
        edn_in: 1-sigma uncertainties matching ``dn_in`` in shape or with a
            single row ``(1, nf)`` that will broadcast to all samples.
        tresp: Optional instrument responses ``(n_resp_samples, nf)``. If
            omitted, synthetic responses are constructed once using
            ``src.common.responses.build_binned_responses`` for deterministic
            behavior in tests and examples.
        tresp_logt: Optional log10(T) coordinates for ``tresp``
            ``(n_resp_samples,)``. Required when ``tresp`` is provided.
        temps: Optional temperature bin edges in linear T (Kelvin). If not
            provided, a default grid compatible with synthetic responses is
            used.
        reg_tweak: Discrepancy principle multiplier for the residual target.
        max_iter: Maximum iterations for non-negativity relaxation during
            regularization selection.
        gloci: If nonzero, use provided per-filter constraints mask (all ones
            when ``1``) to seed the initial pass (vendor parity).
        rgt_fact: Multiplicative increase of regularization when negative DEM
            entries appear.
        dem_norm0: Optional initial normalization per sample/band
            ``(..., nt)``; broadcastable along samples.
        nmu: Number of candidate regularization values per sample (log grid).
        warn: Placeholder for vendor API compatibility (unused).
        emd_int: If True, treat the unknown as EMD on input (no bin factor
            applied to responses). If False (default), solve for DEM.
        emd_ret: If True, return EMD; otherwise return DEM. Conversion is
            applied after solving as needed.
        l_emd: If True, use L1-like weighting; else the vendor-like
            sqrt(weight)/sqrt(dlogT) scheme.
        non_pos: If True, disable positivity enforcement (single iteration).
        rscl: If True, rescale the DEM by the mean observed/predicted ratio.

    Returns:
        Tuple ``(dem, edem, elogt, chisq, dn_reg)``:
        - dem: Array with shape ``(*dn_in.shape[:-1], nt)``; DEM (default) or
          EMD (when ``emd_ret=True``) after any requested conversion. Units
          follow the response normalization.
        - edem: Same shape as ``dem``; propagated uncertainties.
        - elogt: Same shape as ``dem``; effective half-width in log10(T) for
          each bin (converted from an FWHM-like estimate to 1-sigma).
        - chisq: Reduced chi-square per sample with shape ``(*dn_in.shape[:-1],)``.
        - dn_reg: Predicted counts per filter with shape ``(*dn_in.shape,)``.

    Raises:
        RuntimeError: If no CUDA device is available or GPU memory is
            insufficient even after automatic downshifts in batch size.
        ImportError: If CuPy is unavailable in the execution environment.
    """
    with nvtx_range("DN2DEM_PREP", color=0x4CAF50):
        # Normalize input shapes to 2D (n_samples, nf)
        dn2d, dn_shape = _shape_to_2d(np.asarray(dn_in))
        ed2d, _ = _shape_to_2d(np.asarray(edn_in))
        ns, nf = dn2d.shape

        # Prepare axis and response matrix exactly once
        if tresp is None or tresp_logt is None or temps is None:
            # Synthetic path: build responses using common.responses to match
            # internal structure and binning behavior.
            from src.common.responses import build_binned_responses

            # Keep prior behavior from multiGPU.main: nt default to 10
            nt = 10
            rmatrix_b, logt_b, dlogt_b = build_binned_responses(nt=nt, nf=nf)
            # Apply vendor-like numeric scaling for stable magnitudes
            sclf = 1.0e15
            rmatrix = (rmatrix_b * sclf).astype(float, copy=False)
            logt = logt_b.astype(float, copy=False)
            dlogt = dlogt_b.astype(float, copy=False)
            # For DEM vs EMD parity, if emd_int is False, multiply by dlogTfac
            if not emd_int:
                dlogTfac = (10.0**logt) * np.log(10.0**dlogt)
                rmatrix = rmatrix * dlogTfac[:, None]
        else:
            # Vendor instrument path
            rmatrix, logt, dlogt, sclf = _prepare_rmatrix_and_axes(
                tresp, tresp_logt, temps, emd_int
            )

        nt = int(logt.shape[0])

        # Default dem_norm0 to ones if not provided; mirror vendor logic
        if dem_norm0 is None:
            dem_norm0 = np.ones((*dn_shape[:-1], nt), dtype=float)
        # Choose nmu similar to vendor defaults
        if len(dn_shape) == 1:
            # 1D case
            if nmu <= 40:
                nmu = 500
        else:
            if nmu <= 40:
                nmu = 42

        # Broadcast ed rows if needed to match dn samples
        if ed2d.shape[0] == 1 and ns > 1:
            ed2d = np.repeat(ed2d, ns, axis=0)

        # Positivity override
        if non_pos:
            max_iter = 1
            warn = False

        # glc selection per vendor
        if gloci == 1:
            glc = np.ones((nf,), dtype=int)
        else:
            glc = np.zeros((nf,), dtype=int)

        dn_local = dn2d
        ed_local = ed2d
        dem0_local = dem_norm0.reshape(-1, nt) if dem_norm0 is not None else None

    # Run GPU solver locally per rank
    with nvtx_range("DN2DEM_SOLVE", color=0xE65100):
        dem_local, edem_local, elogt_local, chisq_local, dn_reg_local = demmap_pos(
            dn_local,
            ed_local,
            rmatrix,
            logt,
            dlogt,
            glc,
            reg_tweak=reg_tweak,
            max_iter=max_iter,
            rgt_fact=rgt_fact,
            dem_norm0=dem0_local if dem0_local is not None else None,
            nmu=int(nmu),
            warn=warn,
            l_emd=l_emd,
            rscl=rscl,
        )

    # Local return
    dem2d, edem2d, elogt2d, chisq1d, dnreg2d = (
        dem_local,
        edem_local,
        elogt_local,
        chisq_local,
        dn_reg_local,
    )

    # Reshape back and numeric scaling fixes
    with nvtx_range("DN2DEM_POST", color=0x795548):
        dem = _reshape_from_2d(dem2d, dn_shape, nt) * sclf
        edem = _reshape_from_2d(edem2d, dn_shape, nt) * sclf
        # Convert an FWHM-like support width to one-sigma for easier
        # interpretation under a Gaussian approximation: 2*sqrt(2*ln 2)
        fwhm_to_sigma = 2.0 * np.sqrt(2.0 * np.log(2.0))
        elogt = _reshape_from_2d(elogt2d, dn_shape, nt) / fwhm_to_sigma
        # Preserve spatial dimensions; do not squeeze singleton axes
        chisq = chisq1d.reshape(*dn_shape[:-1])
        dn_reg = dnreg2d.reshape(*dn_shape)

        # EMD/DEM conversion at return as per vendor
        if emd_int and emd_ret:
            return dem, edem, elogt, chisq, dn_reg
        if emd_int and not emd_ret:
            # Convert EMD back to DEM by dividing by bin width in linear T
            # Using the same factor as in vendor wrapper (dlogTfac)
            dlogTfac = (10.0**logt) * np.log(10.0**dlogt)
            return dem / dlogTfac, edem / dlogTfac, elogt, chisq, dn_reg
        if not emd_int and emd_ret:
            dlogTfac = (10.0**logt) * np.log(10.0**dlogt)
            return dem * dlogTfac, edem * dlogTfac, elogt, chisq, dn_reg
        return dem, edem, elogt, chisq, dn_reg


__all__ = ["dn2dem_pos"]
