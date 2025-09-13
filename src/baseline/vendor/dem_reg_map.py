import numpy as np

def dem_reg_map(sigmaa,sigmab,U,W,data,err,reg_tweak,nmu=500):
    """
    dem_reg_map
    computes the regularization parameter
    
    Inputs

    sigmaa: 
        gsv vector
    sigmab: 
        gsv vector
    U:      
        gsvd matrix
    V:      
        gsvd matrix
    data:   
        dn data
    err:    
        dn error
    reg_tweak: 
        how much to adjust the chisq each iteration

    Outputs

    opt:
        regularization paramater

    """

    nf = data.shape[0]
    nreg = sigmaa.shape[0]
    arg = np.zeros((nreg, nmu))
    discr = np.zeros((nmu,))

    # Safe ratio of generalized singular values
    eps = np.finfo(float).tiny
    sigs = sigmaa[:nf] / np.maximum(sigmab[:nf], eps)
    sigs = sigs[np.isfinite(sigs) & (sigs > 0)]

    # Fallback range if everything got filtered out
    if sigs.size == 0:
        minx, maxx = 1e-8, 1e2
    else:
        maxx = float(np.max(sigs))
        minx = float((np.min(sigs) ** 2) * 1e-4)  # keep author’s intent
        # ensure strictly positive and separated
        minx = max(minx, 1e-300)
        if not (maxx > minx):
            maxx = minx * 10.0

    # Ensure at least 2 samples
    nmu_eff = int(max(nmu, 2))
    # Log-spaced sampling without manual logs (avoids inf/nan)
    mu = np.geomspace(minx, maxx, num=nmu_eff, dtype=float)

    for kk in range(nf):
        coef = data @ U[kk, :]
        # note: use nmu_eff if you kept arg/discr sized by nmu_eff
        for ii in range(nmu_eff):
            num = mu[ii] * (sigmab[kk] ** 2) * coef
            den = (sigmaa[kk] ** 2 + mu[ii] * (sigmab[kk] ** 2))
            arg[kk, ii] = (num / den) ** 2

    discr[:nmu_eff] = np.sum(arg[:, :nmu_eff], axis=0) - np.sum(err ** 2) * reg_tweak
    opt = mu[int(np.argmin(np.abs(discr[:nmu_eff])))]
    return opt