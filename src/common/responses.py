import numpy as np

def prepare_synthetic_responses(logT_min=5.5, logT_max=7.5, n_tresp=200, nt=24, nf=6):
    logT = np.linspace(logT_min, logT_max, n_tresp)      # T_RESP_LOGT
    centers = np.linspace(logT_min+0.2, logT_max-0.2, nf)
    width = 0.15
    T_RESP = np.exp(-0.5*((logT[:,None]-centers[None,:])/width)**2) + 1e-30
    TEMPS = np.logspace(logT_min, logT_max, nt+1)        # Kelvin edges
    return T_RESP, logT, TEMPS


# Todo falls jemals gewollt: Echte Responses