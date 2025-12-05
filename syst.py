import numpy as np
import pandas as pd
import warnings
from .utils import ensure_lexsorted

def get_hist(weight,data,bins): return np.histogram(data,bins=bins,weights=weight)[0]

def calc_matrices(var_arr,cv):
    nbins = len(cv)
    nuniv = len(var_arr[1])

    cov = np.zeros((nbins,nbins))
    cov_frac = np.zeros((nbins,nbins))
    corr = np.zeros((nbins,nbins))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",message="invalid value encountered in scalar divide")
        for ivar in range(nuniv):
            var = var_arr[:,ivar]
            for i in range(nbins):
                for j in range(nbins):   
                    cov[i,j]      += (var[i] - cv[i])*(var[j] - cv[j])
                    cov_frac[i,j] += (var[i] - cv[i])/(cv[i]) * (var[j] - cv[j])/(cv[j])
        
        cov /= nuniv
        cov_frac /= nuniv

        for i in range(nbins):
            for j in range(nbins):
                corr[i,j] = cov[i,j] / (np.sqrt ( cov[i,i] )* np.sqrt( cov[j,j] ))
    return cov, cov_frac, corr

def get_syst(indf: pd.DataFrame,
             var: str | tuple,
             bins: np.ndarray):
    df = indf.copy()
    
    if isinstance(df, pd.DataFrame):
        df = ensure_lexsorted(df, axis=0)
        df = ensure_lexsorted(df, axis=1)
    
    cv_input = df[var]
    cv_hist = np.histogram(cv_input,bins)[0]

    unisim_col = []
    multisig_col  = [] 
    genie_col = []
    flux_col = []
    for col in df.columns:
        col_list = list(col) # since col is tuple 
        if 'morph' in col:
            unisim_col.append(col_list[0])
        elif 'ps1' in col:
            multisig_col.append(col_list[0])
        elif "GENIE" in col:
            genie_col.append(col)
        elif "Flux" in col:
            flux_col.append(col)
    multisim_col = [genie_col, flux_col]
            
    syst_dict = {}
    nbins = len(bins) 
    for col in unisim_col: 
        # * for unisim, get straight from `morph`
        weights = df[col].morph.to_numpy()
        hists = np.apply_along_axis(get_hist, 0, weights, cv_input, bins)
        syst_dict[col] = [np.reshape(hists,(nbins-1,-1))]
    for col in multisig_col:
        # * for multisigma, get two universes, ps1 and ms1   
        weights = np.stack([df[col].ps1.to_numpy(),df[col].ms1.to_numpy()]).T
        hists = np.apply_along_axis(get_hist, 0, weights, cv_input, bins)
        syst_dict[col] = [hists]
    for i, cols in enumerate(multisim_col):
        # * for multisim, get all universes automatically
        weights = df[cols].to_numpy()
        hists = np.apply_along_axis(get_hist, 0, weights, cv_input, bins)
        if i==0: syst_dict['GENIE'] = [hists]
        if i==1: syst_dict["Flux"] = [hists]
        
    for key in syst_dict.keys():
        hists = syst_dict[key][0]
        cov, frac, corr = calc_matrices(hists,cv_hist)
        syst_dict[key] += [cov, frac, corr]
    return syst_dict