import numpy as np
import pandas as pd
import warnings
from .utils import ensure_lexsorted

def get_hist(weight,data,bins): return np.histogram(data,bins=bins,weights=weight)[0]

def calc_matrices(var_arr,cv):
    def calc_matrices(var_arr, cv):
        """
        Calculate covariance, fractional covariance, and correlation matrices.
        This function computes statistical matrices from variations around a central value,
        commonly used in systematic uncertainty analysis.
        Parameters
        ----------
        var_arr : np.ndarray
            2D array of shape (nbins, nuniv) containing variations for each bin and universe.
            nbins is the number of bins, nuniv is the number of universes/variations.
        cv : np.ndarray
            1D array of shape (nbins,) containing central values for each bin.
        Returns
        -------
        cov : np.ndarray
            2D array of shape (nbins, nbins) containing the covariance matrix.
        cov_frac : np.ndarray
            2D array of shape (nbins, nbins) containing the fractional covariance matrix
            (normalized by central values).
        corr : np.ndarray
            2D array of shape (nbins, nbins) containing the correlation matrix,
            derived from the covariance matrix.
        Notes
        -----
        - Warnings about invalid values in division are suppressed (e.g., division by zero).
        - The matrices returned are normalized to n_universes. 
        - The correlation matrix is computed as: corr[i,j] = cov[i,j] / sqrt(cov[i,i] * cov[j,j])
        """
    
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
    multisim_col = []
    univ_level = -1

    # find the level that the multisim universes begin 
    for col in df.columns:
        if "univ" in "".join(list(col)):
            for i, x in enumerate(col): 
                if x.startswith('univ'): 
                    univ_level = i 
                    break
            break
    
    for col in df.columns:
        if 'morph' in col:
            unisim_col.append(col)
        elif 'ps1' in col:
            multisig_col.append(col)
        elif "univ" in "".join(list(col)):
            if col[:univ_level] not in multisim_col: 
                multisim_col.append(col[:univ_level])
            
    syst_dict = {}
    nbins = len(bins) 

    # Optimized: stack weights and do histogram computation per syst type
    # For unisim: process all unisim columns together
    for col in unisim_col: 
        # for unisim, get straight from `morph`
        weights = df[col].to_numpy()
        n_universes = weights.shape[1] if weights.ndim > 1 else 1
        hists = np.zeros((nbins-1, n_universes))
        
        # Compute histograms for all universes in this column
        if weights.ndim > 1:
            for i in range(n_universes):
                hists[:, i] = get_hist(weights[:, i], cv_input, bins)
        else:
            hists[:, 0] = get_hist(weights, cv_input, bins)
        
        syst_dict[col[2]] = [hists]
    
    # For multisigma: process all multisig columns together
    for col in multisig_col:
        # for multisigma, get two universes, ps1 and ms1
        ps1_col = col 
        ms1_col = tuple([x if x!='ps1' else 'ms1' for x in list(col)])
        ps1_weights = df[ps1_col].to_numpy()
        ms1_weights = df[ms1_col].to_numpy()
        
        hists = np.zeros((nbins-1, 2))
        hists[:, 0] = get_hist(ps1_weights, cv_input, bins)
        hists[:, 1] = get_hist(ms1_weights, cv_input, bins)
        
        syst_dict[col[2]] = [hists]
    
    # For multisim: process all multisim columns together
    for col in multisim_col:
        # for multisim, get all universes automatically
        weights = df[col].to_numpy()
        n_universes = weights.shape[1]
        hists = np.zeros((nbins-1, n_universes))
        
        # Compute histograms for all universes in this column
        for i in range(n_universes):
            hists[:, i] = get_hist(weights[:, i], cv_input, bins)
        
        syst_dict[col[2]] = [hists]
        
    for key in syst_dict.keys():
        hists = syst_dict[key][0]
        cov, frac, corr = calc_matrices(hists,cv_hist)
        syst_dict[key] += [cov, frac, corr]
    return syst_dict