"""
Systematic and statistical uncertainty utilities.

Conventions
-----------
- All output histograms and covariance matrices are **flux-normalized by default**.
  Functions that support disabling this accept a `scale=True` parameter.
- Covariance matrices are normalized by N_universes.
- NaN weights (e.g., GENIE weights for true cosmics) are replaced with 1.0.
- The `normalize` flag refers to **area normalization** (for considering shape-only),
  which preserves the total CV counts but rescales each universe to match.
"""

import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
from .utils import ensure_lexsorted
from .selection import select

def get_hist(weight,data,bins): return np.histogram(data,bins=bins,weights=weight)[0]

def calc_matrices(var_arr, cv):
    """
    Calculate covariance, fractional covariance, and correlation matrices.
    This function computes three related matrices from a set of varied values
    and their central values: the covariance matrix, the fractional covariance
    matrix (normalized by central values), and the correlation matrix.
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
    - Uses vectorized/matrix operations for more efficient computation.
    - Division by zero warnings are suppressed during computation.
    """
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",message="invalid value encountered in divide")
        diffs = var_arr - cv[:, np.newaxis]
        diffs_norm = diffs / cv[:, np.newaxis]
        cov = (diffs @ diffs.T) / diffs.shape[1]
        cov_frac = (diffs_norm @ diffs_norm.T) / diffs_norm.shape[1]
        corr = cov / np.sqrt(np.outer(np.diag(cov),np.diag(cov)))
    return cov, cov_frac, corr

def calc_matrices_explicit(var_arr,cv):
    """Explicit (slow) implementation of calc_matrices, for validation only."""
    
    nbins = len(cv)
    nuniv = len(var_arr[1])
    print("Calculating matrices with ", nuniv, " universes and ", nbins, " bins.")

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
                    cov_frac[i,j] += (var[i] - cv[i])*(var[j] - cv[j]) / (cv[i]*cv[j])
        
        cov /= nuniv
        cov_frac /= nuniv

        for i in range(nbins):
            for j in range(nbins):
                corr[i,j] = cov[i,j] / (np.sqrt ( cov[i,i] )* np.sqrt( cov[j,j] ))
    return cov, cov_frac, corr

def get_syst(indf: pd.DataFrame,
             var: str | tuple,
             bins: np.ndarray,
             scale: bool = True,
             normalize: bool = False) -> dict:
    """Compute systematic uncertainty histograms and covariance matrices.

    Parameters
    ----------
    indf : pd.DataFrame
        Input DataFrame with MultiIndex columns containing systematic weights.
    var : str or tuple
        Column name for the variable to histogram.
    bins : np.ndarray
        Bin edges for histogramming.
    scale : bool, optional
        If True (default), apply flux-POT normalization from `flux_pot_norm` column.
    normalize : bool, optional
        If True, area-normalize each universe histogram to match the CV total counts
        (shape-only uncertainty). Default is False.

    Returns
    -------
    dict
        Dictionary keyed by systematic name, with values containing:
        - 'hists'    : np.ndarray of shape (nbins, nuniv), histograms per universe
        - 'cov'      : np.ndarray of shape (nbins, nbins), covariance matrix
        - 'cov_frac' : np.ndarray of shape (nbins, nbins), fractional covariance
        - 'corr'     : np.ndarray of shape (nbins, nbins), correlation matrix

    Notes
    -----
    - NaN weights are replaced with 1.0 (assumes the systematic doesn't apply to that event; e.g. GENIE on true cosmics).
    - Unisim knobs produce a single universe.
    - Multisigma knobs produce 2 universes (ps1, ms1).
    - Multisim knobs produce N universes.
    """
    df = indf.copy()
    
    if isinstance(df, pd.DataFrame):
        df = ensure_lexsorted(df, axis=0)
        df = ensure_lexsorted(df, axis=1)

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

    scaling = np.ones(indf.shape[0])
    for col in df.columns:
        if ("flux_pot_norm" in col) and scale: 
            scaling = df[col].to_numpy()
        if 'morph' in col:
            unisim_col.append(col)
        elif 'ps1' in col:
            multisig_col.append(col)
        elif "univ" in "".join(list(col)):
            if col[:univ_level] not in multisim_col: 
                multisim_col.append(col[:univ_level])
    if np.array_equal(scaling,np.ones(indf.shape[0])) and scale:
        print("No flux-averaged POT normalization found; flux normalization will be equal to one.")

    # get cv histogram
    cv_input = df[var]
    cv_hist = np.histogram(cv_input,bins,weights=scaling)[0]
    cv_counts = np.sum(cv_hist)

    syst_dict = {}
    nbins = len(bins) 

    for col in unisim_col: 
        # * for unisim, get straight from `morph`
        weights = df[col].to_numpy()
        weights[np.isnan(weights)] = 1.0
        weights *= scaling
        hists = np.apply_along_axis(get_hist, 0, weights, cv_input, bins)
        if normalize:
            hists = hists / np.sum(hists) * cv_counts
        syst_dict[col[2]] = {'hists': np.reshape(hists,(nbins-1,-1))}
    for col in multisig_col:
        # * for multisigma, get two universes, ps1 and ms1
        ps1_col = col 
        ms1_col = tuple([x if x!='ps1' else 'ms1' for x in list(col)])
        weights = np.stack([np.nan_to_num(df[ps1_col].to_numpy(),copy=False,nan=1.0),
                            np.nan_to_num(df[ms1_col].to_numpy(),copy=False,nan=1.0)]).T
        weights *= scaling[:,np.newaxis]
        hists = np.apply_along_axis(get_hist, 0, weights, cv_input, bins)
        if normalize:
            hists = hists / np.sum(hists, axis=0) * cv_counts
        syst_dict[col[2]] = {'hists': hists}
    for col in multisim_col:
        # * for multisim, get all universes automatically
        weights = df[col].to_numpy()
        weights[np.isnan(weights)] = 1.0
        weights *= scaling[:,np.newaxis]
        hists = np.apply_along_axis(get_hist, 0, weights, cv_input, bins)
        if normalize:
            hists = hists / np.sum(hists, axis=0) * cv_counts
        syst_dict[col[2]] = {'hists': hists}
    for key in syst_dict.keys():
        hists = syst_dict[key]['hists']
        cov, frac, corr = calc_matrices(hists,cv_hist)
        syst_dict[key].update({'cov': cov, 'cov_frac': frac, 'corr': corr})
    return syst_dict

def mcstat(indf, nuniv:int=100 , cols: list=['__ntuple','entry','rec.slc..index','run','subrun','evt','df_idx']) -> pd.DataFrame:
    """
    Add MC statistical uncertainty universes to the DataFrame.
    This function generates Poisson-fluctuated weights for each event based on unique seeds
    derived from specified columns. It creates `nuniv` universes of MC statistical weights.
    
    Heavily inspired by Mun's method in: `cafpyana/analysis_village/1mu1p0pi/wienersvd_unfolding.ipynb`
    
    Parameters
    ----------
    indf : pd.DataFrame
        Input dataframe, must contain the columns specified in `cols` for seed generation.
    nuniv : int, optional
        Number of MC statistical universes to create (default is 100).
    cols : list, optional
        List of column names to use for generating unique seeds (default includes common identifiers).
    Returns
    -------
    pd.DataFrame
        DataFrame with added MC statistical weight universes.
    Notes
    -----
    - Unique seeds are generated by hashing the specified columns for each event.
    - Poisson random numbers with mean 1.0 are generated for each universe using the combined seed.
    - The resulting weights are stored in a MultiIndex DataFrame under the 'slc', 'truth', 'MCstat' hierarchy.
    """
    df = indf.copy()
    # check if all the cols are in the input df 
    for col in cols:
        if col not in df.columns.get_level_values(0):
            raise ValueError(f"Column '{col}' not found in DataFrame columns.")
    
    seed_cols = [tuple([col] +[""]*(len(df.columns[0])-1)) for col in cols]
    seed_df = df.reset_index()[seed_cols]

    seed_df['unique_seed'] = seed_df.apply(lambda x: hash(tuple(x)) % 2**32, axis = 1)
    unique_seeds = seed_df.unique_seed.to_numpy()
    univ_seeds = [hash(f"universe_{x}")% 2**32 for x in range(nuniv)]

    mcstat_univ_cols = pd.MultiIndex.from_product([['slc'],['truth'],["MCstat"], [f"univ_{i}" for i in range(nuniv)], [""], [""]],)
    mcstat_univ_wgt = pd.DataFrame(1.0,index=df.index,columns=mcstat_univ_cols,)

    for iuniv in tqdm(range(nuniv)):
        combined_seeds = (unique_seeds + univ_seeds[iuniv]) % 2**32
        weights = np.array([
            np.random.default_rng(int(s)).poisson(1.0) for s in combined_seeds
        ])
        mcstat_univ_wgt[("slc", "truth", "MCstat", f"univ_{iuniv}", "", "")] = weights
    return df.join(mcstat_univ_wgt)


def get_detvar_systs(detvar_dict,stage,var, bins,normalize=False,**kwargs):
    """
    Compute detector variation systematic covariance matrices.

    Notes
    -----
    - Each detector variation is treated as a unisim.
    - Histograms are flux-POT-normalized using `this_dict['scale']`.
      Ensure `detvar_dict[key]['scale']` is set to 1.0 if no normalization is desired.
    - When `normalize=True`, the DetVar histogram is area-normalized to match the CV.
    """
    matrices_dict = {}
    for i, key in enumerate(detvar_dict.keys()): 
        this_dict = detvar_dict[key]
        this_dv   = this_dict['dv_df']
        this_cv   = this_dict['cv_df']
        # this is for flux-normalizing
        this_norm = this_dict['flux_pot_norm']
        
        # lexsort to avoid performance warning on columns 
        # forward selection kwargs to select function
        cv_hist = np.histogram(ensure_lexsorted(select(this_cv,**kwargs)[stage],axis=1)[var],bins=bins)[0]
        dv_hist = np.histogram(ensure_lexsorted(select(this_dv,**kwargs)[stage],axis=1)[var],bins=bins)[0]
        if normalize:
            dv_hist = dv_hist / np.sum(dv_hist) * np.sum(cv_hist)
        
        cov, cov_frac, corr = calc_matrices(var_arr=np.reshape(dv_hist/this_norm,(len(bins)-1,-1)),cv=cv_hist/this_norm)
        matrices_dict[key] = {'hists':np.reshape(dv_hist/this_norm,(len(bins)-1,-1)),
                              'cov': cov,
                              'cov_frac': cov_frac,
                              'corr': corr, 
                              'hist_cv': cv_hist/this_norm}
    return matrices_dict


def get_syst_df(dicts: list, cv_hist: np.ndarray) -> pd.DataFrame:
    """Extract diagonal systematic uncertainties from covariance matrices into a DataFrame.
    
    Parameters
    ----------
    dicts : list
        List of dictionaries containing systematic covariance matrices
    cv_hist : np.ndarray
        Central value histogram for normalization
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: key, category, unc, sum, top5
    """
    categories = ["GENIE", "Flux", "MCstat", "DetVar"]
    
    # Map categories to key extraction logic
    key_extractors = {
        "GENIE":  lambda key: key.split("_")[-1],  
        "Flux":   lambda key: key.split("_")[0],
        "MCstat": lambda key: key,
        "DetVar": lambda key: key.split("_")[-1]
    }
    
    records = []
    
    for d in dicts:
        for key in d:
            cov = d[key]['cov']
            unc = np.sqrt(np.diag(cov)) / cv_hist
            tot = np.sum(unc)/len(unc)

            category = next((cat for cat in categories if cat in key), None)            
            if category is None:
                print(f"Warning: category not found for key '{key}'")
                records.append({"key": key, "category": "Other", "unc": unc, 'sum': tot})
            else:
                extracted_key = key_extractors[category](key)
                records.append({"key": extracted_key, "category": category, "unc": unc, 'sum': tot})
    syst_df = pd.DataFrame(records).sort_values(['category','sum'],ascending=[False,False]) 
    syst_df['top5'] = syst_df.groupby('category')['sum'].rank(method='first', ascending=False) <= 5
    return syst_df