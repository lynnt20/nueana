"""
Systematic and statistical uncertainty utilities.

Conventions
-----------
- All output histograms and covariance matrices are **flux-normalized by default**.
Functions that support disabling this accept a `scale=True` parameter.
- Covariance matrices are normalized by N_universes.
- NaN weights (e.g., GENIE weights for true cosmics) are replaced with 1.0.
"""

import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
from .utils import ensure_lexsorted, apply_event_mask
from .histogram import get_hist1d, get_hist2d
from .selection import select, define_signal
from .classes import XSecInputs
from .constants import integrated_flux

def _expand_weights(df, col, multisim_nuniv, scalars=None):
        weights = df[col].values.astype(np.float64)
        isnan = np.isnan(weights)

        if scalars is None:
            i_vals = np.arange(multisim_nuniv)
            seed_strs = np.char.add(np.char.add("".join(list(col)), i_vals.astype(str)), str(id(df)))
            seed_ints = np.fromiter((hash(s) % (2**32) for s in seed_strs), dtype=np.uint32, count=len(i_vals))
            scalars = np.array([np.random.default_rng(int(s)).normal(0, 1) for s in seed_ints], dtype=np.float64)

        # One random throw per universe (coherent across all events).
        weights_mlt = 1 + (weights - 1)[:, np.newaxis] * scalars[np.newaxis, :]
        weights_mlt = np.maximum(weights_mlt, 0)
        weights_mlt[isnan, :] = 1.0  # reset NaN weights to 1.0
        return weights_mlt
    
def is_xsec(col: tuple, xsec_inputs: XSecInputs | None) -> bool:
    """Check if event rate calculation should be used for cross-section systematics.
    
    Parameters
    ----------
    col : tuple
        MultiIndex column name.
    xsec_inputs : XSecInputs or None
        Cross-section inputs containing truth-level signal dataframe, scaling,
        and true-variable column mappings.
    
    Returns
    -------
    bool
        True if column contains "GENIE" and all xsec inputs are provided; False otherwise.
    """
    return (
        "GENIE" in "".join(list(col))
        and xsec_inputs is not None
        and xsec_inputs.true_signal_df is not None
        and xsec_inputs.reco_var_true is not None
        and xsec_inputs.true_var_true is not None
    )

def calc_matrices(var_arr: np.ndarray, cv: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

def get_xsec_hists(reco_df: pd.DataFrame,
                   xsec_inputs: XSecInputs,
                   reco_weights: np.ndarray,
                   true_signal_weights: np.ndarray,
                   bins: np.ndarray,
                   reco_var_reco: str | tuple) -> np.ndarray:
    """Compute predicted event-rate histograms for cross-section systematic universes.
    
    Parameters
    ----------
    reco_df : pd.DataFrame
        Selected (reco-level) DataFrame. Must contain a ``signal`` column
        (0 = signal, nonzero = background) and the reco/true variable columns.
    xsec_inputs : XSecInputs
        Cross-section inputs with true signal dataframe, scaling, and variable mappings.
    reco_weights : np.ndarray, shape (n_selected, n_universes)
        Per-event systematic weights for the reco-level selected sample.
    true_signal_weights : np.ndarray, shape (n_signal, n_universes)
        Per-event systematic weights for the truth-level signal sample.
    reco_var_reco : tuple
        Column key for the reco-level variable to histogram from the reco-level DataFrame.
        (e.g. ``('primshw','shw','reco_energy')``).
    bins : np.ndarray
        Bin edges (shared for reco and true axes). Overflow is folded into
        the last bin.

    Returns
    -------
    np.ndarray, shape (n_bins, n_universes)
        Predicted reco event-rate histogram for each universe, equal to
        ``response_u @ cv_truth + bkg_u`` where ``response_u`` is the
        per-universe response matrix and ``bkg_u`` is the weighted
        background histogram.
    """
    true_signal_df    = xsec_inputs.true_signal_df
    true_signal_scale = xsec_inputs.true_signal_scale
    reco_var_true     = xsec_inputs.reco_var_true
    true_var_true     = xsec_inputs.true_var_true

    true_signal_df = true_signal_df[true_signal_df.signal==0]  # ensure signal-only truth sample
    signal_mask = reco_df.signal==0
    smearing = np.apply_along_axis(get_hist2d,0,
                                   reco_weights[signal_mask],
                                   reco_df[signal_mask][reco_var_reco],
                                   reco_df[signal_mask][reco_var_true],
                                   bins)

    sig_hist_univ = np.apply_along_axis(get_hist1d,0,
                                        true_signal_weights,
                                        true_signal_df[true_var_true],bins)
    sig_hist_cv = get_hist1d(np.ones(len(true_signal_df))*true_signal_scale,
                              true_signal_df[true_var_true],
                              bins)
    
    response = np.divide(smearing,sig_hist_univ,
                         out=np.zeros_like(smearing),
                         where=sig_hist_univ>0)
    # response x cv 
    sig_reco_hists = np.einsum('ijk,j->ik', response, sig_hist_cv)
    bkg_reco_hists = np.apply_along_axis(get_hist1d,0,
                                         reco_weights[~signal_mask],
                                         reco_df[~signal_mask][reco_var_reco],
                                         bins)
    return sig_reco_hists + bkg_reco_hists

def get_syst_hists(reco_df: pd.DataFrame,
                   reco_var: str | tuple,
                   bins: np.ndarray,
                   scale: bool = True,
                   xsec_inputs: XSecInputs | None = None,
                   expand:bool=False,
                   multisim_nuniv=100) -> tuple[dict, np.ndarray]:
    """Compute only systematic universe histograms (no covariance/correlation matrices).

    Returns
    -------
    tuple[dict, np.ndarray]
        (syst_hist_dict, cv_hist), where:
        - syst_hist_dict[key]['hists'] has shape (nbins, nuniv)
        - cv_hist is the flux-normalized CV histogram
    """
    reco_df = ensure_lexsorted(reco_df, axis=0)
    reco_df = ensure_lexsorted(reco_df, axis=1)

    unisim_col, multisig_col, multisim_col = [], [], []
    univ_level = -1

    for col in reco_df.columns:
        if "univ" in "".join(list(col)):
            for i, x in enumerate(col):
                if str(x).startswith("univ"):
                    univ_level = i
                    break
            break

    scaling = np.ones(reco_df.shape[0])
    for col in reco_df.columns:
        if ("flux_pot_norm" in col) and scale:
            scaling = reco_df[col].values
        if "morph" in col:
            unisim_col.append(tuple(filter(None, col)))
        elif "ps1" in col:
            multisig_col.append(tuple(filter(None, col)))
        elif "univ" in "".join(list(col)):
            base = tuple(filter(None, col))[:univ_level]
            if base not in multisim_col:
                multisim_col.append(base)

    if np.array_equal(scaling, np.ones(reco_df.shape[0])) and scale:
        print("No flux-averaged POT normalization found; flux normalization will be equal to one.")

    cv = get_hist1d(data=reco_df[reco_var], bins=bins, weights=scaling)
    cv_counts = np.sum(cv)
    nbins = len(bins)
    syst_dict = {}

    if expand:
        # use morph and ps1 as 1-sigma variations to create multisim universes with random throws
        for col in tqdm(unisim_col+multisig_col, desc='Running through unisims'):
            i_vals = np.arange(multisim_nuniv)
            seed_strs = np.char.add(np.char.add("".join(list(col)), i_vals.astype(str)), str(id(reco_df)))
            seed_ints = np.fromiter((hash(s) % (2**32) for s in seed_strs), dtype=np.uint32, count=len(i_vals))
            scalars = np.array([np.random.default_rng(int(s)).normal(0, 1) for s in seed_ints], dtype=np.float64)

            weights_mlt = _expand_weights(reco_df, col, multisim_nuniv, scalars=scalars)
            weights_mlt *= scaling[:, np.newaxis]
            if is_xsec(col, xsec_inputs):
                true_weights_mlt = _expand_weights(
                    xsec_inputs.true_signal_df,
                    col[2:],
                    multisim_nuniv,
                    scalars=scalars,
                ) * xsec_inputs.true_signal_scale
                hists = get_xsec_hists(reco_df, xsec_inputs, weights_mlt, true_weights_mlt, bins, reco_var)
            else:
                hists = np.apply_along_axis(get_hist1d, 0, weights_mlt, reco_df[reco_var], bins)

            syst_dict[col[2]] = {'hists': hists}
    else: 
        # unisim
        for col in tqdm(unisim_col, desc='Running through unisims'):
            weights = reco_df[col].values.astype(np.float64)
            weights[np.isnan(weights)] = 1.0
            weights[(weights>10) | (weights < 0)] = 1.0 
            weights *= scaling

            if is_xsec(col, xsec_inputs):
                true_signal_weights = xsec_inputs.true_signal_df[col[2:]].values.astype(np.float64) * xsec_inputs.true_signal_scale
                hists = get_xsec_hists(reco_df, xsec_inputs,
                                    weights.reshape(-1, 1),
                                    true_signal_weights.reshape(-1, 1),
                                    bins,
                                    reco_var)
            else:
                hists = np.apply_along_axis(get_hist1d, 0, weights, reco_df[reco_var], bins).reshape((nbins - 1, -1))

            syst_dict[col[2]] = {'hists': hists}

        # multisig (ps1/ms1 pairs)
        for col in tqdm(multisig_col, desc='Running through multisig'):
            ps1_col = col
            ms1_col = tuple([x if x != "ps1" else "ms1" for x in list(col)])

            ps1 = np.nan_to_num(reco_df[ps1_col].values.astype(np.float64), copy=False, nan=1.0)
            ms1 = np.nan_to_num(reco_df[ms1_col].values.astype(np.float64), copy=False, nan=1.0)
            ps1[(ps1>10) | (ps1<0)] = 1  # cap extreme outliers to avoid dominating the histograms
            ms1[(ms1>10) | (ms1<0)] = 1
            weights = np.stack([ps1, ms1]).T * scaling[:, np.newaxis]
            
            if is_xsec(col, xsec_inputs):
                true_signal_ps1 = np.nan_to_num(xsec_inputs.true_signal_df[ps1_col[2:]].values.astype(np.float64), copy=False, nan=1.0)
                true_signal_ms1 = np.nan_to_num(xsec_inputs.true_signal_df[ms1_col[2:]].values.astype(np.float64), copy=False, nan=1.0)
                true_signal_weights = np.stack([true_signal_ps1, true_signal_ms1]).T * xsec_inputs.true_signal_scale
                hists = get_xsec_hists(reco_df, xsec_inputs, weights, true_signal_weights, bins, reco_var)
            else:
                hists = np.apply_along_axis(get_hist1d, 0, weights, reco_df[reco_var], bins)

            syst_dict[col[2]] = {'hists': hists}

    # multisim
    for col in tqdm(multisim_col, desc='Running through multisims'):
        weights = reco_df[col].values.astype(np.float64)
        weights[np.isnan(weights)] = 1.0
        weights[(weights>10) | (weights<0)] = 1
        weights *= scaling[:, np.newaxis]

        if is_xsec(col, xsec_inputs):
            true_signal_weights = xsec_inputs.true_signal_df[col[2:]].values.astype(np.float64) * xsec_inputs.true_signal_scale
            hists = get_xsec_hists(reco_df, xsec_inputs, weights, true_signal_weights, bins, reco_var)
        else:
            hists = np.apply_along_axis(get_hist1d, 0, weights, reco_df[reco_var], bins)

        syst_dict[col[2]] = {'hists': hists}

    return syst_dict, cv

def get_syst(*args, **kwargs) -> dict:
    """Backward-compatible API: returns hists + cov/cov_frac/corr per systematic."""
    syst_dict, cv = get_syst_hists(*args, **kwargs)
    
    for key in syst_dict:
        cov, cov_frac, corr = calc_matrices(syst_dict[key]['hists'], cv)
        syst_dict[key]['cov'] = cov
        syst_dict[key]['cov_frac'] = cov_frac
        syst_dict[key]['corr'] = corr

    return syst_dict

def mcstat(indf, nuniv:int=100 , cols: list=['__ntuple','entry','rec.slc..index','run','subrun','evt','sample']) -> pd.DataFrame:
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


def get_detvar_systs(detvar_dict,var,bins,event_type: str | None = "all",**selection_kwargs):
    """Compute detector variation systematic covariance matrices.
    
    Parameters
    ----------
    detvar_dict : dict
        Dictionary mapping detector variation names to dictionaries containing:
        - 'dv_df': DataFrame or list of DataFrames with detector variations
        - 'cv_df': DataFrame with central value
        - 'flux_pot_norm': Normalization factor
    var : str or tuple
        Column name for the variable to histogram.
    bins : np.ndarray
        Bin edges for histogramming.
    **selection_kwargs
        Additional keyword arguments forwarded to the `select` function.

    Returns
    -------
    dict
        Dictionary mapping detector variation names to dictionaries containing:
        - 'hists': Histograms per variation
        - 'cov': Covariance matrix
        - 'cov_frac': Fractional covariance matrix
        - 'corr': Correlation matrix
        - 'hist_cv': Central value histogram

    Notes
    -----
    - If ``this_dict['dv_df']`` is a single DataFrame, the variation is treated
      as a **unisim** (one universe).
    - If ``this_dict['dv_df']`` is a list of DataFrames, the variation is treated
      as a **multisim** (one universe per DataFrame).
    - Histograms are flux-POT-normalized using ``this_dict['flux_pot_norm']``.
      Ensure ``detvar_dict[key]['flux_pot_norm']`` is set to 1.0 if no normalization is desired.
    - When ``normalize=True``, each DetVar histogram is area-normalized to match the CV.
    """
    matrices_dict = {}
    for i, key in tqdm(enumerate(detvar_dict.keys())): 
        this_dict = detvar_dict[key]
        this_dv   = this_dict['dv_df']
        this_cv   = this_dict['cv_df']
        # this is for flux-normalizing
        # re-normalize in case flux calculation has changed
        this_norm = integrated_flux*(this_dict['pot']/1e6)
        
        # lexsort to avoid performance warning on columns 
        # forward selection kwargs to select function
        if selection_kwargs:
            cv_sel = select(this_cv, savedict=False, **selection_kwargs)
        else:
            cv_sel = this_cv
        cv_sel = apply_event_mask(ensure_lexsorted(cv_sel, axis=1), event_type)
        cv_hist = get_hist1d(data=cv_sel[var],bins=bins)/this_norm

        # support both unisim (single df) and multisim (list of dfs)
        dv_dfs = this_dv if isinstance(this_dv, list) else [this_dv]
        if selection_kwargs:
            dv_dfs = [select(dv, savedict=False, **selection_kwargs) for dv in dv_dfs]
        dv_hists = np.column_stack([
            get_hist1d(
                data=apply_event_mask(ensure_lexsorted(dv, axis=1),event_type)[var],bins=bins)
            for dv in dv_dfs
        ])/this_norm  # shape: (nbins, nuniv)
        
        cov, cov_frac, corr = calc_matrices(var_arr=dv_hists, cv=cv_hist)
        matrices_dict[key] = {'hists': dv_hists,
                              'cov': cov,
                              'cov_frac': cov_frac,
                              'corr': corr, 
                              'hist_cv': cv_hist}
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
    categories = ["GENIE", "Flux", "MCstat", "DetVar",'Geant4']

    def classify_category(key: str) -> str | None:
        """Map raw systematic keys to high-level categories."""
        if ("SBNNuSyst" in key) or ("SuSAv2" in key):
            return "GENIE"
        return next((cat for cat in categories if cat in key), None)

    def classify_detvar_subcategory(detvar_key: str) -> str:
        """Map detector variation names to analysis subcategories."""
        key = detvar_key.lower()
        tokens = key.replace("-", "_").split("_")

        if "wiremod" in key or "wire_mod" in key or "wire" in key:
            return "WireMod"
        if "sce" in key or "spacecharge" in key or "space_charge" in key:
            return "SCE"
        if "pmt" in key or "opdet" in key or "opdetector" in key or "light" in key:
            return "PMT"
        if (
            any(tag in key for tag in ["ccal", "phi", "alpha", "beta90", "beta_90", "betap90"])
            or "r" in tokens
        ):
            return "calorimetry"
        return "other"
    
    # Map categories to key extraction logic
    key_extractors = {
        "GENIE":  lambda key: "_".join(key.split("_")[4:]),  
        "Flux":   lambda key: key.split("_")[0],
        "MCstat": lambda key: key,
        "DetVar": lambda key: "".join(key.split("_")[1:]),
        "Geant4": lambda key: key.split("_")[1],
    }
    
    records = []
    
    for d in dicts:
        for key in d:
            cov = d[key]['cov']
            unc = np.sqrt(np.diag(cov)) / cv_hist
            tot = np.sum(unc)/len(unc)

            category = classify_category(key)
            if category is None:
                print(f"Warning: category not found for key '{key}'")
                records.append({
                    "key": key,
                    "category": "Other",
                    "subcategory": "Other",
                    "unc": unc,
                    "sum": tot,
                })
            else:
                extracted_key = key_extractors[category](key)
                subcategory = category
                if category == "DetVar":
                    subcategory = classify_detvar_subcategory(extracted_key)

                records.append({
                    "key": extracted_key,
                    "category": category,
                    "subcategory": subcategory,
                    "unc": unc,
                    "sum": tot,
                })
    syst_df = pd.DataFrame(records).sort_values(['category','sum'],ascending=[False,False]) 
    syst_df['top5'] = syst_df.groupby('category')['sum'].rank(method='first', ascending=False) <= 5
    return syst_df