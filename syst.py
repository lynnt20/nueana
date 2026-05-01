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

__all__ = [
    'is_xsec',
    'calc_matrices',
    'get_xsec_hists',
    'get_syst_hists',
    'get_syst',
    'mcstat',
    'get_detvar_systs',
    'get_syst_df',
    'make_multiverse_weights',
]
from .utils import ensure_lexsorted, apply_event_mask
from .histogram import get_hist1d, get_hist2d
from .selection import select, define_signal
from .classes import XSecInputs
from .constants import integrated_flux
from makedf.geniesyst import regen_systematics, ar23p_genie_systematics
    
def is_xsec(col: tuple, xsec_inputs: XSecInputs | None) -> bool:
    """Check if event rate calculation should be used for cross-section systematics.

    Parameters
    ----------
    col : tuple
        MultiIndex column name. The knob name is expected at index 2
        (e.g. ``('slc', 'truth', '<knob>', ...)``).
    xsec_inputs : XSecInputs or None
        Cross-section inputs containing truth-level signal dataframe, scaling,
        and true-variable column mappings.

    Returns
    -------
    bool
        True if the knob is in regen_systematics or ar23p_genie_systematics
        and all xsec inputs are provided; False otherwise.
    """
    return (
        col[2] in _XSEC_KNOBS
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

def get_xsec_hists(reco_df: pd.DataFrame,
                   xsec_inputs: XSecInputs,
                   reco_weights: np.ndarray,
                   true_signal_weights: np.ndarray,
                   bins: np.ndarray,
                   reco_var_reco: str | tuple,
                   return_response: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
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
    return_response : bool, optional
        If True, also return the per-universe response matrix. Default False.

    Returns
    -------
    hists : np.ndarray, shape (n_bins, n_universes)
        Predicted reco event-rate histogram for each universe, equal to
        ``response_u @ cv_truth + bkg_u`` where ``response_u`` is the
        per-universe response matrix and ``bkg_u`` is the weighted
        background histogram.
    response : np.ndarray, shape (n_bins_reco, n_bins_true, n_universes)
        Per-universe response (smearing) matrix. Only returned when
        ``return_response=True``.
    """
    true_signal_df    = xsec_inputs.true_signal_df
    true_signal_scale = xsec_inputs.true_signal_scale
    reco_var_true     = xsec_inputs.reco_var_true
    true_var_true     = xsec_inputs.true_var_true
    
    true_signal_df = ensure_lexsorted(true_signal_df,axis=1)

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
    hists = sig_reco_hists + bkg_reco_hists
    if return_response:
        return hists, response
    return hists

def get_syst_hists(reco_df: pd.DataFrame,
                   reco_var: str | tuple,
                   bins: np.ndarray,
                   scale: bool = True,
                   xsec_inputs: XSecInputs | None = None,
                   multisim_nuniv=100,
                   save_response: bool = False) -> tuple[dict, np.ndarray]:
    """Compute only systematic universe histograms (no covariance/correlation matrices).

    Parameters
    ----------
    save_response : bool, optional
        If True, store the per-universe response matrix under ``syst_dict[key]['response']``
        for GENIE-type (xsec) systematics. Shape is ``(nbins_reco, nbins_true, nuniv)``.
        Default False.

    Returns
    -------
    tuple[dict, np.ndarray]
        (syst_hist_dict, cv_hist), where:
        - syst_hist_dict[key]['hists'] has shape (nbins, nuniv)
        - syst_hist_dict[key]['response'] has shape (nbins_reco, nbins_true, nuniv),
          present only for xsec systematics when ``save_response=True``
        - cv_hist is the flux-normalized CV histogram
    """
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
    nbins = len(bins)
    syst_dict = {}
    
    # unisim
    for col in tqdm(unisim_col, desc='Running through unisims'):
        weights = reco_df[col].values.astype(np.float64)
        weights[np.isnan(weights)] = 1.0
        weights[(weights>10) | (weights < 0)] = 1.0 
        weights *= scaling

        response = None
        if is_xsec(col, xsec_inputs):
            true_signal_weights = xsec_inputs.true_signal_df[col[2:]].values.astype(np.float64) * xsec_inputs.true_signal_scale
            result = get_xsec_hists(reco_df, xsec_inputs,
                                    weights.reshape(-1, 1),
                                    true_signal_weights.reshape(-1, 1),
                                    bins,
                                    reco_var,
                                    return_response=save_response)
            hists, response = result if save_response else (result, None)
        else:
            hists = np.apply_along_axis(get_hist1d, 0, weights, reco_df[reco_var], bins).reshape((nbins - 1, -1))

        entry = {'hists': hists}
        if response is not None:
            entry['response'] = response
        syst_dict[col[2]] = entry

    # multisig (ps1/ms1 pairs)
    for col in tqdm(multisig_col, desc='Running through multisig'):
        ps1_col = col
        ms1_col = tuple([x if x != "ps1" else "ms1" for x in list(col)])

        ps1 = np.nan_to_num(reco_df[ps1_col].values.astype(np.float64), copy=False, nan=1.0)
        ms1 = np.nan_to_num(reco_df[ms1_col].values.astype(np.float64), copy=False, nan=1.0)
        weights = np.stack([ps1, ms1]).T 
        weights[(weights>10) | (weights < 0)] = 1.0 
        weights *= scaling[:, np.newaxis]
        
        response = None
        if is_xsec(col, xsec_inputs):
            true_signal_ps1 = np.nan_to_num(xsec_inputs.true_signal_df[ps1_col[2:]].values.astype(np.float64), copy=False, nan=1.0)
            true_signal_ms1 = np.nan_to_num(xsec_inputs.true_signal_df[ms1_col[2:]].values.astype(np.float64), copy=False, nan=1.0)
            true_signal_weights = np.stack([true_signal_ps1, true_signal_ms1]).T * xsec_inputs.true_signal_scale
            result = get_xsec_hists(reco_df, xsec_inputs, weights, true_signal_weights, bins, reco_var,
                                    return_response=save_response)
            hists, response = result if save_response else (result, None)
        else:
            hists = np.apply_along_axis(get_hist1d, 0, weights, reco_df[reco_var], bins)

        entry = {'hists': hists}
        if response is not None:
            entry['response'] = response
        syst_dict[col[2]] = entry

    # multisim
    for col in tqdm(multisim_col, desc='Running through multisims'):
        weights = reco_df[col].values.astype(np.float64)
        weights[np.isnan(weights)] = 1.0
        weights[(weights>10) | (weights<0)] = 1
        weights *= scaling[:, np.newaxis]

        response = None
        if is_xsec(col, xsec_inputs):
            true_signal_weights = xsec_inputs.true_signal_df[col[2:]].values.astype(np.float64) * xsec_inputs.true_signal_scale
            result = get_xsec_hists(reco_df, xsec_inputs, weights, true_signal_weights, bins, reco_var,
                                    return_response=save_response)
            hists, response = result if save_response else (result, None)
        else:
            hists = np.apply_along_axis(get_hist1d, 0, weights, reco_df[reco_var], bins)

        entry = {'hists': hists}
        if response is not None:
            entry['response'] = response
        syst_dict[col[2]] = entry

    return syst_dict, cv

def get_syst(*args, save_response: bool = False, **kwargs) -> dict:
    """Backward-compatible API: returns hists + cov/cov_frac/corr per systematic.

    Parameters
    ----------
    save_response : bool, optional
        Forwarded to :func:`get_syst_hists`. When True, stores the per-universe
        response matrix under ``syst_dict[key]['response']`` for xsec systematics.
    """
    syst_dict, cv = get_syst_hists(*args, save_response=save_response, **kwargs)
    
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


def get_detvar_systs(detvar_dict, var, bins,
                     event_type: str | None = "all",
                     extra_cuts=None,
                     skip_cuts=None,
                     **selection_kwargs):
    """Compute detector variation systematic covariance matrices.

    Parameters
    ----------
    detvar_dict : dict
        Dictionary mapping detector variation names to dictionaries containing:
        - 'dv_df': DataFrame or list of DataFrames with detector variations
        - 'cv_df': DataFrame with central value
        - 'pot': POT for flux normalization
    var : str or tuple
        Column name for the variable to histogram.
    bins : np.ndarray
        Bin edges for histogramming.
    event_type : str or None, default 'all'
        Event mask applied after selection (see :func:`~nueana.utils.apply_event_mask`).
    extra_cuts : callable or list of callable, optional
        Additional cut function(s) applied after the standard selection pipeline.
        Each must accept a DataFrame and return a boolean mask, e.g.
        ``lambda df: abs(df.x) > 10``. Forwarded to :func:`~nueana.selection.select`.
    skip_cuts : list of str, optional
        Named selection stages to skip. Forwarded to :func:`~nueana.selection.select`.
        Valid names: ``'preselection'``, ``'flash matching'``, ``'shower energy'``,
        ``'muon rejection'``, ``'conversion gap'``, ``'dEdx'``,
        ``'opening angle'``, ``'shower length'``.
    **selection_kwargs
        Any other keyword arguments forwarded to :func:`~nueana.selection.select`
        (e.g. ``min_dedx``, ``max_track_length``).

    Returns
    -------
    dict
        Dictionary mapping detector variation names to dictionaries containing:
        - 'hists': per-variation histograms, shape (nbins, nuniv)
        - 'cov': covariance matrix
        - 'cov_frac': fractional covariance matrix
        - 'corr': correlation matrix
        - 'hist_cv': central-value histogram

    Notes
    -----
    If ``this_dict['dv_df']`` is a single DataFrame the variation is treated as
    a unisim; if it is a list of DataFrames it is treated as a multisim.
    """
    _needs_select = bool(selection_kwargs) or extra_cuts is not None or skip_cuts is not None
    _sel_kw = dict(savedict=False, extra_cuts=extra_cuts, skip_cuts=skip_cuts, **selection_kwargs)

    matrices_dict = {}
    for i, key in tqdm(enumerate(detvar_dict.keys())):
        this_dict = detvar_dict[key]
        this_dv   = this_dict['dv_df']
        this_cv   = this_dict['cv_df']
        this_norm = integrated_flux * (this_dict['pot'] / 1e6)

        cv_sel = select(this_cv, **_sel_kw) if _needs_select else this_cv
        cv_sel = apply_event_mask(ensure_lexsorted(cv_sel, axis=1), event_type)
        cv_hist = get_hist1d(data=cv_sel[var], bins=bins) / this_norm

        dv_dfs = this_dv if isinstance(this_dv, list) else [this_dv]
        if _needs_select:
            dv_dfs = [select(dv, **_sel_kw) for dv in dv_dfs]
        dv_hists = np.column_stack([
            get_hist1d(data=apply_event_mask(ensure_lexsorted(dv, axis=1), event_type)[var], bins=bins)
            for dv in dv_dfs
        ]) / this_norm

        cov, cov_frac, corr = calc_matrices(var_arr=dv_hists, cv=cv_hist)
        matrices_dict[key] = {
            'hists':    dv_hists,
            'cov':      cov,
            'cov_frac': cov_frac,
            'corr':     corr,
            'hist_cv':  cv_hist,
        }
    return matrices_dict


# Keys that should be classified as GENIE but don't contain "GENIE" in their name.
# Extracted keys for these get a "+" suffix to flag the special treatment.
_GENIE_ALIASES = frozenset({"SBNNuSyst", "SuSAv2"})

# Full set of GENIE knob names that use the xsec (event-rate) calculation path.
_XSEC_KNOBS = frozenset(regen_systematics + ar23p_genie_systematics)

# Ordered list of (subcategory, substrings) for DetVar classification.
# Checked top-to-bottom; first match wins. The calorimetry entry also
# catches keys where "r" appears as a standalone token (recombination).
_DETVAR_SUBCATEGORIES: list[tuple[str, list[str]]] = [
    ("WireMod",     ["wiremod"]),
    ("SCE",         ["sce"]),
    ("PMT",         ["pmt"]),
    ("calorimetry", ["ccal", "phi", "alpha", "beta90", "beta_90", "betap90","Ecorr",'yz']),
]

_CATEGORY_KEYWORDS = ["GENIE", "Flux", "MCstat", "DetVar", "Geant4"]


def _extract_genie_key(key: str) -> str:
    """Extract GENIE systematic key.
    
    For standard GENIE keys with multisigma/multisim pattern, extract text after the pattern.
    For special cases like MECq0q3InterpWeighting, format as Model_MEC_q0binN.
    
    Parameters
    ----------
    key : str
        The full GENIE systematic key.
    
    Returns
    -------
    str
        Extracted key fragment.
    """
    # Try extracting after multisigma_ or multisim_ pattern
    for pattern in ["multisigma_", "multisim_"]:
        if pattern in key:
            return key.split(pattern, 1)[1]
    
    # Special handling for MECq0q3InterpWeighting keys
    # Format: MECq0q3InterpWeighting_SuSAv2To{Model}_q0binned_MECResponse_q0bin{N}
    if "MECq0q3InterpWeighting" in key:
        parts = key.split("_")
        # parts[1] contains "SuSAv2ToValenica" or "SuSAv2ToMartini" etc.
        model = parts[1].split("To")[1]  # Extract text after "To"
        q0_bin = parts[-1]  # Last part is "q0bin0", "q0bin1", etc.
        return f"{model}_MEC_{q0_bin}"
    
    # Fallback to old behavior for unrecognized patterns
    return "_".join(key.split("_")[4:])


_KEY_EXTRACTORS = {
    "GENIE":  _extract_genie_key,
    "Flux":   lambda key: key.split("_")[0],
    "MCstat": lambda key: key,
    "DetVar": lambda key: "".join(key.split("_")[1:]),
    "Geant4": lambda key: key.split("_")[1],
}


def _classify_category(key: str) -> str | None:
    """Map a raw systematic key to its high-level category, or None if unknown."""
    if any(alias in key for alias in _GENIE_ALIASES):
        return "GENIE"
    return next((cat for cat in _CATEGORY_KEYWORDS if cat in key), None)


def _classify_detvar_subcategory(detvar_key: str) -> str:
    """Map a detector variation key to its analysis subcategory."""
    key    = detvar_key.lower()
    tokens = key.replace("-", "_").split("_")
    for subcategory, keywords in _DETVAR_SUBCATEGORIES:
        if any(kw in key for kw in keywords):
            return subcategory
    if "r" in tokens:
        return "calorimetry"
    return "other"


def get_syst_df(dicts: list, cv_hist: np.ndarray) -> pd.DataFrame:
    """Extract diagonal systematic uncertainties from covariance matrices into a DataFrame.

    Parameters
    ----------
    dicts : list
        List of systematic dictionaries (each maps key → ``{'cov': ndarray, ...}``).
    cv_hist : np.ndarray
        Central-value histogram used to convert absolute uncertainties to fractional.

    Returns
    -------
    pd.DataFrame
        Columns: ``key``, ``category``, ``subcategory``, ``unc``, ``sum``, ``top5``.
        Sorted by category then mean fractional uncertainty (descending).
        ``top5`` flags the five largest sources per category.
    """
    records = []

    for d in dicts:
        for raw_key in d:
            cov = d[raw_key]['cov']
            unc = np.sqrt(np.diag(cov)) / cv_hist
            tot = float(np.mean(unc))

            category = _classify_category(raw_key)
            if category is None:
                print(f"Warning: category not found for key '{raw_key}'")
                records.append({
                    "key": raw_key, "category": "Other", "subcategory": "Other",
                    "unc": unc, "sum": tot,
                })
                continue

            extracted_key = _KEY_EXTRACTORS[category](raw_key)
            if category == "GENIE" and any(alias in raw_key for alias in _GENIE_ALIASES):
                extracted_key += "+"

            subcategory = _classify_detvar_subcategory(extracted_key) if category == "DetVar" else category
            records.append({
                "key": extracted_key, "category": category, "subcategory": subcategory,
                "unc": unc, "sum": tot,
            })

    syst_df = pd.DataFrame(records).sort_values(['category', 'sum'], ascending=[False, False])
    syst_df['top5'] = syst_df.groupby('category')['sum'].rank(method='first', ascending=False) <= 5
    return syst_df

def make_multiverse_weights(nudf, evtdf, knob_list, n_univs=100, evt_prefix=None):
    
    def _seed(knob, i, df_idx):
        np.random.seed(hash(knob + str(i) + str(df_idx)) % (2**32))

    def _cap(wgt):
        return np.minimum(np.maximum(wgt, 0), 10)

    def _morph_wgt(base_weight):
        return _cap(1 + (base_weight - 1) * 2 * np.abs(np.random.normal(0, 1)))

    def _multisigma_wgt(sigma_weight):
        return _cap(1 + (sigma_weight - 1) * np.random.normal(0, 1))

    # evt_prefix: tuple of column levels to prepend to knob name in evtdf (e.g., ('slc', 'truth'))
    new_columns_nudf = {}
    new_columns_evtdf = {}
    
    if nudf.index.names != evtdf.index.names:
        raise ValueError("Index names of nudf and evtdf must match.")

    for knob in tqdm(knob_list):
        evt_knob_key = (evt_prefix + (knob,)) if evt_prefix else knob

        ## nudf
        this_cols_nudf = nudf[knob].columns
        if len(this_cols_nudf) == 1:  # morph
            for i in range(n_univs):
                _seed(knob, i, 0)
                new_columns_nudf[(knob, f"univ_{i}")] = _morph_wgt(nudf[knob].morph)
        elif len(this_cols_nudf) == 7:  # multisigma
            for i in range(n_univs):
                _seed(knob, i, 0)
                new_columns_nudf[(knob, f"univ_{i}")] = _multisigma_wgt(nudf[knob].ps1)

        ## evtdf
        this_cols_evtdf = evtdf[evt_knob_key].columns
        if len(this_cols_evtdf) == 1:  # morph
            for i in range(n_univs):
                _seed(knob, i, 1)
                new_columns_evtdf[evt_knob_key + (f"univ_{i}",)] = _morph_wgt(evtdf[evt_knob_key].morph)
        elif len(this_cols_evtdf) == 7:  # multisigma
            for i in range(n_univs):
                _seed(knob, i, 1)
                new_columns_evtdf[evt_knob_key + (f"univ_{i}",)] = _multisigma_wgt(evtdf[evt_knob_key].ps1)

    # Convert dict to DataFrame
    new_cols_nudf = pd.DataFrame(new_columns_nudf, index=nudf.index)
    new_cols_evtdf = pd.DataFrame(new_columns_evtdf, index=evtdf.index)

    # Pad column tuples to match original MultiIndex level count
    if len(new_cols_nudf.columns) > 0:
        n_levels_nu = nudf.columns.nlevels
        padded_cols_nu = [col + ("",) * (n_levels_nu - len(col)) for col in new_cols_nudf.columns]
        new_cols_nudf.columns = pd.MultiIndex.from_tuples(padded_cols_nu, names=nudf.columns.names)

    if len(new_cols_evtdf.columns) > 0:
        n_levels_evt = evtdf.columns.nlevels
        padded_cols_evt = [col + ("",) * (n_levels_evt - len(col)) for col in new_cols_evtdf.columns]
        new_cols_evtdf.columns = pd.MultiIndex.from_tuples(padded_cols_evt, names=evtdf.columns.names)

    nudf = pd.concat([nudf, new_cols_nudf], axis=1)
    evtdf = pd.concat([evtdf, new_cols_evtdf], axis=1)

    # Synchronize univ weights using the shared row index
    for knob in knob_list:
        if "multisim" in knob:
            continue
        target_cols = [
            col for col in nudf.columns
            if knob in "_".join(list(col)) and 'univ_' in "_".join(list(col))
        ]
        mapped_vals = nudf[target_cols][nudf.index.isin(evtdf.index)]
        evt_target_cols = [(evt_prefix + col) if evt_prefix else col for col in target_cols]
        n_levels_evt = evtdf.columns.nlevels
        evt_target_cols = [col + ("",) * (n_levels_evt - len(col)) for col in evt_target_cols]
        if mapped_vals.isna().any().any():
            print(f"Found NaN values in mapped_vals for knob: {knob}")
        evtdf[evt_target_cols] = mapped_vals

    return nudf, evtdf