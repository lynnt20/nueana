"""
Systematic and statistical uncertainty utilities.

Conventions
-----------
- All output histograms and covariance matrices are **flux-normalized by default**.
Functions that support disabling this accept a `scale=True` parameter.
- Covariance matrices are normalized by N_universes.
- NaN weights (e.g., GENIE weights for true cosmics) are replaced with 1.0.
"""
from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm

__all__ = [
    'is_xsec',
    'calc_matrices',
    'decompose_cov',
    'key_in_allowed',
    'get_xsec_hists',
    'get_syst_hists',
    'get_syst',
    'mcstat',
    'get_detvar_systs',
    'get_syst_df',
    'make_multiverse_weights',
    'slim_multisim_weights',
]
from .utils import ensure_lexsorted, apply_event_mask
from .utils import get_hist1d, get_hist2d, digitize_with_overflow
from .selection import select
from .analysis import define_signal
from .classes import XSecInputs
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

def decompose_cov(C: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decompose a covariance matrix into normalization and shape components.

    The normalization component is the fully-correlated outer-product piece that
    scales all bins proportionally. The shape component is the remainder, and
    by construction has zero net normalization (``C_shape.sum() == 0``).

    Parameters
    ----------
    C : np.ndarray, shape (n, n)
        Covariance matrix containing both normalization and shape contributions.
    h : np.ndarray, shape (n,)
        Nominal histogram (central-value bin counts) used to define the
        normalization direction.

    Returns
    -------
    C_norm : np.ndarray, shape (n, n)
        Normalization covariance: ``outer(h, h) * sum(C) / sum(h)**2``.
    C_shape : np.ndarray, shape (n, n)
        Shape covariance: ``C - C_norm``.

    Notes
    -----
    ``C_shape`` can have small negative eigenvalues near zero from floating-point
    subtraction. This is expected and harmless for error-band plotting; use a
    pseudoinverse if you need to invert ``C_shape`` (e.g. for a chi-square).
    """
    N = float(h.sum())
    C_norm  = np.outer(h, h) * float(C.sum()) / N**2
    C_shape = C - C_norm
    return C_norm, C_shape


def _get_xsec_hists_inner(
    smear_flat_idx: np.ndarray,
    w_sig: np.ndarray,
    truth_sig_idx: np.ndarray,
    true_signal_weights: np.ndarray,
    sig_hist_cv: np.ndarray,
    bkg_reco_idx: np.ndarray,
    w_bkg: np.ndarray,
    n_bins: int,
    return_response: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Core xsec histogram computation using pre-digitized indices.

    Called by :func:`get_xsec_hists` (single call) and by
    :func:`get_syst_hists` (which pre-computes the indices once across all knobs).

    Parameters
    ----------
    smear_flat_idx : np.ndarray of int, shape (n_sig,)
        Flattened 2D smearing-matrix index for signal events:
        ``sig_reco_idx * n_bins + sig_true_idx``.
    w_sig : np.ndarray, shape (n_sig, n_univs)
        Scaled weights for selected signal events.
    truth_sig_idx : np.ndarray of int, shape (n_truth,)
        Bin indices for truth-level signal events (from the truth DataFrame).
    true_signal_weights : np.ndarray, shape (n_truth, n_univs)
        Weights for truth-level signal events.
    sig_hist_cv : np.ndarray, shape (n_bins,)
        CV truth-level signal histogram.
    bkg_reco_idx : np.ndarray of int, shape (n_bkg,)
        Bin indices for background events in the reco variable.
    w_bkg : np.ndarray, shape (n_bkg, n_univs)
        Scaled weights for background events.
    n_bins : int
        Number of histogram bins (``len(bins) - 1``).
    return_response : bool, optional
        If True, also return the per-universe response matrix.
    """
    smearing = np.array([
        np.bincount(smear_flat_idx, weights=w_sig[:, u], minlength=n_bins * n_bins)
        for u in range(w_sig.shape[1])
    ], dtype=float).T.reshape(n_bins, n_bins, -1)

    sig_hist_univ = np.array([
        np.bincount(truth_sig_idx, weights=true_signal_weights[:, u], minlength=n_bins)
        for u in range(true_signal_weights.shape[1])
    ], dtype=float).T

    response = np.divide(smearing, sig_hist_univ,
                         out=np.zeros(smearing.shape, dtype=np.float64),
                         where=sig_hist_univ > 0)
    sig_reco_hists = np.einsum('ijk,j->ik', response, sig_hist_cv)

    bkg_reco_hists = np.array([
        np.bincount(bkg_reco_idx, weights=w_bkg[:, u], minlength=n_bins)
        for u in range(w_bkg.shape[1])
    ]).T

    hists = sig_reco_hists + bkg_reco_hists
    if return_response:
        return hists, response
    return hists


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
    true_signal_df    = xsec_inputs.true_signal_df   # pre-filtered & lexsorted by XSecInputs
    true_signal_scale = xsec_inputs.true_signal_scale
    reco_var_true     = xsec_inputs.reco_var_true
    true_var_true     = xsec_inputs.true_var_true

    n_bins      = len(bins) - 1
    signal_mask = reco_df.signal == 0

    reco_sig      = reco_df[signal_mask]
    sig_reco_idx  = digitize_with_overflow(reco_sig[reco_var_reco], bins)
    sig_true_idx  = digitize_with_overflow(reco_sig[reco_var_true], bins)
    smear_flat_idx = sig_reco_idx * n_bins + sig_true_idx
    w_sig         = reco_weights[signal_mask]

    truth_sig_idx = digitize_with_overflow(true_signal_df[true_var_true], bins)
    sig_hist_cv   = get_hist1d(np.ones(len(true_signal_df)) * true_signal_scale,
                               true_signal_df[true_var_true], bins)

    bkg_reco_idx = digitize_with_overflow(reco_df[~signal_mask][reco_var_reco], bins)
    w_bkg        = reco_weights[~signal_mask]

    return _get_xsec_hists_inner(smear_flat_idx, w_sig, truth_sig_idx, true_signal_weights,
                                  sig_hist_cv, bkg_reco_idx, w_bkg, n_bins, return_response)

def get_syst_hists(reco_df: pd.DataFrame,
                   reco_var: str | tuple,
                   bins: np.ndarray,
                   scale: bool = True,
                   mcbnb_pot: float | None = None,
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

    if scale and 'weights_mc' in reco_df.columns.get_level_values(0):
        scaling = reco_df.weights_mc.values.ravel()
    else:
        scaling = np.ones(reco_df.shape[0])
    for col in reco_df.columns:
        if "morph" in col:
            unisim_col.append(tuple(filter(None, col)))
        elif "ps1" in col:
            multisig_col.append(tuple(filter(None, col)))
        elif "univ" in "".join(list(col)):
            base = tuple(filter(None, col))[:univ_level]
            if base not in multisim_col:
                multisim_col.append(base)

    nbins    = len(bins)
    n_out    = nbins - 1
    reco_idx = digitize_with_overflow(reco_df[reco_var], bins)
    cv       = np.bincount(reco_idx, weights=scaling, minlength=n_out).astype(float)
    syst_dict = {}

    # Pre-compute xsec digitized indices — constant across all xsec knobs
    _xsec = None
    if (xsec_inputs is not None
            and xsec_inputs.true_signal_df is not None
            and xsec_inputs.reco_var_true is not None
            and xsec_inputs.true_var_true is not None):
        _sig_mask      = reco_df.signal == 0
        _reco_sig      = reco_df[_sig_mask]
        _sig_reco_idx  = digitize_with_overflow(_reco_sig[reco_var], bins)
        _sig_true_idx  = digitize_with_overflow(_reco_sig[xsec_inputs.reco_var_true], bins)
        _smear_flat_idx = _sig_reco_idx * n_out + _sig_true_idx
        _truth_sig_idx = digitize_with_overflow(xsec_inputs.true_signal_df[xsec_inputs.true_var_true], bins)
        _bkg_reco_idx  = digitize_with_overflow(reco_df[~_sig_mask][reco_var], bins)
        _sig_hist_cv   = get_hist1d(
            np.ones(len(xsec_inputs.true_signal_df)) * xsec_inputs.true_signal_scale,
            xsec_inputs.true_signal_df[xsec_inputs.true_var_true], bins,
        )
        _xsec = dict(sig_mask=_sig_mask, smear_flat_idx=_smear_flat_idx,
                     truth_sig_idx=_truth_sig_idx, bkg_reco_idx=_bkg_reco_idx,
                     sig_hist_cv=_sig_hist_cv)
    
    # unisim
    if len(unisim_col)>0:
        for col in tqdm(unisim_col, desc='Running through unisims'):
            weights = reco_df[col].values.astype(np.float64)
            weights[np.isnan(weights)] = 1.0
            weights[(weights>10) | (weights < 0)] = 1.0 
            weights *= scaling

            response = None
            if is_xsec(col, xsec_inputs):
                true_signal_weights = xsec_inputs.true_signal_df[col[2:]].values.astype(np.float64) * xsec_inputs.true_signal_scale
                w_sig = weights[_xsec['sig_mask']].reshape(-1, 1)
                w_bkg = weights[~_xsec['sig_mask']].reshape(-1, 1)
                result = _get_xsec_hists_inner(
                    _xsec['smear_flat_idx'], w_sig, _xsec['truth_sig_idx'],
                    true_signal_weights.reshape(-1, 1),
                    _xsec['sig_hist_cv'], _xsec['bkg_reco_idx'], w_bkg, n_out,
                    return_response=save_response,
                )
                hists, response = result if save_response else (result, None)
            else:
                hists = np.bincount(reco_idx, weights=weights, minlength=n_out).reshape(n_out, 1)

            entry = {'hists': hists}
            if response is not None:
                entry['response'] = response
            syst_dict[col[2]] = entry

    # multisig (ps1/ms1 pairs)
    if len(multisig_col)>0:
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
                w_sig = weights[_xsec['sig_mask']]
                w_bkg = weights[~_xsec['sig_mask']]
                result = _get_xsec_hists_inner(
                    _xsec['smear_flat_idx'], w_sig, _xsec['truth_sig_idx'], true_signal_weights,
                    _xsec['sig_hist_cv'], _xsec['bkg_reco_idx'], w_bkg, n_out,
                    return_response=save_response,
                )
                hists, response = result if save_response else (result, None)
            else:
                hists = np.array([
                    np.bincount(reco_idx, weights=weights[:, u], minlength=n_out)
                    for u in range(weights.shape[1])
                ]).T

            entry = {'hists': hists}
            if response is not None:
                entry['response'] = response
            syst_dict[col[2]] = entry

    # multisim
    if len(multisim_col)>0:
        for col in tqdm(multisim_col, desc='Running through multisims'):
            weights = reco_df[col].values.astype(np.float64)
            weights[np.isnan(weights)] = 1.0
            weights[(weights>10) | (weights<0)] = 1
            weights *= scaling[:, np.newaxis]

            response = None
            if is_xsec(col, xsec_inputs):
                true_signal_weights = xsec_inputs.true_signal_df[col[2:]].values.astype(np.float64) * xsec_inputs.true_signal_scale
                w_sig = weights[_xsec['sig_mask']]
                w_bkg = weights[~_xsec['sig_mask']]
                result = _get_xsec_hists_inner(
                    _xsec['smear_flat_idx'], w_sig, _xsec['truth_sig_idx'], true_signal_weights,
                    _xsec['sig_hist_cv'], _xsec['bkg_reco_idx'], w_bkg, n_out,
                    return_response=save_response,
                )
                hists, response = result if save_response else (result, None)
            else:
                hists = np.array([
                    np.bincount(reco_idx, weights=weights[:, u], minlength=n_out)
                    for u in range(weights.shape[1])
                ]).T

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

def mcstat(indf, nuniv: int = 100,
           cols: list | None = None) -> pd.DataFrame:
    """Add MC statistical uncertainty universes to the DataFrame.

    Generates Poisson-fluctuated weights for each event based on unique seeds
    derived from specified identifier columns, producing ``nuniv`` independent
    universe weight columns.

    Heavily inspired by Mun's method in:
    ``cafpyana/analysis_village/1mu1p0pi/wienersvd_unfolding.ipynb``

    Parameters
    ----------
    indf : pd.DataFrame
        Input DataFrame. Must contain the columns (or index names) listed in
        ``cols`` for seed generation.
    nuniv : int, optional
        Number of MC statistical universes to create (default 100).
    cols : list of str, optional
        Column names (at level 0) or index names used to build a unique
        per-event seed.  Defaults to
        ``['__ntuple', 'entry', 'rec.slc..index', 'run', 'subrun', 'evt', 'sample']``.

    Returns
    -------
    pd.DataFrame
        ``indf`` with ``nuniv`` MC-stat weight columns appended under the
        ``('slc', 'truth', 'MCstat', 'univ_i', '', '')`` MultiIndex hierarchy.

    Notes
    -----
    Each event is seeded from ``hash(event_identifiers) % 2**32``; all
    ``nuniv`` Poisson(1.0) draws for that event are generated from a single
    ``default_rng`` call, giving O(n_events) RNG constructions instead of
    O(n_events × nuniv).
    """
    if cols is None:
        cols = ['__ntuple', 'entry', 'rec.slc..index', 'run', 'subrun', 'evt', 'sample']

    top_level = set(indf.columns.get_level_values(0)) | set(indf.index.names)
    for col in cols:
        if col not in top_level:
            raise ValueError(f"Column '{col}' not found in DataFrame columns or index.")

    seed_cols = [tuple([col] + [""] * (len(indf.columns[0]) - 1)) for col in cols]
    unique_seeds = (
        indf.reset_index()[seed_cols]
        .apply(lambda x: hash(tuple(x)) % 2**32, axis=1)
        .to_numpy()
    )
    n_events = len(unique_seeds)

    # One RNG per event; draw all nuniv universes at once (~nuniv× fewer objects).
    out = np.empty((n_events, nuniv), dtype=np.float32)
    for i, seed in enumerate(tqdm(unique_seeds, desc='MCstat universes')):
        out[i] = np.random.default_rng(int(seed)).poisson(1.0, size=nuniv)

    mcstat_univ_cols = pd.MultiIndex.from_product(
        [['slc'], ['truth'], ['MCstat'], [f"univ_{i}" for i in range(nuniv)], [''], ['']]
    )
    return indf.join(pd.DataFrame(out, index=indf.index, columns=mcstat_univ_cols))


def get_detvar_systs(detvar_dict, var, bins,
                     event_type: str | None = "all",
                     cuts=None,
                     **select_kwargs):
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
    cuts : list of CutSpec, optional
        Custom cut sequence forwarded to :func:`~nueana.selection.select`.
        Defaults to ``DEFAULT_CUTS`` when None. Build custom lists with
        :func:`~nueana.selection.modify_cut`, :func:`~nueana.selection.drop_cuts`,
        or :class:`~nueana.selection.CutSpec`.
    **select_kwargs
        Additional keyword arguments forwarded to :func:`~nueana.selection.select`
        (e.g. ``stage``, ``spring``, ``shower_scale``).

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
    _needs_select = cuts is not None or bool(select_kwargs)
    if _needs_select: print("Applying selection to detector variation samples...")
    _sel_kw = dict(savedict=False, cuts=cuts, **select_kwargs)

    matrices_dict = {}
    for i, key in tqdm(enumerate(detvar_dict.keys())):
        this_dict = detvar_dict[key]
        this_dv   = this_dict['dv_df']
        this_cv   = this_dict['cv_df']
        this_norm = this_dict['pot']

        def _ensure_signal(df):
            """Add the signal column via define_signal if it is missing."""
            if event_type in ("signal", "background") and "signal" not in df.columns:
                df = define_signal(df, prefix=('slc', 'truth'))
            return df

        cv_sel = select(this_cv, **_sel_kw) if _needs_select else this_cv
        cv_sel = apply_event_mask(_ensure_signal(ensure_lexsorted(cv_sel, axis=1)), event_type)
        cv_hist = get_hist1d(data=cv_sel[var], bins=bins) / this_norm

        dv_dfs = this_dv if isinstance(this_dv, list) else [this_dv]
        if _needs_select:
            dv_dfs = [select(dv, **_sel_kw) for dv in dv_dfs]
        dv_hists = np.column_stack([
            get_hist1d(data=apply_event_mask(_ensure_signal(ensure_lexsorted(dv, axis=1)), event_type)[var], bins=bins)
            for dv in dv_dfs
        ]) / this_norm

        cov, cov_frac, corr = calc_matrices(var_arr=dv_hists, cv=cv_hist)
        out_key = key if key.startswith("DetVar_") else f"DetVar_{key}"
        matrices_dict[out_key] = {
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
    ("calorimetry", ["ccal", "phi", "alpha", "beta90", "beta_90", "betap90","Ecorr",'yz','calo']),
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
    "DetVar": lambda key: "_".join(key.split("_")[1:]),
    "Geant4": lambda key: key.split("_")[1],
}


def _classify_category(key: str) -> str | None:
    """Map a raw systematic key to its high-level category, or None if unknown."""
    if any(alias in key for alias in _GENIE_ALIASES):
        return "GENIE"
    return next((cat for cat in _CATEGORY_KEYWORDS if cat in key), None)


def key_in_allowed(key: str, allowed_keys) -> bool:
    """Return True if a systematic key's category appears in allowed_keys.

    Parameters
    ----------
    key : str
        Raw systematic key string (as it appears in syst_dict).
    allowed_keys : sequence of str or None
        Category names to include (e.g. ``('GENIE', 'MCstat')``).
        GENIE aliases (SBNNuSyst, SuSAv2) are handled correctly.
        None means all keys are allowed.

    Returns
    -------
    bool
    """
    if allowed_keys is None:
        return True
    return _classify_category(key) in allowed_keys


def _classify_detvar_subcategory(detvar_key: str) -> str:
    """Map a detector variation key to its analysis subcategory."""
    key    = detvar_key.lower()
    tokens = key.replace("-", "_").split("_")
    for subcategory, keywords in _DETVAR_SUBCATEGORIES:
        if any(kw.lower() in key for kw in keywords):
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
        Columns: ``key``, ``category``, ``subcategory``, ``unc_diag``, ``unc_diag_avg``, ``unc_norm``, ``top5``.
        ``unc_diag``     — per-bin fractional uncertainty: sqrt(diag(cov)) / cv.
        ``unc_diag_avg`` — mean of ``unc_diag``; summary of per-bin shape uncertainty.
        ``unc_norm``     — normalization fraction: sqrt(sum_ij cov[i,j]) / sum(cv).
        Sorted by category then ``unc_norm`` (descending).
        ``top5`` flags the five largest sources per category.
    """
    records = []

    N_tot  = float(np.sum(cv_hist))
    for d in dicts:
        for raw_key in d:
            cov      = d[raw_key]['cov']
            unc_diag = np.sqrt(np.diag(cov)) / cv_hist

            cov_sum = float(np.sum(cov))
            if cov_sum < 0:
                cov_sum = 0.0
            unc_norm = float(np.sqrt(cov_sum) / N_tot) if N_tot > 0 else 0.0

            category = _classify_category(raw_key)
            if category is None:
                print(f"Warning: category not found for key '{raw_key}'")
                records.append({
                    "key": raw_key, "category": "Other", "subcategory": "Other",
                    "unc_diag": unc_diag, "unc_diag_avg": float(np.mean(unc_diag)), "unc_norm": unc_norm,
                })
                continue

            extracted_key = _KEY_EXTRACTORS[category](raw_key)
            if category == "GENIE" and any(alias in raw_key for alias in _GENIE_ALIASES):
                extracted_key += "+"

            subcategory = _classify_detvar_subcategory(extracted_key) if category == "DetVar" else category
            records.append({
                "key": extracted_key, "category": category, "subcategory": subcategory,
                "unc_diag": unc_diag, "unc_diag_avg": float(np.mean(unc_diag)), "unc_norm": unc_norm,
            })

    syst_df = pd.DataFrame(records).sort_values(['category', 'unc_norm'], ascending=[False, False])
    syst_df['top5'] = syst_df.groupby('category')['unc_norm'].rank(method='first', ascending=False) <= 5
    return syst_df

def make_multiverse_weights(evtdf, knob_list, n_univs=100, evt_prefix=None, nudf=None,
                            drop_originals=False):
    """Expand unisim/multisigma knobs into multisim-style universe weight columns.

    For each knob in ``knob_list``, converts morph (unisim) or ps1/ms1 (multisigma)
    weights into ``n_univs`` Gaussian-sampled universe columns. When ``nudf`` is
    provided, the same universes are generated for both DataFrames and the resulting
    ``evtdf`` universe weights are overwritten with the values from ``nudf`` for
    events present in both (synchronization). When ``nudf`` is omitted, only
    ``evtdf`` is processed and synchronization is skipped.

    Parameters
    ----------
    evtdf : pd.DataFrame
        Event-level DataFrame with CAF-style MultiIndex columns.
    knob_list : list of str
        Systematic knob names to expand. Each must exist in both ``evtdf``
        (under ``evt_prefix`` if provided) and ``nudf`` (if provided).
    n_univs : int, optional
        Number of universe columns to generate per knob (default 100).
    evt_prefix : tuple, optional
        Column-level prefix to prepend to knob names when accessing ``evtdf``
        (e.g. ``('slc', 'truth')``). If None, knob names are used directly.
    nudf : pd.DataFrame, optional
        Neutrino-level DataFrame. When provided, universe weights are generated
        for ``nudf`` as well and synchronized into ``evtdf``. Must share index
        names with ``evtdf``.
    drop_originals : bool, default False
        If True, drop the original morph/multisigma columns for each converted
        knob from the output DataFrame(s). Columns are only dropped for knobs
        that were successfully expanded (i.e. detected as morph or multisigma).

    Returns
    -------
    evtdf : pd.DataFrame
        Updated event-level DataFrame with new universe columns appended.
        Returned alone when ``nudf`` is None.
    nudf, evtdf : tuple of pd.DataFrame
        Both DataFrames with new universe columns, after synchronization.
        Returned when ``nudf`` is provided.
    """
    if nudf is not None and nudf.index.names != evtdf.index.names:
        raise ValueError("Index names of nudf and evtdf must match.")

    def _draws(knob, df_idx):
        """Return (n_univs,) array of per-universe Gaussian draws with reproducible seeds.

        Uses hashlib (not Python's hash()) so seeds are identical across processes
        regardless of PYTHONHASHSEED — required for cross-notebook synchronization.
        """
        out = np.empty(n_univs)
        for i in range(n_univs):
            digest = hashlib.md5(f"{knob}{i}{df_idx}".encode()).hexdigest()
            np.random.seed(int(digest, 16) % 2**32)
            out[i] = np.random.normal(0, 1)
        return out

    # --- Pre-scan: identify which knobs expand so we can pre-allocate ---
    # This avoids accumulating n_knobs × n_univs separate arrays in a dict,
    # which would hold 2–3× the final new-column footprint simultaneously.
    evtdf_active = []  # (knob, evtdf_knob_key, evtdf_knob_cols, kind)
    nudf_active  = []  # (knob, nudf_knob_cols, kind)
    evtdf_cols_to_drop = []
    nudf_cols_to_drop  = []

    for knob in knob_list:
        evtdf_knob_key  = (evt_prefix + (knob,)) if evt_prefix else knob
        evtdf_knob_cols = evtdf[evtdf_knob_key].columns
        if len(evtdf_knob_cols) == 1:
            evtdf_active.append((knob, evtdf_knob_key, evtdf_knob_cols, 'morph'))
            if drop_originals:
                key_t = evtdf_knob_key if isinstance(evtdf_knob_key, tuple) else (evtdf_knob_key,)
                evtdf_cols_to_drop.extend(key_t + col for col in evtdf_knob_cols)
        elif len(evtdf_knob_cols) == 7:
            evtdf_active.append((knob, evtdf_knob_key, evtdf_knob_cols, 'multisigma'))
            if drop_originals:
                key_t = evtdf_knob_key if isinstance(evtdf_knob_key, tuple) else (evtdf_knob_key,)
                evtdf_cols_to_drop.extend(key_t + col for col in evtdf_knob_cols)

        if nudf is not None:
            nudf_knob_cols = nudf[knob].columns
            if len(nudf_knob_cols) == 1:
                nudf_active.append((knob, nudf_knob_cols, 'morph'))
                if drop_originals:
                    nudf_cols_to_drop.extend((knob,) + col for col in nudf_knob_cols)
            elif len(nudf_knob_cols) == 7:
                nudf_active.append((knob, nudf_knob_cols, 'multisigma'))
                if drop_originals:
                    nudf_cols_to_drop.extend((knob,) + col for col in nudf_knob_cols)

    # --- Pre-allocate output arrays; fill per-knob slice then del wgts immediately ---
    evtdf_new_arr  = np.empty((len(evtdf), len(evtdf_active) * n_univs), dtype=np.float32)
    evtdf_new_cols = []

    for j, (knob, evtdf_knob_key, _cols, kind) in enumerate(tqdm(evtdf_active)):
        base  = (evtdf[evtdf_knob_key].morph.values if kind == 'morph'
                 else evtdf[evtdf_knob_key].ps1.values)
        draws = _draws(knob, 1)
        if kind == 'morph':
            wgts = np.clip(1 + (base[:, None] - 1) * 2 * np.abs(draws), 0, 10)
        else:
            wgts = np.clip(1 + (base[:, None] - 1) * draws, 0, 10)
        evtdf_new_arr[:, j * n_univs:(j + 1) * n_univs] = wgts
        del wgts
        evtdf_new_cols.extend(evtdf_knob_key + (f"univ_{i}",) for i in range(n_univs))

    nudf_univ_keys = {}
    nudf_new_arr   = (np.empty((len(nudf), len(nudf_active) * n_univs), dtype=np.float32)
                      if nudf is not None and nudf_active else None)
    nudf_new_cols  = []

    if nudf is not None and nudf_new_arr is not None:
        for j, (knob, nudf_knob_cols, kind) in enumerate(nudf_active):
            base  = (nudf[knob].morph.values if kind == 'morph'
                     else nudf[knob].ps1.values)
            draws = _draws(knob, 0)
            if kind == 'morph':
                wgts = np.clip(1 + (base[:, None] - 1) * 2 * np.abs(draws), 0, 10)
            else:
                wgts = np.clip(1 + (base[:, None] - 1) * draws, 0, 10)
            nudf_new_arr[:, j * n_univs:(j + 1) * n_univs] = wgts
            del wgts
            keys = [(knob, f"univ_{i}") for i in range(n_univs)]
            nudf_new_cols.extend(keys)
            nudf_univ_keys[knob] = keys

    # --- Build and concat new evtdf columns ---
    evtdf_nlevels = evtdf.columns.nlevels
    evtdf_padded  = [col + ("",) * (evtdf_nlevels - len(col)) for col in evtdf_new_cols]
    evtdf_new_df  = pd.DataFrame(
        evtdf_new_arr, index=evtdf.index,
        columns=pd.MultiIndex.from_tuples(evtdf_padded, names=evtdf.columns.names),
    )
    del evtdf_new_arr
    evtdf = pd.concat([evtdf, evtdf_new_df], axis=1)
    del evtdf_new_df

    if nudf is None:
        if drop_originals and evtdf_cols_to_drop:
            evtdf = evtdf.drop(columns=evtdf_cols_to_drop)
        return evtdf

    # --- Build and concat new nudf columns ---
    nudf_nlevels = nudf.columns.nlevels
    nudf_padded  = [col + ("",) * (nudf_nlevels - len(col)) for col in nudf_new_cols]
    if nudf_padded:
        nudf_new_df = pd.DataFrame(
            nudf_new_arr, index=nudf.index,
            columns=pd.MultiIndex.from_tuples(nudf_padded, names=nudf.columns.names),
        )
        del nudf_new_arr
        nudf = pd.concat([nudf, nudf_new_df], axis=1)
        del nudf_new_df

    # --- Synchronize nudf universe weights into evtdf ---
    nudf_nlevels  = nudf.columns.nlevels
    evtdf_nlevels = evtdf.columns.nlevels
    nudf_in_evtdf = nudf.index.isin(evtdf.index)

    for knob in knob_list:
        if "multisim" in knob or knob not in nudf_univ_keys:
            continue
        raw_nu_cols     = nudf_univ_keys[knob]
        nudf_univ_cols  = [col + ("",) * (nudf_nlevels  - len(col)) for col in raw_nu_cols]
        raw_evt_cols    = [(evt_prefix + col) if evt_prefix else col for col in raw_nu_cols]
        evtdf_univ_cols = [col + ("",) * (evtdf_nlevels - len(col)) for col in raw_evt_cols]
        synced_vals = nudf.loc[nudf_in_evtdf, nudf_univ_cols]
        if synced_vals.isna().any().any():
            print(f"Found NaN values in synced_vals for knob: {knob}")
        evtdf[evtdf_univ_cols] = synced_vals

    if drop_originals:
        if evtdf_cols_to_drop:
            evtdf = evtdf.drop(columns=evtdf_cols_to_drop)
        if nudf_cols_to_drop:
            nudf = nudf.drop(columns=nudf_cols_to_drop)
    return nudf, evtdf


def slim_multisim_weights(
    df: pd.DataFrame,
    categories: list[str] | None = None,
    n_univs: int = 100,
) -> pd.DataFrame:
    """Combine per-knob multisim universe weights into one slim set per category.

    For each requested category, multiplies the per-universe weights of all multisim
    knobs in that category together universe-index-by-universe-index, yielding one
    combined set of ``n_univs`` universes. This mirrors the ``slim`` option in
    cafpyana's ``getsyst.py`` and reduces column count for downstream covariance
    calculations while preserving inter-knob correlations.

    Only multisim columns (those with ``univ_*`` sub-columns) are combined.
    Unisim (morph) and multisigma (ps1/ms1) knobs are not affected.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with CAF-style MultiIndex columns containing systematic weight
        columns produced by cafpyana's getsyst.
    categories : list of str, optional
        Categories to slim. Defaults to ``['GENIE', 'Flux', 'Geant4']``.
    n_univs : int, optional
        Number of universes per slim category (default 100). If a knob has fewer
        universes than ``n_univs``, the remaining slim universes are left at 1.0
        (no contribution from that knob).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with slim columns appended under the keys
        ``('slc', 'truth', '<Category>_slim', 'univ_i', '', ...)``.
        Individual knob columns are not removed.

    Examples
    --------
    >>> df_slim = slim_multisim_weights(df, categories=['GENIE', 'Flux', 'Geant4'])
    >>> # df_slim now has ('slc','truth','GENIE_slim','univ_0',...) etc. appended
    >>> # Pass to get_syst as normal — category classification still works
    >>> syst_dict = get_syst(df_slim, reco_var, bins)
    """
    if categories is None:
        categories = ['GENIE', 'Flux', 'Geant4']

    df = ensure_lexsorted(df, axis=1)
    n_levels = df.columns.nlevels

    # Detect which level holds the 'univ_*' token
    univ_level = -1
    for col in df.columns:
        for i, x in enumerate(col):
            if str(x).startswith("univ"):
                univ_level = i
                break
        if univ_level >= 0:
            break

    if univ_level < 0:
        return df

    # Collect unique multisim bases using the same logic as get_syst_hists
    seen: set = set()
    multisim_bases: list = []
    for col in df.columns:
        if "univ" in "".join(str(c) for c in col):
            base = tuple(filter(None, col))[:univ_level]
            if base not in seen:
                seen.add(base)
                multisim_bases.append(base)

    # Group bases by requested category
    cat_bases: dict[str, list] = {cat: [] for cat in categories}
    for base in multisim_bases:
        knob = base[2]
        cat = _classify_category(knob)
        if cat in cat_bases:
            cat_bases[cat].append(base)

    new_cols: dict = {}
    pad = ("",) * (n_levels - 4)

    for cat, bases in cat_bases.items():
        if not bases:
            continue

        slim_weights = np.ones((len(df), n_univs), dtype=np.float64)
        for base in bases:
            w = df[base].values.astype(np.float64)
            w = np.nan_to_num(w, nan=1.0)
            np.clip(w, 0, 10, out=w)
            n = min(n_univs, w.shape[1])
            slim_weights[:, :n] *= w[:, :n]

        slim_key = ("slc", "truth", f"{cat}_slim")
        for i in range(n_univs):
            new_cols[slim_key + (f"univ_{i}",) + pad] = slim_weights[:, i]

    if not new_cols:
        return df

    # Drop all original multisim columns that were folded into a slim category
    slimmed_bases = {base for cat, bases in cat_bases.items() for base in bases}
    cols_to_drop = [
        col for col in df.columns
        if tuple(filter(None, col))[:univ_level] in slimmed_bases
    ]
    df = df.drop(columns=cols_to_drop)

    new_df = pd.DataFrame(new_cols, index=df.index)
    new_df.columns = pd.MultiIndex.from_tuples(list(new_cols.keys()), names=df.columns.names)
    return pd.concat([df, new_df], axis=1)