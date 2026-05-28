from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import replace

from .utils import ensure_lexsorted, apply_event_mask, flux_pot_weights
from .io import load_dfs
from .selection import select
from .selection import select_sideband
from .utils import get_hist1d
from .syst import calc_matrices, get_syst, get_syst_df, get_detvar_systs
from .detvar import load_detvar_dict
from .classes import SystematicsOutput, XSecInputs
from .analysis import integrated_flux, signal_dict, POT_NORM_UNC, NTARGETS_UNC
from .preprocess import preprocess_mc, add_pi0
from . import config

__all__ = [
    'get_corr_from_cov',
    'get_fractional_covariance',
    'add_uncertainty',
    'add_fractional_uncertainty',
    'get_intime_cov',
    'get_total_cov',
    'load_detvar_dicts',
]


def get_corr_from_cov(cov):
    sigma = np.sqrt(np.diag(cov))
    denom = np.outer(sigma, sigma)

    corr = np.divide(
        cov,
        denom,
        out=np.zeros_like(cov, dtype=float),
        where=denom > 0
    )

    np.fill_diagonal(corr, 1.0)
    return corr

def get_fractional_covariance(cov, rate_hist_cv):
    rate_hist_cv = np.asarray(rate_hist_cv)
    denom = np.outer(rate_hist_cv, rate_hist_cv)

    frac_cov = np.divide(
        cov,
        denom,
        out=np.zeros_like(cov, dtype=float),
        where=denom > 0
    )
    return frac_cov


def _sum_covariances_from_dicts(syst_dicts, n_bins):
    total_cov = np.zeros((n_bins, n_bins))
    for syst_dict in syst_dicts:
        for entry in syst_dict.values():
            total_cov += entry["cov"]
    return total_cov


def _collect_rate_systs(indf, reco_var, bins, mcbnb_pot, rate_hist_cv):
    syst_dict = get_syst(reco_df=indf, reco_var=reco_var, bins=bins, mcbnb_pot=mcbnb_pot)
    total_cov = _sum_covariances_from_dicts([syst_dict], rate_hist_cv.size)
    return syst_dict, total_cov, get_syst_df([syst_dict], rate_hist_cv)


def _collect_xsec_systs(indf, reco_var, bins, mcbnb_pot, xsec_hist_cv, xsec_inputs):
    syst_dict = get_syst(reco_df=indf, reco_var=reco_var, bins=bins, mcbnb_pot=mcbnb_pot, xsec_inputs=xsec_inputs)
    total_cov = _sum_covariances_from_dicts([syst_dict], xsec_hist_cv.size)
    return syst_dict, total_cov, get_syst_df([syst_dict], xsec_hist_cv)


def _collect_detvar_systs(detvar_dict, reco_var, bins, event_type, cuts, select_kwargs, rate_hist_cv, xsec_hist_cv=None):
    syst_dict = get_detvar_systs(detvar_dict, reco_var, bins, event_type=event_type, cuts=cuts, **select_kwargs)
    total_cov = _sum_covariances_from_dicts([syst_dict], rate_hist_cv.size)
    rate_df = get_syst_df([syst_dict], rate_hist_cv)
    xsec_df = get_syst_df([syst_dict], xsec_hist_cv) if xsec_hist_cv is not None else None
    return syst_dict, total_cov, rate_df, xsec_df


def load_detvar_dicts(detvar_files=None):
    """Load and combine detector variation dictionaries from HDF5 files.

    Load this once per session and pass the result directly to
    :func:`get_total_cov` via its ``detvar_dict`` parameter to avoid
    re-loading on every call (which can take minutes).

    Parameters
    ----------
    detvar_files : list of str, optional
        Paths to detvar HDF5 files. Defaults to ``config.DETVAR_DICT_FILES``.

    Returns
    -------
    dict
        Combined detector variation dictionary.
    """
    if detvar_files is None:
        detvar_files = config.DETVAR_DICT_FILES

    combined_dict = {}
    for detvar_file in detvar_files:
        combined_dict.update(load_detvar_dict(detvar_file))

    return combined_dict


def add_uncertainty(
    result: SystematicsOutput,
    cov: np.ndarray,
    key: str,
    category: str | None = None,
    target: str = "both",
    unc: np.ndarray | None = None,
    hists: np.ndarray | None = None,
    sum_value: float | None = None,
    top5: bool = False,
):
    """
    Add a user-defined covariance contribution to a SystematicsOutput.

    Parameters
    ----------
    result
        Existing systematics result object.
    cov
        Covariance matrix contribution to add. Must match (nbins, nbins).
    key
        Dictionary/dataframe key label for the new source.
    category
        Category label for the dataframe entry. Defaults to `key`.
    target
        Where to apply this uncertainty: "rate", "xsec", or "both".
    unc
        Optional per-bin fractional uncertainty array (``unc_diag`` column).
        Defaults to sqrt(diag(cov))/hist_cv.
    hists
        Optional universe histogram array stored in the systematic dictionary.
        Shape must be (nbins, nuniverses) or (nbins,).
    sum_value
        Optional normalization fraction (``unc_norm`` column).
        Defaults to sqrt(sum_ij cov[i,j]) / sum(hist_cv).
    top5
        Value for the `top5` column in the added row.
    """
    if not key:
        raise ValueError("key must be a non-empty string")
    if target not in {"rate", "xsec", "both"}:
        raise ValueError("target must be one of: 'rate', 'xsec', 'both'")
    if category is None:
        category = key

    rate_hist_cv = np.asarray(result.rate_hist_cv, dtype=float)
    cov = np.asarray(cov, dtype=float)
    if cov.shape != (rate_hist_cv.size, rate_hist_cv.size):
        raise ValueError(
            f"cov shape {cov.shape} does not match expected {(rate_hist_cv.size, rate_hist_cv.size)}"
        )

    if target in {"xsec", "both"} and not result.has_xsec:
        raise ValueError("xsec covariance is not available in this SystematicsOutput")

    if unc is None:
        unc = np.divide(np.sqrt(np.diag(cov)), rate_hist_cv, out=np.zeros_like(rate_hist_cv, dtype=float), where=rate_hist_cv > 0)
    else:
        unc = np.asarray(unc, dtype=float)
        if unc.shape != rate_hist_cv.shape:
            raise ValueError(f"unc shape {unc.shape} does not match hist_cv shape {rate_hist_cv.shape}")

    if sum_value is None:
        cov_sum = max(0.0, float(np.sum(cov)))
        n_tot   = float(np.sum(rate_hist_cv))
        sum_value = float(np.sqrt(cov_sum) / n_tot) if n_tot > 0 else 0.0

    if hists is not None:
        hists = np.asarray(hists, dtype=float)
        if hists.ndim == 1:
            hists = hists.reshape(-1, 1)
        if hists.ndim != 2 or hists.shape[0] != rate_hist_cv.size:
            raise ValueError(
                f"hists must have shape (nbins, nuniverses); got {hists.shape} for nbins={rate_hist_cv.size}"
            )

    syst_row = pd.DataFrame(
        {
            "key": [key],
            "category": [category],
            "unc_diag": [unc],
            "unc_diag_avg": [float(np.mean(unc))],
            "unc_norm": [sum_value],
            "top5": [top5],
        }
    )

    rate_syst_entry = {
        "cov": cov,
        "cov_frac": get_fractional_covariance(cov, rate_hist_cv),
        "corr": get_corr_from_cov(cov),
    }
    if hists is not None:
        rate_syst_entry["hists"] = hists

    updates = {}

    if target in {"rate", "both"}:
        updates["rate_cov"] = result.rate_cov + cov
        updates["rate_syst_df"] = pd.concat([result.rate_syst_df, syst_row], ignore_index=True)
        updates["rate_syst_dict"] = {**result.rate_syst_dict, key: rate_syst_entry}

    if target in {"xsec", "both"}:
        xsec_cv_arr = np.asarray(result.xsec_hist_cv, dtype=float)
        xsec_syst_entry = {
            "cov": cov,
            "cov_frac": get_fractional_covariance(cov, xsec_cv_arr),
            "corr": get_corr_from_cov(cov),
        }
        if hists is not None:
            xsec_syst_entry["hists"] = hists
        updates["xsec_cov"] = result.xsec_cov + cov
        updates["xsec_syst_df"] = pd.concat([result.xsec_syst_df, syst_row], ignore_index=True)
        updates["xsec_syst_dict"] = {**result.xsec_syst_dict, key: xsec_syst_entry}

    return replace(result, **updates)


def add_fractional_uncertainty(
    result: SystematicsOutput,
    frac_unc: float | np.ndarray,
    key: str,
    category: str | None = None,
    correlation: str = "fully_correlated",
):
    """
    Add a per-bin fractional uncertainty with configurable correlation.

    Parameters
    ----------
    result
        Existing systematics result object.
    frac_unc
        Fractional uncertainty: either a scalar (applied uniformly to all bins)
        or a per-bin array (e.g. [0.05, 0.2, 0.2, 0.2]).
    key
        Dictionary/dataframe key label for the new source.
    category
        Category label for the dataframe entry. Defaults to `key`.
    correlation
        Correlation model for bin-to-bin structure:
        - "fully_correlated": 100% correlated across bins (default).
        - "diagonal": uncorrelated between bins.
    """
    if category is None:
        category = key

    rate_hist_cv = np.asarray(result.rate_hist_cv, dtype=float)
    frac_unc = np.asarray(frac_unc, dtype=float)
    if frac_unc.ndim == 0:
        frac_unc = np.broadcast_to(frac_unc, rate_hist_cv.shape).copy()
    if frac_unc.shape != rate_hist_cv.shape:
        raise ValueError(
            f"frac_unc shape {frac_unc.shape} does not match hist_cv shape {rate_hist_cv.shape}"
        )
    if np.any(frac_unc < 0):
        raise ValueError("frac_unc entries must be non-negative")
    if correlation not in {"diagonal", "fully_correlated"}:
        raise ValueError("correlation must be one of: 'diagonal', 'fully_correlated'")

    sigma_rate = frac_unc * rate_hist_cv
    cov_rate = np.diag(sigma_rate ** 2) if correlation == "diagonal" else np.outer(sigma_rate, sigma_rate)
    result = add_uncertainty(
        result=result, cov=cov_rate, key=key, category=category,
        target="rate", unc=frac_unc, sum_value=float(np.mean(frac_unc)),
    )

    if result.has_xsec:
        xsec_hist_cv = np.asarray(result.xsec_hist_cv, dtype=float)
        sigma_xsec = frac_unc * xsec_hist_cv
        cov_xsec = np.diag(sigma_xsec ** 2) if correlation == "diagonal" else np.outer(sigma_xsec, sigma_xsec)
        result = add_uncertainty(
            result=result, cov=cov_xsec, key=key, category=category,
            target="xsec", unc=frac_unc, sum_value=float(np.mean(frac_unc)),
        )

    return result

def get_intime_cov(selected_df, var, bins,
                   mcbnb_ngen,
                   mcbnb_pot,
                   threshold=0.05,
                   event_type: str | None = "all",
                   select_region: str = "signal",
                   cuts=None,
                   **select_kwargs):
    mcint_dfs = load_dfs(config.INTIME_FILE, ['histgenevtdf', 'nuecc'])
    scale = mcbnb_ngen / mcint_dfs['histgenevtdf'].TotalGenEvents.sum()
    mcint_df = mcint_dfs['nuecc']
    mcint_df = preprocess_mc(mcint_df)
    mcint_df = add_pi0(mcint_df)

    if select_region == "signal":
        mcint_df = select(mcint_df, savedict=False, cuts=cuts, **select_kwargs)
    elif select_region == "control":
        mcint_df = select_sideband(mcint_df, savedict=False, cuts=cuts, **select_kwargs)
    else:
        mcint_df = select(mcint_df, savedict=False, cuts=cuts, **select_kwargs)

    selected_df = apply_event_mask(ensure_lexsorted(selected_df, axis=1), event_type)
    mcint_df = apply_event_mask(ensure_lexsorted(mcint_df, axis=1))

    selected_fpw = flux_pot_weights(selected_df, mcbnb_pot, integrated_flux)
    mcint_fpw    = np.full(len(mcint_df), scale / (integrated_flux * (mcbnb_pot / 1e6)))

    rate_hist_cv = get_hist1d(data=selected_df[var], bins=bins, weights=selected_fpw)

    offbeam_mask = selected_df.signal.values != signal_dict['offbeam']
    selected_no_offbeam_df = selected_df[offbeam_mask]
    rate_hist_cv_removed = get_hist1d(
        data=selected_no_offbeam_df[var],
        bins=bins,
        weights=selected_fpw[offbeam_mask],
    )

    int_hist = get_hist1d(data=mcint_df[var], bins=bins, weights=mcint_fpw)
    dv_hist = rate_hist_cv_removed + int_hist

    matrices = calc_matrices(dv_hist.reshape(len(bins) - 1, -1), rate_hist_cv)
    cov = matrices[0]
    unc = np.divide(
        np.sqrt(np.diag(cov)),
        rate_hist_cv,
        out=np.zeros_like(rate_hist_cv, dtype=float),
        where=rate_hist_cv > 0,
    )

    # bins above threshold keep their own uncertainty; others get a uniform floor
    large_unc = unc > threshold
    uniform_unc_val = np.max(unc[~large_unc]) if np.any(~large_unc) else np.max(unc)
    unc_final = np.where(large_unc, unc, uniform_unc_val)

    cov_final = np.outer(unc_final * rate_hist_cv, unc_final * rate_hist_cv)
    return cov_final
    
def get_total_cov(reco_df, reco_var, bins, mcbnb_pot,
                  cuts=None, projected_pot=1e20,
                  mcbnb_ngen: float | None = None,
                  intime_threshold: float = 0.05,
                  event_type: str | None = "all",
                  select_region: str = "signal",
                  uncertainty_keys: list[str] | tuple[str, ...] | set[str] | None = None,
                  xsec_inputs: XSecInputs | None = None,
                  detvar_dict: dict | None = None,
                  pot_norm_unc: float = POT_NORM_UNC,
                  ntargets_unc: float = NTARGETS_UNC,
                  **select_kwargs):
    """
    Get the total event-rate covariance matrix and systematic dataframe for a
    given variable. Optionally also compute the xsec covariance matrix and
    systematic dataframe when xsec_inputs are provided.

    The data statistical uncertainty is added as a separate "Datastat" entry in
    the returned event-rate dataframe ONLY, and in the xsec dataframe when requested.

    Parameters
    ----------
    reco_df : pd.DataFrame
        Reconstructed event data
    reco_var : str or tuple
        Variable to histogram
    bins : np.ndarray
        Bin edges
    mcbnb_pot : float
        Monte Carlo BNB POT (or the main sample to normalize to)
    cuts : list of CutSpec, optional
        Custom cut sequence forwarded to detector-variation selection.
        Defaults to ``DEFAULT_CUTS`` when None. Build with
        :func:`modify_cut`, :func:`drop_cuts`, or :class:`CutSpec`.
    **select_kwargs
        Additional keyword arguments forwarded to :func:`~nueana.selection.select`
        for detector-variation and in-time cosmic selection
        (e.g. ``stage``, ``spring``, ``shower_scale``).
    projected_pot : float, optional
        Projected POT for data statistics calculation
    mcbnb_ngen : float, optional
        Number of generated events for in-time calculation
    intime_threshold : float, optional
        Threshold for in-time uncertainty handling, default is 0.05 (5%)
    event_type : str or None, optional
        Event mask ('all', 'signal', 'background'), default is 'all'
    select_region : str, optional
        Which detector variation dictionary to use: 'signal' (default), 'control', or 'all'.
    uncertainty_keys : list[str] or tuple[str, ...] or set[str] or None, optional
        Which uncertainty blocks to include. Allowed keys are:
        'rate', 'xsec', 'detv', 'norm', 'cosmic'.
        If None, defaults to {'rate', 'detv', 'norm', 'cosmic'} and adds
        'xsec' only when xsec_inputs is provided.
    xsec_inputs : XSecInputs, optional
        Cross-section calculation inputs.
    detvar_dict : dict, optional
        Pre-loaded detector variation dictionary (from :func:`load_detvar_dicts`).
        Pass this when calling ``get_total_cov`` multiple times in a session to
        avoid reloading the pickle files on each call. If None and ``'detv'`` is
        in ``uncertainty_keys``, the dict is loaded automatically.
    pot_norm_unc : float, optional
        Fractional uncertainty on beam exposure (POT counting).
        Defaults to ``analysis.POT_NORM_UNC`` (2%).
    ntargets_unc : float, optional
        Fractional uncertainty on the number of Ar targets.
        Defaults to ``analysis.NTARGETS_UNC`` (1%).

    Returns
    -------
    SystematicsOutput
        Systematic uncertainties with rate (and optionally cross-section) covariances.

    Notes
    -----
    The combination order is:
    1) rate systematics
    2) xsec systematics (optional)
    3) detector-variation systematics
    4) flat normalization uncertainties
    5) in-time cosmic uncertainty (optional)
    """
    allowed_uncertainty_keys = {"rate", "xsec", "detv", "norm", "cosmic"}
    if uncertainty_keys is None:
        selected_uncertainty_keys = {"rate", "detv", "norm", "cosmic"}
        if xsec_inputs is not None:
            selected_uncertainty_keys.add("xsec")
    else:
        selected_uncertainty_keys = set(uncertainty_keys)
    invalid_keys = selected_uncertainty_keys - allowed_uncertainty_keys
    if invalid_keys:
        raise ValueError(
            f"uncertainty_keys contains invalid entries: {sorted(invalid_keys)}. "
            f"Allowed keys are: {sorted(allowed_uncertainty_keys)}"
        )

    include_rate = "rate" in selected_uncertainty_keys
    include_xsec = "xsec" in selected_uncertainty_keys
    include_detv = "detv" in selected_uncertainty_keys
    include_norm = "norm" in selected_uncertainty_keys
    include_cosmic = "cosmic" in selected_uncertainty_keys

    if include_xsec and xsec_inputs is None:
        raise ValueError("'xsec' requested in uncertainty_keys, but xsec_inputs is None")

    # Load detvar dict if needed
    select_region_map = {
        "signal": config.DETVAR_DICT_SIGNAL,
        "control": config.DETVAR_DICT_CONTROL,
        "all": config.DETVAR_DICT_FILES,
    }
    if select_region not in select_region_map:
        raise ValueError(f"select_region must be one of {list(select_region_map.keys())}, got '{select_region}'")
    if include_detv and detvar_dict is None:
        detvar_path = select_region_map[select_region]
        print(f"Loading detvar dictionary for region: {select_region}, located at: {detvar_path}")
        detvar_dict = load_detvar_dicts(detvar_path) if select_region == "all" else load_detvar_dict(detvar_path)
        print(f"  Loaded {len(detvar_dict)} detector variation entries")

    # CV histograms
    sorted_df = apply_event_mask(ensure_lexsorted(reco_df, axis=1), event_type)
    _fpw = flux_pot_weights(sorted_df, mcbnb_pot, integrated_flux)
    rate_hist_cv = get_hist1d(data=sorted_df[reco_var], weights=_fpw, bins=bins)
    signal_mask = sorted_df.signal == 0
    xsec_hist_cv = get_hist1d(data=sorted_df[signal_mask][reco_var], weights=_fpw[signal_mask], bins=bins)

    empty_syst_df = pd.DataFrame(columns=["key", "category", "unc_diag", "unc_diag_avg", "unc_norm", "top5"])
    n_bins = rate_hist_cv.size

    rate_syst_dict: dict = {}
    rate_total_cov = np.zeros((n_bins, n_bins))
    rate_syst_frames: list[pd.DataFrame] = []

    xsec_syst_dict: dict = {}
    xsec_total_cov = np.zeros((n_bins, n_bins))
    xsec_syst_frames: list[pd.DataFrame] = []

    if include_rate:
        d, c, df = _collect_rate_systs(sorted_df, reco_var, bins, mcbnb_pot, rate_hist_cv)
        rate_syst_dict.update(d); rate_total_cov += c; rate_syst_frames.append(df)

    if include_xsec:
        d, c, df = _collect_xsec_systs(sorted_df, reco_var, bins, mcbnb_pot, xsec_hist_cv, xsec_inputs)
        xsec_syst_dict.update(d); xsec_total_cov += c; xsec_syst_frames.append(df)

    if include_detv:
        d, c, rate_df, xsec_df = _collect_detvar_systs(
            detvar_dict, reco_var, bins, event_type, cuts, select_kwargs, rate_hist_cv,
            xsec_hist_cv=xsec_hist_cv if include_xsec else None,
        )
        rate_syst_dict.update(d); rate_total_cov += c; rate_syst_frames.append(rate_df)
        if include_xsec:
            xsec_syst_dict.update(d); xsec_total_cov += c; xsec_syst_frames.append(xsec_df)

    if include_rate:
        data_err = np.sqrt(
            get_hist1d(data=sorted_df[reco_var], weights=sorted_df.weights_mc, bins=bins)
            * (projected_pot / mcbnb_pot)
        )
        flux_scale = integrated_flux * (projected_pot / 1e6)
        data_unc = np.divide(data_err, flux_scale * rate_hist_cv,
                             out=np.zeros_like(data_err, dtype=float), where=rate_hist_cv > 0)
        _data_unc_norm = float(np.sqrt(np.sum(data_err**2))) / (flux_scale * float(np.sum(rate_hist_cv))) if np.sum(rate_hist_cv) > 0 else 0.0
        rate_syst_frames.append(pd.DataFrame(
            {'key': ['Datastat'], 'category': ['Datastat'], 'unc_diag': [data_unc],
             'unc_diag_avg': [float(np.mean(data_unc))], 'unc_norm': [_data_unc_norm], 'top5': [False]}
        ))
        if include_xsec:
            data_unc_xsec = np.divide(data_err, flux_scale * xsec_hist_cv,
                                      out=np.zeros_like(data_err, dtype=float), where=xsec_hist_cv > 0)
            _data_unc_norm_xsec = float(np.sqrt(np.sum(data_err**2))) / (flux_scale * float(np.sum(xsec_hist_cv))) if np.sum(xsec_hist_cv) > 0 else 0.0
            xsec_syst_frames.append(pd.DataFrame(
                {'key': ['Datastat'], 'category': ['Datastat'], 'unc_diag': [data_unc_xsec],
                 'unc_diag_avg': [float(np.mean(data_unc_xsec))], 'unc_norm': [_data_unc_norm_xsec], 'top5': [False]}
            ))

    rate_syst_df = pd.concat(rate_syst_frames, ignore_index=True) if rate_syst_frames else empty_syst_df.copy()
    xsec_syst_df = pd.concat(xsec_syst_frames, ignore_index=True) if xsec_syst_frames else empty_syst_df.copy()

    intime_cov = None
    if include_cosmic and mcbnb_ngen is not None:
        intime_cov = get_intime_cov(
            selected_df=sorted_df, var=reco_var, bins=bins,
            mcbnb_ngen=mcbnb_ngen, mcbnb_pot=mcbnb_pot, threshold=intime_threshold,
            event_type=event_type, select_region=select_region, cuts=cuts, **select_kwargs,
        )

    result = SystematicsOutput(
        rate_hist_cv=rate_hist_cv,
        rate_cov=rate_total_cov,
        rate_syst_df=rate_syst_df,
        rate_syst_dict=rate_syst_dict,
        mcbnb_pot=mcbnb_pot,
        xsec_hist_cv=xsec_hist_cv if include_xsec else None,
        xsec_cov=xsec_total_cov if include_xsec else None,
        xsec_syst_df=xsec_syst_df if include_xsec else None,
        xsec_syst_dict=xsec_syst_dict if include_xsec else None,
    )

    if include_norm:
        result = add_fractional_uncertainty(result=result, frac_unc=pot_norm_unc,
                                            key="BeamExposure", category="BeamExposure")
        result = add_fractional_uncertainty(result=result, frac_unc=ntargets_unc,
                                            key="NTargets", category="NTargets")

    if include_cosmic and intime_cov is not None:
        rate_cv = np.asarray(result.rate_hist_cv, dtype=float)
        intime_unc = np.divide(np.sqrt(np.diag(intime_cov)), rate_cv,
                               out=np.zeros_like(rate_cv), where=rate_cv > 0)
        result = add_uncertainty(result=result, cov=np.asarray(intime_cov, dtype=float),
                                 key="Cosmic", category="Cosmic", target="rate",
                                 unc=intime_unc, sum_value=float(np.mean(intime_unc)))
        if result.has_xsec:
            xsec_cv = np.asarray(result.xsec_hist_cv, dtype=float)
            intime_unc_xsec = np.divide(np.sqrt(np.diag(intime_cov)), xsec_cv,
                                        out=np.zeros_like(xsec_cv), where=xsec_cv > 0)
            result = add_uncertainty(result=result, cov=np.asarray(intime_cov, dtype=float),
                                     key="Cosmic", category="Cosmic", target="xsec",
                                     unc=intime_unc_xsec,
                                     sum_value=float(np.mean(intime_unc_xsec)))

    return result