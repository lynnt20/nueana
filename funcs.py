import numpy as np
import pandas as pd
from dataclasses import replace

__all__ = [
    'get_corr_from_cov',
    'get_fractional_covariance',
    'add_uncertainty',
    'add_flat_norm_uncertainty',
    'add_fractional_uncertainty',
    'get_intime_cov',
    'get_total_cov',
    'load_detvar_dicts',
]
import warnings
import pickle
from tqdm import tqdm
from .utils import ensure_lexsorted, apply_event_mask
from .io import load_dfs
from .selection import select, select_sideband
from .histogram import get_hist1d, get_hist2d
from .syst import calc_matrices, get_syst, get_syst_df, get_detvar_systs
from .classes import SystematicsOutput, XSecInputs
from .constants import integrated_flux, signal_dict, POT_NORM_UNC, NTARGETS_UNC
from . import config

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

def get_fractional_covariance(cov, cv_hist):
    cv_hist = np.asarray(cv_hist)
    denom = np.outer(cv_hist, cv_hist)

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


def load_detvar_dicts(detvar_files=None):
    """Load and combine detector variation dictionaries from pickle files.

    Load this once per session and pass the result directly to
    :func:`get_total_cov` via its ``detvar_dict`` parameter to avoid
    re-loading on every call (which can take minutes).

    Parameters
    ----------
    detvar_files : list of str, optional
        Paths to detvar pickle files. Defaults to ``config.DETVAR_DICT_FILES``.

    Returns
    -------
    dict
        Combined detector variation dictionary.
    """
    if detvar_files is None:
        detvar_files = config.DETVAR_DICT_FILES

    combined_dict = {}
    for detvar_file in detvar_files:
        with open(detvar_file, 'rb') as f:
            combined_dict.update(pickle.load(f))

    return combined_dict


def _apply_norm_and_intime_uncertainties(
    result: SystematicsOutput,
    intime_cov: np.ndarray | None = None,
    include_norm: bool = True,
    include_cosmic: bool = True,
    pot_norm_unc: float = POT_NORM_UNC,
    ntargets_unc: float = NTARGETS_UNC,
):
    updated = result
    if include_norm:
        updated = add_flat_norm_uncertainty(
            result=updated,
            frac_unc=pot_norm_unc,
            key="BeamExposure",
            category="BeamExposure",
        )
        updated = add_flat_norm_uncertainty(
            result=updated,
            frac_unc=ntargets_unc,
            key="NTargets",
            category="NTargets",
        )

    if include_cosmic and intime_cov is not None:
        cv_hist = np.asarray(updated.hist_cv, dtype=float)
        intime_unc = np.divide(
            np.sqrt(np.diag(intime_cov)),
            cv_hist,
            out=np.zeros_like(cv_hist, dtype=float),
            where=cv_hist > 0,
        )
        updated = add_uncertainty(
            result=updated,
            cov=np.asarray(intime_cov, dtype=float),
            key="Cosmic",
            category="Cosmic",
            target="both" if updated.has_xsec else "rate",
            unc=intime_unc,
            sum_value=float(np.mean(intime_unc)),
        )

    return updated


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
        Optional per-bin fractional uncertainty array to store in the df.
        Defaults to sqrt(diag(cov))/hist_cv.
    hists
        Optional universe histogram array stored in the systematic dictionary.
        Shape must be (nbins, nuniverses) or (nbins,).
    sum_value
        Optional summary scalar for the df. Defaults to mean(unc).
    top5
        Value for the `top5` column in the added row.
    """
    if not key:
        raise ValueError("key must be a non-empty string")
    if target not in {"rate", "xsec", "both"}:
        raise ValueError("target must be one of: 'rate', 'xsec', 'both'")
    if category is None:
        category = key

    cv_hist = np.asarray(result.hist_cv, dtype=float)
    cov = np.asarray(cov, dtype=float)
    if cov.shape != (cv_hist.size, cv_hist.size):
        raise ValueError(
            f"cov shape {cov.shape} does not match expected {(cv_hist.size, cv_hist.size)}"
        )

    if target in {"xsec", "both"} and not result.has_xsec:
        raise ValueError("xsec covariance is not available in this SystematicsOutput")

    if unc is None:
        unc = np.divide(np.sqrt(np.diag(cov)), cv_hist, out=np.zeros_like(cv_hist, dtype=float), where=cv_hist > 0)
    else:
        unc = np.asarray(unc, dtype=float)
        if unc.shape != cv_hist.shape:
            raise ValueError(f"unc shape {unc.shape} does not match hist_cv shape {cv_hist.shape}")

    if sum_value is None:
        sum_value = float(np.mean(unc))

    if hists is not None:
        hists = np.asarray(hists, dtype=float)
        if hists.ndim == 1:
            hists = hists.reshape(-1, 1)
        if hists.ndim != 2 or hists.shape[0] != cv_hist.size:
            raise ValueError(
                f"hists must have shape (nbins, nuniverses); got {hists.shape} for nbins={cv_hist.size}"
            )

    syst_row = pd.DataFrame(
        {
            "key": [key],
            "category": [category],
            "unc": [unc],
            "sum": [sum_value],
            "top5": [top5],
        }
    )

    syst_entry = {
        "cov": cov,
        "cov_frac": get_fractional_covariance(cov, cv_hist),
        "corr": get_corr_from_cov(cov),
    }
    if hists is not None:
        syst_entry["hists"] = hists

    updates = {}

    if target in {"rate", "both"}:
        updates["rate_cov"] = result.rate_cov + cov
        updates["rate_syst_df"] = pd.concat([result.rate_syst_df, syst_row], ignore_index=True)
        updates["rate_syst_dict"] = {**result.rate_syst_dict, key: syst_entry}

    if target in {"xsec", "both"}:
        updates["xsec_cov"] = result.xsec_cov + cov
        updates["xsec_syst_df"] = pd.concat([result.xsec_syst_df, syst_row], ignore_index=True)
        updates["xsec_syst_dict"] = {**result.xsec_syst_dict, key: syst_entry}

    return replace(result, **updates)


def add_flat_norm_uncertainty(
    result: SystematicsOutput,
    frac_unc: float,
    key: str,
    category: str | None = None,
    correlation: str = "fully_correlated",
):
    """
    Add a fully correlated flat normalization uncertainty to a SystematicsOutput.

    Parameters
    ----------
    result
        Existing systematics result object.
    frac_unc
        Fractional uncertainty (e.g. 0.02 for 2%).
    key
        Dictionary/dataframe key label for the new source.
    category
        Category label for the dataframe entry. Defaults to `key`.
    correlation
        Correlation model for bin-to-bin structure:
        - "fully_correlated": 100% correlated across bins (default).
        - "diagonal": uncorrelated between bins.
    """
    if frac_unc < 0:
        raise ValueError("frac_unc must be non-negative")
    if category is None:
        category = key

    cv_hist = np.asarray(result.hist_cv, dtype=float)
    frac_unc_arr = np.full(cv_hist.shape, frac_unc, dtype=float)
    return add_fractional_uncertainty(
        result=result,
        frac_unc=frac_unc_arr,
        key=key,
        category=category,
        correlation=correlation,
    )


def add_fractional_uncertainty(
    result: SystematicsOutput,
    frac_unc: np.ndarray,
    key: str,
    category: str | None = None,
    correlation: str = "fully_correlated",
):
    """
    Add a per-bin fractional uncertainty array with configurable correlation.

    Parameters
    ----------
    result
        Existing systematics result object.
    frac_unc
        Per-bin fractional uncertainties (e.g. [0.05, 0.2, 0.2, 0.2]).
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
    target = "both" if result.has_xsec else "rate"

    cv_hist = np.asarray(result.hist_cv, dtype=float)
    frac_unc = np.asarray(frac_unc, dtype=float)
    if frac_unc.shape != cv_hist.shape:
        raise ValueError(
            f"frac_unc shape {frac_unc.shape} does not match hist_cv shape {cv_hist.shape}"
        )
    if np.any(frac_unc < 0):
        raise ValueError("frac_unc entries must be non-negative")
    if correlation not in {"diagonal", "fully_correlated"}:
        raise ValueError("correlation must be one of: 'diagonal', 'fully_correlated'")

    sigma = frac_unc * cv_hist
    if correlation == "diagonal":
        cov = np.diag(sigma ** 2)
    else:
        cov = np.outer(sigma, sigma)

    return add_uncertainty(
        result=result,
        cov=cov,
        key=key,
        category=category,
        target=target,
        unc=frac_unc,
        sum_value=float(np.mean(frac_unc)),
    )

def get_intime_cov (selected_df, var, bins, 
                    mcbnb_ngen,
                    mcbnb_pot,
                    threshold = 0.05,
                    event_type: str | None = "all",
                    select_region: str = "signal",
                    **selection_kwargs):
    mcint_dfs = load_dfs(config.INTIME_FILE,['histgenevtdf','nuecc'])
    scale = mcbnb_ngen/mcint_dfs['histgenevtdf'].TotalGenEvents.sum()
    if select_region == "signal":    mcint_df = select(mcint_dfs['nuecc'], savedict=False)
    elif select_region == "control": mcint_df = select_sideband(mcint_dfs['nuecc'], savedict=False)
    else:                            mcint_df = select(mcint_dfs['nuecc'], savedict=False, **selection_kwargs)
    mcint_df[('flux_pot_norm', '', '', '', '', '')] = scale/(integrated_flux * (mcbnb_pot / 1e6))
    # sort to avoid performance warning
    selected_df = apply_event_mask(ensure_lexsorted(selected_df,axis=1),event_type)
    mcint_df    = apply_event_mask(ensure_lexsorted(mcint_df,axis=1)) 
    
    cv_hist = get_hist1d(data=selected_df[var], bins=bins, 
                             weights=selected_df.flux_pot_norm)
    # remove offbeam contribution
    selected_no_offbeam_df = selected_df[selected_df.signal!=signal_dict['offbeam']]
    cv_hist_removed = get_hist1d(data = selected_no_offbeam_df[var],
                                     bins=bins, 
                                     weights = selected_no_offbeam_df.flux_pot_norm)
    
    # add the intime contribution
    int_hist = get_hist1d(data=mcint_df[var], bins=bins,weights=mcint_df.flux_pot_norm)
    dv_hist = cv_hist_removed + int_hist

    matrices = calc_matrices(dv_hist.reshape(len(bins)-1,-1),cv_hist)
    cov = matrices[0]
    unc = np.sqrt(np.diag(cov))/cv_hist
    # if the uncertainty is large enough, keep it for that bin
    # otherwise, we use the largest non-large uncertainty as a uniform uncertainty for all 
    large_unc = unc > threshold
    if np.any(~large_unc):
        uniform_unc_val = np.max(unc[~large_unc])
    else:
        uniform_unc_val = np.max(unc)
    unc_final = np.where(large_unc, unc, uniform_unc_val)
    # apply fully correlated uncertainty
    cov_final = np.outer(unc_final*cv_hist, unc_final*cv_hist)
    return cov_final
    
def get_total_cov(reco_df, reco_var, bins, mcbnb_pot,
                  selection_kwargs=None, projected_pot=1e20,
                  mcbnb_ngen: float | None = None,
                  intime_threshold: float = 0.05,
                  event_type: str | None = "all",
                  select_region: str = "signal",
                  uncertainty_keys: list[str] | tuple[str, ...] | set[str] | None = None,
                  xsec_inputs: XSecInputs | None = None,
                  detvar_dict: dict | None = None):
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
    selection_kwargs : dict, optional
        Additional selection cuts to apply
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
    if selection_kwargs is None:
        selection_kwargs = {}

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

    # -----------------------------
    # 1) Validate detvar region and load detvar inputs
    # -----------------------------
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
        if select_region == "all":
            detvar_dict = load_detvar_dicts(detvar_path)
        else:
            with open(detvar_path, 'rb') as f:
                detvar_dict = pickle.load(f)
        print(f"  Loaded {len(detvar_dict)} detector variation entries")
    elif not include_detv:
        detvar_dict = {}

    # Common selected sample and CV histogram used by all covariance terms.
    sorted_df = apply_event_mask(ensure_lexsorted(reco_df, axis=1), event_type)
    cv_hist = get_hist1d(data=sorted_df[reco_var], weights=sorted_df.flux_pot_norm, bins=bins)

    empty_syst_df = pd.DataFrame(columns=["key", "category", "unc", "sum", "top5"])

    # Rolling "overall" containers for the selected uncertainty blocks.
    rate_total_syst_dict: dict = {}
    rate_total_cov = np.zeros((cv_hist.size, cv_hist.size))
    rate_syst_frames: list[pd.DataFrame] = []

    if include_xsec:
        xsec_total_syst_dict: dict = {}
        xsec_total_cov = np.zeros((cv_hist.size, cv_hist.size))
        xsec_syst_frames: list[pd.DataFrame] = []

    # -----------------------------
    # 2) Rate systematics (non-detvar)
    # -----------------------------
    if include_rate:
        rate_syst_dict = get_syst(reco_df=sorted_df, reco_var=reco_var, bins=bins)
        rate_total_syst_dict.update(rate_syst_dict)
        rate_total_cov += _sum_covariances_from_dicts([rate_syst_dict], cv_hist.size)
        rate_syst_frames.append(get_syst_df([rate_syst_dict], cv_hist))

    # -----------------------------
    # 3) XSec systematics (optional, non-detvar)
    # -----------------------------
    if include_xsec:
        xsec_syst_dict = get_syst(reco_df=sorted_df, reco_var=reco_var, bins=bins, xsec_inputs=xsec_inputs)
        xsec_total_syst_dict.update(xsec_syst_dict)
        xsec_total_cov += _sum_covariances_from_dicts([xsec_syst_dict], cv_hist.size)
        xsec_syst_frames.append(get_syst_df([xsec_syst_dict], cv_hist))

    # -----------------------------
    # 4) Detector-variation systematics
    # -----------------------------
    if include_detv:
        detv_syst_dict = get_detvar_systs(detvar_dict, reco_var, bins, event_type=event_type, **selection_kwargs)
        detv_cov = _sum_covariances_from_dicts([detv_syst_dict], cv_hist.size)
        detv_syst_df = get_syst_df([detv_syst_dict], cv_hist)

        rate_total_syst_dict.update(detv_syst_dict)
        rate_total_cov += detv_cov
        rate_syst_frames.append(detv_syst_df)

        if include_xsec:
            xsec_total_syst_dict.update(detv_syst_dict)
            xsec_total_cov += detv_cov
            xsec_syst_frames.append(detv_syst_df)

    # Data statistical uncertainty row (kept as separate source label).
    if include_rate:
        data_err = np.sqrt(get_hist1d(data=sorted_df[reco_var], weights=reco_df.weights_mc, bins=bins) * (projected_pot / mcbnb_pot))
        flux_scale = integrated_flux * (projected_pot / 1e6)
        data_unc = np.divide(data_err, flux_scale * cv_hist, out=np.zeros_like(data_err, dtype=float), where=cv_hist > 0)
        data_syst_df = pd.DataFrame({'key': ['Datastat'], 'category': ['Datastat'], 'unc': [data_unc], 'sum': [np.mean(data_unc)], 'top5': [False]})
        rate_syst_frames.append(data_syst_df)
        if include_xsec:
            xsec_syst_frames.append(data_syst_df)

    rate_syst_df = pd.concat(rate_syst_frames, ignore_index=True) if rate_syst_frames else empty_syst_df.copy()
    if include_xsec:
        xsec_syst_df = pd.concat(xsec_syst_frames, ignore_index=True) if xsec_syst_frames else empty_syst_df.copy()

    # -----------------------------
    # 5) In-time cosmic covariance (optional)
    # -----------------------------
    intime_cov = None
    if include_cosmic and mcbnb_ngen is not None:
        intime_cov = get_intime_cov(selected_df=sorted_df, var=reco_var, bins=bins,
                                    mcbnb_ngen=mcbnb_ngen, mcbnb_pot=mcbnb_pot, threshold=intime_threshold,
                                    event_type=event_type, select_region=select_region, **selection_kwargs)

    # -----------------------------
    # 6) Final assembly + flat normalization/in-time additions
    # -----------------------------
    if not include_xsec:
        base_output = SystematicsOutput(
            hist_cv=cv_hist,
            rate_cov=rate_total_cov,
            rate_syst_df=rate_syst_df,
            rate_syst_dict=rate_total_syst_dict,
        )
        return _apply_norm_and_intime_uncertainties(
            base_output,
            intime_cov=intime_cov,
            include_norm=include_norm,
            include_cosmic=include_cosmic,
        )

    base_output = SystematicsOutput(
        hist_cv=cv_hist,
        rate_cov=rate_total_cov,
        rate_syst_df=rate_syst_df,
        rate_syst_dict=rate_total_syst_dict,
        xsec_cov=xsec_total_cov,
        xsec_syst_df=xsec_syst_df,
        xsec_syst_dict=xsec_total_syst_dict,
    )
    return _apply_norm_and_intime_uncertainties(
        base_output,
        intime_cov=intime_cov,
        include_norm=include_norm,
        include_cosmic=include_cosmic,
    )