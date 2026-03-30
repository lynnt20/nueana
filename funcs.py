import numpy as np
import pandas as pd
from dataclasses import replace
import warnings
import pickle
from tqdm import tqdm
from .utils import ensure_lexsorted
from .io import load_dfs
from .selection import select
from .histogram import get_hist1d, get_hist2d
from .syst import *
from .classes import SystematicsOutput, XSecInputs
from .constants import integrated_flux, signal_dict
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


def _load_detvar_dicts(
    detvar_files=None,
):
    """Load and combine detector variation dictionaries from pickle files.
    
    Parameters
    ----------
    detvar_files : list of str, optional
        List of paths to detvar dictionary pickle files. If None, uses config.DETVAR_DICT_FILES.
    
    Returns
    -------
    dict
        Combined detector variation and recombination dictionary.
    """
    if detvar_files is None:
        detvar_files = config.DETVAR_DICT_FILES
        
    combined_dict = {}
    for detvar_file in detvar_files:
        with open(detvar_file, 'rb') as f:
            file_dict = pickle.load(f)
            combined_dict.update(file_dict)
    
    return combined_dict


def _block_diag_cov(cov_a, cov_b):
    cov_a = np.asarray(cov_a, dtype=float)
    cov_b = np.asarray(cov_b, dtype=float)
    if cov_a.ndim != 2 or cov_b.ndim != 2 or cov_a.shape[0] != cov_a.shape[1] or cov_b.shape[0] != cov_b.shape[1]:
        raise ValueError("cov_a and cov_b must be square 2D covariance matrices")
    n_a = cov_a.shape[0]
    n_b = cov_b.shape[0]
    out = np.zeros((n_a + n_b, n_a + n_b), dtype=float)
    out[:n_a, :n_a] = cov_a
    out[n_a:, n_a:] = cov_b
    return out


def _normalize_event_mask(event_mask: str | None) -> str:
    if event_mask is None:
        return "all"
    if event_mask not in {"all", "signal", "background"}:
        raise ValueError("event_mask must be one of: 'all', 'signal', 'background', or None")
    return event_mask


def _apply_event_mask(df: pd.DataFrame, event_mask: str | None) -> pd.DataFrame:
    mask = _normalize_event_mask(event_mask)
    if mask == "signal":
        return df[df.signal == 0]
    if mask == "background":
        return df[df.signal != 0]
    return df


def _hists_from_frac_unc(cv_hist: np.ndarray, frac_unc: np.ndarray) -> np.ndarray:
    cv_hist = np.asarray(cv_hist, dtype=float)
    frac_unc = np.asarray(frac_unc, dtype=float)
    if frac_unc.shape != cv_hist.shape:
        raise ValueError(f"frac_unc shape {frac_unc.shape} does not match hist_cv shape {cv_hist.shape}")
    # One shifted universe whose bin-wise fractional shift matches frac_unc.
    return (cv_hist * (1.0 + frac_unc)).reshape(-1, 1)


def _apply_norm_and_intime_uncertainties(
    result: SystematicsOutput,
    intime_cov: np.ndarray | None = None,
    pot_norm_unc: float = 0.02,
    ntargets_unc: float = 0.01,
):
    updated = add_flat_norm_uncertainty(
        result=result,
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

    if intime_cov is not None:
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
            hists=_hists_from_frac_unc(cv_hist, intime_unc),
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
    """
    if frac_unc < 0:
        raise ValueError("frac_unc must be non-negative")
    if category is None:
        category = key

    cv_hist = np.asarray(result.hist_cv, dtype=float)
    flat_cov = (frac_unc ** 2) * np.outer(cv_hist, cv_hist)
    unc = np.full(cv_hist.shape, frac_unc, dtype=float)
    return add_uncertainty(
        result=result,
        cov=flat_cov,
        key=key,
        category=category,
        target="both" if result.has_xsec else "rate",
        unc=unc,
        hists=_hists_from_frac_unc(cv_hist, unc),
        sum_value=float(frac_unc),
    )


def add_fractional_uncertainty(
    result: SystematicsOutput,
    frac_unc: np.ndarray,
    key: str,
    category: str | None = None,
    target: str = "both",
    correlation: str = "diagonal",
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
    target
        Where to apply this uncertainty: "rate", "xsec", or "both".
    correlation
        Correlation model for bin-to-bin structure:
        - "diagonal": uncorrelated between bins.
        - "fully_correlated": 100% correlated across bins.
    """
    if category is None:
        category = key

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


def add_unisim_uncertainty(
    result: SystematicsOutput,
    alt_hist: np.ndarray,
    key: str,
    category: str | None = None,
    target: str = "both",
):
    """
    Add a single-universe (unisim) uncertainty to a SystematicsOutput.

    The alternate histogram is interpreted as one shifted prediction.
    The covariance contribution is built from:
        delta = alt_hist - hist_cv
        cov = outer(delta, delta)

    Parameters
    ----------
    result
        Existing systematics result object.
    alt_hist
        Alternate prediction histogram with the same shape as result.hist_cv.
    key
        Dictionary/dataframe key label for the new source.
    category
        Category label for the dataframe entry. Defaults to `key`.
    target
        Where to apply this uncertainty: "rate", "xsec", or "both".
    """
    if category is None:
        category = key

    cv_hist = np.asarray(result.hist_cv, dtype=float)
    alt_hist = np.asarray(alt_hist, dtype=float)
    if alt_hist.shape != cv_hist.shape:
        raise ValueError(
            f"alt_hist shape {alt_hist.shape} does not match hist_cv shape {cv_hist.shape}"
        )

    delta = alt_hist - cv_hist
    uni_cov = np.outer(delta, delta)
    unc = np.divide(np.abs(delta), cv_hist, out=np.zeros_like(cv_hist, dtype=float), where=cv_hist > 0)

    return add_uncertainty(
        result=result,
        cov=uni_cov,
        key=key,
        category=category,
        target=target,
        unc=unc,
        hists=alt_hist.reshape(-1, 1),
        sum_value=float(np.mean(unc)),
    )

def get_intime_cov (selected_df, var, bins, 
                    mcbnb_ngen,
                    mcbnb_pot,
                    threshold = 0.05,
                    intime_file="/scratch/7DayLifetime/lynnt/MCP2025B_v10_06_00_09/intime.df",
                    event_mask: str | None = "all",
                    **selection_kwargs):
    mcint_dfs = load_dfs(intime_file,['histgenevtdf','nuecc'])
    scale = mcbnb_ngen/mcint_dfs['histgenevtdf'].TotalGenEvents.sum()
    mcint_df = select(mcint_dfs['nuecc'],savedict=False,**selection_kwargs)
    mcint_df[('weights_mc', '', '', '', '', '')] = scale
    mcint_df[('flux_pot_norm', '', '', '', '', '')] = mcint_df.weights_mc/(integrated_flux * (mcbnb_pot / 1e6))
    # sort to avoid performance warning
    selected_df = ensure_lexsorted(selected_df,axis=1)
    mcint_df = ensure_lexsorted(mcint_df,axis=1)
    selected_df = _apply_event_mask(selected_df, event_mask)
    mcint_df = _apply_event_mask(mcint_df, event_mask)
    
    cv_hist = get_hist1d(data=selected_df[var], bins=bins, 
                             weights=selected_df.flux_pot_norm)
    # remove offbeam contribution
    selected_no_offbeam_df = selected_df[selected_df.signal!=signal_dict['offbeam']]
    cv_hist_removed = get_hist1d(data = selected_no_offbeam_df[var],
                                     bins=bins, 
                                     weights = selected_no_offbeam_df.flux_pot_norm)
    
    # add the intime contribution
    int_hist = get_hist1d(data=mcint_df[var], bins=bins,
                              weights=mcint_df.flux_pot_norm)
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
                  normalize=False, selection_kwargs=None, projected_pot=1e20, 
                  mcbnb_ngen: float | None = None,
                  intime_threshold: float = 0.05,
                  intime_file='/scratch/7DayLifetime/lynnt/MCP2025B_v10_06_00_09/intime.df',
                  event_mask: str | None = "all",
                  xsec_inputs: XSecInputs | None = None):
    """
    Get the total event-rate covariance matrix and systematic dataframe for a
    given variable. Optionally also compute the xsec covariance matrix and
    systematic dataframe when xsec_inputs are provided.

    The data statistical uncertainty is added as a separate "Datastat" entry in
    the returned event-rate dataframe, and in the xsec dataframe when requested.

    The returned result also includes systematic dictionaries:
    - rateate_syst_dict (includes DetVar keys)
    - xsec_syst_dict (includes DetVar keys, when xsec_inputs are provided)
    """

    if selection_kwargs is None:
        selection_kwargs = {}

    detvar_dict = _load_detvar_dicts()

    sorted_df = _apply_event_mask(ensure_lexsorted(reco_df, axis=1), event_mask)
    cv_hist = get_hist1d(data=sorted_df[reco_var], weights=sorted_df.flux_pot_norm, bins=bins)

    detv_syst_dict = get_detvar_systs(
        detvar_dict,
        reco_var,
        bins,
        normalize=normalize,
        event_mask=event_mask,
        **selection_kwargs,
    )
    rate_syst_dict = get_syst(reco_df=sorted_df, reco_var=reco_var, bins=bins, normalize=normalize)
    rate_total_syst_dict = {**rate_syst_dict, **detv_syst_dict}
    rate_syst_df = get_syst_df([rate_syst_dict, detv_syst_dict], cv_hist)

    data_err = np.sqrt(get_hist1d(data=sorted_df[reco_var], weights=reco_df.weights_mc, bins=bins)* (projected_pot / mcbnb_pot))
    flux_scale = integrated_flux * (projected_pot / 1e6)
    data_unc = np.divide(data_err, flux_scale * cv_hist, out=np.zeros_like(data_err, dtype=float), where=cv_hist > 0)
    data_syst_df = pd.DataFrame({'key': ['Datastat'], 'category': ['Datastat'], 'unc': [data_unc], 'sum': [np.mean(data_unc)], 'top5': [False]})

    rate_syst_df = pd.concat([rate_syst_df, data_syst_df], ignore_index=True)
    rate_cov = _sum_covariances_from_dicts([rate_syst_dict, detv_syst_dict], cv_hist.size)

    intime_cov = None
    if mcbnb_ngen is not None:
        intime_cov = get_intime_cov(
            selected_df=sorted_df,
            var=reco_var,
            bins=bins,
            mcbnb_ngen=mcbnb_ngen,
            mcbnb_pot=mcbnb_pot,
            threshold=intime_threshold,
            intime_file=intime_file,
            event_mask=event_mask,
            **selection_kwargs,
        )

    if xsec_inputs is None:
        base_output = SystematicsOutput(
            hist_cv=cv_hist,
            rate_cov=rate_cov,
            rate_syst_df=rate_syst_df,
            rate_syst_dict=rate_total_syst_dict,
        )
        return _apply_norm_and_intime_uncertainties(base_output, intime_cov=intime_cov)

    xsec_syst_dict = get_syst(
        reco_df=sorted_df,
        reco_var=reco_var,
        bins=bins,
        normalize=normalize,
        xsec_inputs=xsec_inputs,
    )
    xsec_total_syst_dict = {**xsec_syst_dict, **detv_syst_dict}
    xsec_syst_df = get_syst_df([xsec_syst_dict, detv_syst_dict], cv_hist)
    xsec_syst_df = pd.concat([xsec_syst_df, data_syst_df], ignore_index=True)
    xsec_cov = _sum_covariances_from_dicts([xsec_syst_dict, detv_syst_dict], cv_hist.size)

    base_output = SystematicsOutput(
        hist_cv=cv_hist,
        rate_cov=rate_cov,
        rate_syst_df=rate_syst_df,
        rate_syst_dict=rate_total_syst_dict,
        xsec_cov=xsec_cov,
        xsec_syst_df=xsec_syst_df,
        xsec_syst_dict=xsec_total_syst_dict,
    )
    return _apply_norm_and_intime_uncertainties(base_output, intime_cov=intime_cov)
    
'''
Control Region
'''

def _combine_syst_hist_dicts(sel_dict, ctrl_dict):
    combined = {}
    shared_keys = set(sel_dict).intersection(set(ctrl_dict))

    missing_sel = sorted(set(ctrl_dict) - set(sel_dict))
    missing_ctrl = sorted(set(sel_dict) - set(ctrl_dict))
    if missing_sel or missing_ctrl:
        raise ValueError(
            "Systematic keys do not match between selected and control regions. "
            f"Missing in selected: {missing_sel}; missing in control: {missing_ctrl}"
        )

    for key in shared_keys:
        sel_h = sel_dict[key]["hists"]
        ctrl_h = ctrl_dict[key]["hists"]

        if sel_h.ndim == 1:
            sel_h = sel_h.reshape(-1, 1)
        if ctrl_h.ndim == 1:
            ctrl_h = ctrl_h.reshape(-1, 1)

        if sel_h.shape[1] != ctrl_h.shape[1]:
            raise ValueError(
                f"Universe-count mismatch for {key}: "
                f"selected={sel_h.shape[1]}, control={ctrl_h.shape[1]}"
            )

        combined[key] = {"hists": np.concatenate([sel_h, ctrl_h], axis=0)}

    return combined

def _add_matrices_in_place(syst_hist_dict, cv_hist):
    for key in syst_hist_dict:
        cov, cov_frac, corr = calc_matrices(syst_hist_dict[key]["hists"], cv_hist)
        syst_hist_dict[key]["cov"] = cov
        syst_hist_dict[key]["cov_frac"] = cov_frac
        syst_hist_dict[key]["corr"] = corr
    return syst_hist_dict

def get_total_cov_combined(
    reco_df,
    reco_control_df,
    reco_var,
    bins,
    mcbnb_pot: float | None = None,
    normalize=False,
    selected_selection_kwargs=None,
    control_selection_kwargs=None,
    selected_event_mask: str | None = "all",
    control_event_mask: str | None = "all",
    mcbnb_ngen: float | None = None,
    intime_threshold: float = 0.05,
    intime_file='/scratch/7DayLifetime/lynnt/MCP2025B_v10_06_00_09/intime.df',
    xsec_inputs: XSecInputs | None = None,
):
    # ! TODO: add data statistics systematics
    """
    Build covariance/results for a concatenated selected+control measurement.

    Workflow:
    1) run get_syst_hists separately in selected and control regions,
    2) concatenate per-systematic histograms,
    3) run calc_matrices on the concatenated histograms.

    Notes:
    - `reco_var` and `bins` are shared between selected/control.
    - selection kwargs can still differ between selected/control.
    - Returned result includes systematic dictionaries:
      rateate_syst_dict includes DetVar keys, and xsec_syst_dict (if requested)
      includes DetVar keys.
    """
    if selected_selection_kwargs is None:
        selected_selection_kwargs = {}
    if control_selection_kwargs is None:
        control_selection_kwargs = {}

    # CV histograms for selected and control, then concatenate.
    reco_df = _apply_event_mask(ensure_lexsorted(reco_df, axis=1), selected_event_mask)
    reco_control_df = _apply_event_mask(ensure_lexsorted(reco_control_df, axis=1), control_event_mask)

    cv_sel = get_hist1d(data=reco_df[reco_var], weights=reco_df.flux_pot_norm, bins=bins)
    cv_ctrl = get_hist1d(data=reco_control_df[reco_var], weights=reco_control_df.flux_pot_norm, bins=bins)
    cv_hist = np.concatenate([cv_sel, cv_ctrl])

    # Event-rate reweight systematics from selected/control, then concatenate and matrixify.
    rate_sel_hists, _ = get_syst_hists(
        reco_df=reco_df,
        reco_var=reco_var,
        bins=bins,
        normalize=normalize,
    )
    rate_ctrl_hists, _ = get_syst_hists(
        reco_df=reco_control_df,
        reco_var=reco_var,
        bins=bins,
        normalize=normalize,
    )
    rate_syst_dict = _add_matrices_in_place(
        _combine_syst_hist_dicts(rate_sel_hists, rate_ctrl_hists),
        cv_hist,
    )
    
    detvar_dict = _load_detvar_dicts()
    detv_sel_hists = get_detvar_systs(
        detvar_dict,
        reco_var,
        bins,
        normalize=normalize,
        event_mask=selected_event_mask,
        **selected_selection_kwargs,
    )
    detv_ctrl_hists = get_detvar_systs(
        detvar_dict,
        reco_var,
        bins,
        normalize=normalize,
        event_mask=control_event_mask,
        **control_selection_kwargs,
    )
    detv_syst_dict = _add_matrices_in_place(
        _combine_syst_hist_dicts(detv_sel_hists, detv_ctrl_hists),
        cv_hist,
    )
    rate_total_syst_dict = {**rate_syst_dict, **detv_syst_dict}

    rate_syst_df = get_syst_df([rate_syst_dict, detv_syst_dict], cv_hist)
    # rate_syst_df = pd.concat([rate_syst_df, data_syst_df], ignore_index=True)
    rate_cov = _sum_covariances_from_dicts([rate_syst_dict, detv_syst_dict], cv_hist.size)

    intime_cov = None
    if mcbnb_ngen is not None and mcbnb_pot is not None:
        intime_sel_cov = get_intime_cov(
            selected_df=reco_df,
            var=reco_var,
            bins=bins,
            mcbnb_ngen=mcbnb_ngen,
            mcbnb_pot=mcbnb_pot,
            threshold=intime_threshold,
            intime_file=intime_file,
            event_mask=selected_event_mask,
            **selected_selection_kwargs,
        )
        intime_ctrl_cov = get_intime_cov(
            selected_df=reco_control_df,
            var=reco_var,
            bins=bins,
            mcbnb_ngen=mcbnb_ngen,
            mcbnb_pot=mcbnb_pot,
            threshold=intime_threshold,
            intime_file=intime_file,
            event_mask=control_event_mask,
            **control_selection_kwargs,
        )
        intime_cov = _block_diag_cov(intime_sel_cov, intime_ctrl_cov)
    elif mcbnb_ngen is not None or mcbnb_pot is not None:
        warnings.warn(
            "Both mcbnb_ngen and mcbnb_pot are required for IntimeCosmics; skipping this uncertainty",
            stacklevel=2,
        )

    if xsec_inputs is None:
        base_output = SystematicsOutput(
            hist_cv=cv_hist,
            rate_cov=rate_cov,
            rate_syst_df=rate_syst_df,
            rate_syst_dict=rate_total_syst_dict,
        )
        return _apply_norm_and_intime_uncertainties(base_output, intime_cov=intime_cov)
    
    print("Calculating cross-section systematics...")
    xsec_sel_hists, _ = get_syst_hists(
        reco_df=reco_df,
        reco_var=reco_var,
        bins=bins,
        normalize=normalize,
        xsec_inputs=xsec_inputs,
    )
    xsec_ctrl_hists, _ = get_syst_hists(
        reco_df=reco_control_df,
        reco_var=reco_var,
        bins=bins,
        normalize=normalize,
        xsec_inputs=xsec_inputs,
    )
    xsec_syst_dict = _add_matrices_in_place(
        _combine_syst_hist_dicts(xsec_sel_hists, xsec_ctrl_hists),
        cv_hist,
    )
    xsec_total_syst_dict = {**xsec_syst_dict, **detv_syst_dict}
    xsec_syst_df = get_syst_df([xsec_syst_dict, detv_syst_dict], cv_hist)
    # rate_syst_df = pd.concat([rate_syst_df, data_syst_df], ignore_index=True)
    xsec_cov = _sum_covariances_from_dicts([xsec_syst_dict, detv_syst_dict], cv_hist.size)

    base_output = SystematicsOutput(
        hist_cv=cv_hist,
        rate_cov=rate_cov,
        rate_syst_df=rate_syst_df,
        rate_syst_dict=rate_total_syst_dict,
        xsec_cov=xsec_cov,
        xsec_syst_df=xsec_syst_df,
        xsec_syst_dict=xsec_total_syst_dict,
    )
    return _apply_norm_and_intime_uncertainties(base_output, intime_cov=intime_cov)
    
    
    