"""Fake-data test (FDT) helpers: fake histogram building and ensemble trial runner.

Unfolding infrastructure (UnfoldInput, get_response_matrix, plot_unfolded_result)
lives in :mod:`nueana.unfold`. This module contains only the FDT-specific layer:
building reweighted fake-data histograms and running/plotting the random-background
ensemble study.
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections.abc import Callable

from .utils import get_hist1d, ensure_lexsorted
from .classes import VariableConfig
from .analysis import integrated_flux, NTARGETS
from .funcs import chi_squared
from .unfold import UnfoldInput, _xsec_ylabel

__all__ = [
    'make_fake_data_hists',
    'run_random_background_fdt',
    'plot_random_background_fdt',
]


# ---------------------------------------------------------------------------


def make_fake_data_hists(
    reco_df: pd.DataFrame,
    true_df: pd.DataFrame,
    var: VariableConfig,
    reco_mask: np.ndarray,
    true_mask: np.ndarray,
    weight: float,
    side_df: pd.DataFrame | None = None,
    ccbc_cov: dict | None = None,
    side_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    """Build background-subtracted fake-data and modified signal histograms for an FDT.

    All returned histograms are in absolute event-count units at the sample's
    nominal POT (``weights_mc`` summed, no flux division), consistent with the
    covariance matrices and CV histograms in :class:`~nueana.classes.SystematicsOutput`.

    Without CCBC inputs the CV background (signal != 0) is subtracted.  When
    ``side_df`` and ``ccbc_cov`` are provided the CCBC-constrained background is
    subtracted instead, using the constraint:

    .. math::

        \\hat{B}_S = B_S^{\\rm CV}
            + C_{B_S n_C}\\, C_{n_C n_C}^{+}\\, (n_C^{\\rm FD} - n_C^{\\rm CV})

    Parameters
    ----------
    reco_df : pd.DataFrame
        Full selected signal-region sample (signal + background) with
        ``weights_mc`` and ``signal`` columns.
    true_df : pd.DataFrame
        Truth-level signal-only sample with ``weights_mc``.
    var : VariableConfig
        Variable configuration supplying column keys and bin edges.
    reco_mask : np.ndarray of bool
        Events in reco_df to reweight.
    true_mask : np.ndarray of bool
        Events in true_df to reweight.
    weight : float
        Multiplicative scale applied to masked events.
    side_df : pd.DataFrame or None, optional
        Control-region (sideband) sample with ``weights_mc``.  Required when
        using CCBC constraint.
    ccbc_cov : dict or None, optional
        Output of :func:`~nueana.ccbc.get_ccbc_cov`.  When provided (together
        with ``side_df``) the CCBC-constrained background is subtracted instead
        of the CV background.
    side_mask : np.ndarray of bool or None, optional
        Events in side_df to reweight.  Pass ``None`` to leave side_df
        unmodified (CV sideband — constraint correction is zero).

    Returns
    -------
    fd_meas : np.ndarray
        Background-subtracted fake measurement in absolute event-count units at mcbnb_pot.
    fd_true : np.ndarray
        Modified truth-level signal histogram in absolute event-count units at mcbnb_pot.
    fd_ns : np.ndarray
        Total (signal + background) fake-data signal-region histogram in
        absolute event-count units at mcbnb_pot, before background subtraction.
        Plays the role of observed data (``D_S``) in
        :func:`~nueana.ccbc.plot_ccbc_fd_comparison`.
    fd_nc : np.ndarray or None
        Control-region fake-data histogram in absolute event-count units at mcbnb_pot,
        or ``None`` when ``side_df`` is not provided.
    fd_Bs : np.ndarray
        True reweighted background histogram in absolute event-count units at mcbnb_pot
        — the background-event (``signal != 0``) contribution under the FDT weights.
        Distinct from the CV background (no reweighting) and from the
        CCBC-constrained prediction; use as the "ground-truth background" overlay
        in :func:`~nueana.ccbc.plot_ccbc_fd_comparison`.
    """
    reco_df = ensure_lexsorted(reco_df, axis=1)
    true_df = ensure_lexsorted(true_df, axis=1)

    reco_weights = reco_df.weights_mc.values.copy()
    reco_weights[reco_mask] *= weight

    true_weights = true_df.weights_mc.values.copy()
    true_weights[true_mask] *= weight

    fd_ns = get_hist1d(
        data=reco_df[var.var_evt_reco_col], bins=var.bins, weights=reco_weights,
    )
    fd_true = get_hist1d(
        data=true_df[var.var_nu_col], bins=var.bins, weights=true_weights,
    )
    bg_weights = reco_weights.copy()
    bg_weights[reco_df.signal.values == 0] = 0
    fd_Bs = get_hist1d(
        data=reco_df[var.var_evt_reco_col], bins=var.bins, weights=bg_weights,
    )

    if (side_df is None) != (ccbc_cov is None):
        warnings.warn(
            "CCBC constraint requires both side_df and ccbc_cov; "
            "only one was provided — falling back to CV background subtraction.",
            stacklevel=2,
        )

    if side_df is not None and ccbc_cov is not None:
        from .ccbc import get_constrained_background
        side_df = ensure_lexsorted(side_df, axis=1)
        side_weights = side_df.weights_mc.values.copy()
        if side_mask is not None:
            side_weights[side_mask] *= weight
        fd_nc = get_hist1d(
            data=side_df[var.var_evt_reco_col], bins=var.bins, weights=side_weights,
        )
        fd_Bs_constrained = get_constrained_background(ccbc_cov, fd_nc)
        fd_meas = fd_ns - fd_Bs_constrained
        return fd_meas, fd_true, fd_ns, fd_nc, fd_Bs

    backgr_df = reco_df[reco_df.signal != 0]
    cv_backgr_hist = get_hist1d(
        data=backgr_df[var.var_evt_reco_col], bins=var.bins,
        weights=backgr_df.weights_mc,
    )
    fd_meas = fd_ns - cv_backgr_hist
    return fd_meas, fd_true, fd_ns, None, fd_Bs


# ---------------------------------------------------------------------------


def run_random_background_fdt(
    reco_df: pd.DataFrame,
    true_df: pd.DataFrame,
    side_df: pd.DataFrame,
    ccbc_cov: dict,
    var: VariableConfig,
    unf_input: UnfoldInput,
    wienersvd_fn: Callable,
    mask_fns: list[Callable[[pd.DataFrame], np.ndarray]],
    n_trials: int = 100,
    weight_range: tuple[float, float] = (0.5, 1.5),
    rng: np.random.Generator | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Ensemble fake-data test with randomly reweighted background categories.

    For each trial, draws one random weight per background category from
    ``weight_range`` (uniform), builds fake-data histograms, applies the CCBC
    constraint, unfolds, and records the total integrated cross-section and
    chi-squared p-value vs the CV signal truth.  The unconstrained result is
    recorded in parallel so the two can be compared.

    Masks from ``mask_fns`` are evaluated on ``reco_df`` (restricted to
    background events, ``signal != 0``) and on ``side_df`` (all events).
    Category weights are applied via ``np.select`` semantics: the first
    matching mask wins, so masks should be mutually exclusive.  Events that
    match no mask keep weight 1.0.

    Parameters
    ----------
    reco_df : pd.DataFrame
        Full selected signal-region sample (signal + background) with
        ``weights_mc`` and ``signal`` columns.
    true_df : pd.DataFrame
        Truth-level signal-only sample with ``weights_mc``.  Not reweighted;
        ``unf_input.cv_signal`` is used as the truth reference throughout.
    side_df : pd.DataFrame
        Control-region (sideband) sample with ``weights_mc``.
    ccbc_cov : dict
        Output of :func:`~nueana.ccbc.get_ccbc_cov`.  Must contain
        ``cov_ns_ns``, ``cov_ms_ms``, ``Bs_output``, ``cov_Bs_nc``, and
        ``pinv_nc_nc``.
    var : VariableConfig
        Variable configuration supplying column keys and bin edges.
    unf_input : UnfoldInput
        Pre-built unfolding inputs (response matrix, CV signal, systematic
        covariances, ``mcbnb_pot``).
    wienersvd_fn : callable
        The WienerSVD function imported from cafpyana.
    mask_fns : list of callables
        Each callable takes a lexsorted DataFrame and returns a boolean
        array/Series defining one background category (e.g.
        ``primtrk_is_muon``).  Evaluated once before the loop.
    n_trials : int, default 100
        Number of random trials.
    weight_range : (float, float), default (0.5, 1.5)
        Lower and upper bounds of the uniform weight distribution.
    rng : np.random.Generator or None
        NumPy random generator.  A fresh ``np.random.default_rng()`` is
        created if not provided.
    show_progress : bool, default True
        Display a tqdm progress bar when tqdm is available.

    Returns
    -------
    pd.DataFrame
        One row per trial.  Columns:

        ``weights``      — tuple of per-category scale factors (one per mask_fn)
        ``n_bkg``        — total weighted background count in the signal region
        ``n_side``       — total weighted count in the sideband
        ``xsec_constr``  — total CCBC-constrained cross-section [cm²/nucleon]
        ``xsec_uncon``   — total unconstrained cross-section [cm²/nucleon]
        ``chisq_constr`` — χ² of constrained unfolded vs smeared CV truth
        ``chisq_uncon``  — χ² of unconstrained unfolded vs smeared CV truth
        ``pval_constr``  — p-value for constrained result
        ``pval_uncon``   — p-value for unconstrained result
        ``xsec_err_constr`` — expected ±1σ on total xsec from constrained UnfoldCov
                             (= sqrt(sum(UnfoldCov)); constant across trials)
        ``xsec_err_uncon``  — same for unconstrained
        ``unfold_constr``   — per-trial unfolded spectrum array (xsec units, shape (nbins,))
        ``unfold_uncon``    — same for unconstrained
        ``cv_smear_constr`` — smeared CV truth used as reference (constant; shape (nbins,))
        ``cv_smear_uncon``  — same for unconstrained
        ``xsec_cv``         — CV total cross-section (constant reference)
        ``ndof``            — number of true bins (chi-squared degrees of freedom)
        ``n_bkg_cv``        — CV background count in the signal region (constant)
        ``n_side_cv``       — CV sideband count (constant)
    """
    from .ccbc import get_constrained_background

    if rng is None:
        rng = np.random.default_rng()

    reco_df = ensure_lexsorted(reco_df, axis=1)
    side_df = ensure_lexsorted(side_df, axis=1)

    reco_base = reco_df.weights_mc.values.copy()
    side_base = side_df.weights_mc.values.copy()
    is_bkg    = reco_df.signal.values != 0

    # Evaluate mask functions once; background-only filter applied to reco
    reco_masks = [np.asarray(fn(reco_df), dtype=bool) & is_bkg for fn in mask_fns]
    side_masks = [np.asarray(fn(side_df), dtype=bool)          for fn in mask_fns]

    pot_scale = float(ccbc_cov.get("pot_scale", 1.0))
    bkg_S_cv  = np.asarray(ccbc_cov["Bs_output"].rate_hist_cv) * pot_scale
    cov_pre   = np.asarray(ccbc_cov["cov_ns_ns"])
    cov_post  = np.asarray(ccbc_cov["cov_ms_ms"])

    xsec_scale = 1.0 / (integrated_flux * unf_input.mcbnb_pot * NTARGETS)
    xsec_cv    = float(np.sum(unf_input.cv_signal)) * xsec_scale
    ndof       = len(unf_input.cv_signal)

    n_bkg_cv  = float(np.dot(reco_base, is_bkg))
    n_side_cv = float(np.sum(side_base))

    unf_input_1bin = unf_input.integrate()
    cov_pre_1bin   = np.array([[np.sum(cov_pre)]])
    cov_post_1bin  = np.array([[np.sum(cov_post)]])

    try:
        from scipy.stats import chi2 as _chi2_dist
        def _pval(chisq: float) -> float:
            return float(_chi2_dist.sf(chisq, df=ndof))
    except Exception:
        def _pval(chisq: float) -> float:  # type: ignore[misc]
            return float("nan")

    # UnfoldCov = WF @ Cov @ WF.T doesn't depend on Measure, so all per-bin
    # and integrated errors can be extracted once with a zero measurement.
    _zero     = np.zeros(unf_input.response.shape[0])
    _zero_1   = np.zeros(1)
    cv_unfold_err_constr = np.sqrt(np.diag(
        unf_input.unfold(wienersvd_fn, measure=_zero, total_cov=cov_post)["UnfoldCov"]
    ))
    cv_unfold_err_uncon = np.sqrt(np.diag(
        unf_input.unfold(wienersvd_fn, measure=_zero, total_cov=cov_pre)["UnfoldCov"]
    ))
    xsec_err_constr = float(np.sqrt(
        unf_input_1bin.unfold(wienersvd_fn, measure=_zero_1, total_cov=cov_post_1bin)["UnfoldCov"][0, 0]
    ))
    xsec_err_uncon = float(np.sqrt(
        unf_input_1bin.unfold(wienersvd_fn, measure=_zero_1, total_cov=cov_pre_1bin)["UnfoldCov"][0, 0]
    ))

    itr = range(n_trials)
    if show_progress:
        try:
            from tqdm import tqdm
            itr = tqdm(itr, desc="FDT trials")
        except ImportError:
            pass

    records = []
    for _ in itr:
        trial_weights = rng.uniform(weight_range[0], weight_range[1], size=len(mask_fns))

        if reco_masks:
            w_reco_scale = np.select(reco_masks, trial_weights.tolist(), default=1.0)
            w_side_scale = np.select(side_masks, trial_weights.tolist(), default=1.0)
        else:
            w_reco_scale = np.ones(len(reco_base))
            w_side_scale = np.ones(len(side_base))

        reco_w = reco_base * w_reco_scale
        side_w = side_base * w_side_scale

        n_bkg  = float(np.dot(reco_w, is_bkg))
        n_side = float(np.sum(side_w))

        fd_ns = get_hist1d(
            data=reco_df[var.var_evt_reco_col], bins=var.bins,
            weights=reco_w,
        )
        fd_nc = get_hist1d(
            data=side_df[var.var_evt_reco_col], bins=var.bins,
            weights=side_w,
        )

        fd_Bs_constr   = get_constrained_background(ccbc_cov, fd_nc)
        fd_meas_constr = fd_ns - fd_Bs_constr
        fd_meas_uncon  = fd_ns - bkg_S_cv

        res_constr = unf_input.unfold(wienersvd_fn, measure=fd_meas_constr, total_cov=cov_post)
        res_uncon  = unf_input.unfold(wienersvd_fn, measure=fd_meas_uncon,  total_cov=cov_pre)

        # Single-bin unfolding for the integrated cross-section
        res_1bin_constr = unf_input_1bin.unfold(
            wienersvd_fn,
            measure=np.array([np.sum(fd_meas_constr)]),
            total_cov=cov_post_1bin,
        )
        res_1bin_uncon = unf_input_1bin.unfold(
            wienersvd_fn,
            measure=np.array([np.sum(fd_meas_uncon)]),
            total_cov=cov_pre_1bin,
        )
        xsec_constr = float(res_1bin_constr["unfold"][0])
        xsec_uncon  = float(res_1bin_uncon["unfold"][0])

        cv_smear_constr = res_constr["AddSmear"] @ unf_input.cv_signal * xsec_scale
        cv_smear_uncon  = res_uncon["AddSmear"]  @ unf_input.cv_signal * xsec_scale

        chisq_constr = float(chi_squared(res_constr["unfold"] - cv_smear_constr, res_constr["UnfoldCov"]))
        chisq_uncon  = float(chi_squared(res_uncon["unfold"]  - cv_smear_uncon,  res_uncon["UnfoldCov"]))

        records.append({
            "weights":          tuple(float(w) for w in trial_weights),
            "n_bkg":            n_bkg,
            "n_side":           n_side,
            "xsec_constr":      xsec_constr,
            "xsec_uncon":       xsec_uncon,
            "chisq_constr":     chisq_constr,
            "chisq_uncon":      chisq_uncon,
            "pval_constr":      _pval(chisq_constr),
            "pval_uncon":       _pval(chisq_uncon),
            "unfold_constr":    res_constr["unfold"].copy(),
            "unfold_uncon":     res_uncon["unfold"].copy(),
            "cv_smear_constr":  cv_smear_constr.copy(),
            "cv_smear_uncon":   cv_smear_uncon.copy(),
        })

    result_df = pd.DataFrame(records)
    result_df.index.name = "trial"
    result_df["xsec_cv"]             = xsec_cv
    result_df["xsec_err_constr"]     = xsec_err_constr
    result_df["xsec_err_uncon"]      = xsec_err_uncon
    result_df["ndof"]                = ndof
    result_df["n_bkg_cv"]            = n_bkg_cv
    result_df["n_side_cv"]           = n_side_cv
    result_df["cv_unfold_err_constr"] = [cv_unfold_err_constr] * n_trials
    result_df["cv_unfold_err_uncon"]  = [cv_unfold_err_uncon]  * n_trials
    return result_df


def plot_random_background_fdt(
    result_df: pd.DataFrame,
    var: VariableConfig | None = None,
    axes: np.ndarray | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Summary plots for a :func:`run_random_background_fdt` ensemble.

    Produces a 1×2 figure (p-value histogram and integrated cross-section
    histogram).  When ``var`` is supplied and the DataFrame contains per-trial
    spectra columns (``unfold_constr`` / ``unfold_uncon``), a second row of
    panels is added showing the full unfolded spectrum ensemble — 16th–84th
    percentile band across trials plus the median, compared to the smeared CV
    truth.

    Parameters
    ----------
    result_df : pd.DataFrame
        Output of :func:`run_random_background_fdt`.
    var : VariableConfig, optional
        Variable configuration for the spectrum panels.  When provided (and
        per-trial spectra are present in ``result_df``), two additional panels
        are drawn showing the shape of the unfolded ensemble.
    axes : array of Axes, optional
        Pre-existing axes.  Shape (2,) when ``var`` is None; shape (2, 2) when
        ``var`` is provided.  A new figure is created if not supplied.

    Returns
    -------
    fig : plt.Figure
    axes : np.ndarray
        Shape (2,) without spectrum panels; shape (2, 2) with them.
    """
    has_spectra = (
        var is not None
        and "unfold_constr" in result_df.columns
        and "unfold_uncon"  in result_df.columns
    )

    if axes is None:
        if has_spectra:
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            plt.subplots_adjust(hspace=0.45, wspace=0.3)
        else:
            fig, _ax = plt.subplots(1, 2, figsize=(10, 4))
            plt.subplots_adjust(wspace=0.3)
            axes = _ax.reshape(1, 2)
    else:
        axes = np.asarray(axes)
        fig  = axes.flat[0].get_figure()

    axes = axes.reshape(-1, 2)   # uniform (n_rows, 2) indexing
    ax_pval, ax_xsec = axes[0]

    xsec_cv = float(result_df["xsec_cv"].iloc[0])

    # ── Row 0 left: p-value histogram ──────────────────────────────────────
    bins_p = np.linspace(0, 1, 21)
    ax_pval.hist(result_df["pval_uncon"],  bins=bins_p, alpha=0.4, hatch="/", color="gray",   label="unconstrained")
    ax_pval.hist(result_df["pval_constr"], bins=bins_p, alpha=0.7,            color="purple", label="constrained")
    ax_pval.set_xlabel("p-value")
    ax_pval.set_ylabel("Trials")
    ax_pval.set_title("p-value distribution")
    ax_pval.legend(fontsize=9)

    # ── Row 0 right: integrated cross-section histogram ────────────────────
    # xsec_err is constant across trials (UnfoldCov doesn't depend on Measure)
    xsec_err_constr = float(result_df["xsec_err_constr"].iloc[0])
    xsec_err_uncon  = float(result_df["xsec_err_uncon"].iloc[0])

    all_xsec = np.concatenate([result_df["xsec_constr"], result_df["xsec_uncon"]])
    lo, hi   = all_xsec.min(), all_xsec.max()
    spread   = max(hi - lo, xsec_err_uncon * 2)
    bins_x   = np.linspace(lo - 0.15 * spread, hi + 0.15 * spread, 31)
    ax_xsec.hist(result_df["xsec_uncon"],  bins=bins_x, alpha=0.4, hatch = "/", color="gray",   label=rf"unconstrained $\pm1\sigma$ ({xsec_err_uncon:.2e})")
    ax_xsec.hist(result_df["xsec_constr"], bins=bins_x, alpha=0.7,              color="purple", label=rf"constrained  $\pm1\sigma$ ({xsec_err_constr:.2e})")
    ax_xsec.axvline(xsec_cv, color="k", ls="--", lw=1.2, label="CV")
    ax_xsec.axvspan(xsec_cv - xsec_err_uncon,  xsec_cv + xsec_err_uncon,
                    alpha=0.10, color="gray")
    ax_xsec.axvspan(xsec_cv - xsec_err_constr, xsec_cv + xsec_err_constr,
                    alpha=0.20, color="purple")
    ax_xsec.set_xlabel(r"Total $\sigma$ [cm$^2$ nucleon$^{-1}$]")
    ax_xsec.set_ylabel("Trials")
    ax_xsec.set_title("Single-Bin cross-section")
    ax_xsec.legend(fontsize=9)

    # ── Row 1: per-bin unfolded spectrum ensemble ──────────────────────────
    if has_spectra:
        bins_v  = var.bins
        centers = var.bin_centers
        widths  = np.diff(bins_v)

        ylabel = _xsec_ylabel(var)

        all_unfold_uncon  = np.stack(result_df["unfold_uncon"].values)   # (n_trials, nbins)
        all_unfold_constr = np.stack(result_df["unfold_constr"].values)
        cv_smear_uncon    = np.asarray(result_df["cv_smear_uncon"].iloc[0])
        cv_smear_constr   = np.asarray(result_df["cv_smear_constr"].iloc[0])
        cv_err_uncon      = np.asarray(result_df["cv_unfold_err_uncon"].iloc[0])
        cv_err_constr     = np.asarray(result_df["cv_unfold_err_constr"].iloc[0])

        for ax, all_unfold, cv_smear, cv_err, color, title in [
            (axes[1, 0], all_unfold_uncon,  cv_smear_uncon,  cv_err_uncon,  "gray",   "Unconstrained"),
            (axes[1, 1], all_unfold_constr, cv_smear_constr, cv_err_constr, "purple", "Constrained"),
        ]:
            per_w = all_unfold / widths          # (n_trials, nbins), differential xsec

            for i, row in enumerate(per_w):
                ax.stairs(row, bins_v, color=color, lw=0.8, alpha=0.2,
                          label="trials" if i == 0 else None)
            ax.stairs(cv_smear / widths, bins_v, color="k", lw=2)
            ax.errorbar(centers, cv_smear / widths, yerr=cv_err / widths,
                        fmt="k.", lw=1.5, capsize=3, label="CV smeared truth ±1σ")
            ax.set_xlabel(var.var_labels[0], fontsize=10)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(title)
            ax.set_xticks(bins_v)
            ax.set_xticklabels(var.bin_labels, fontsize=8)
            ax.legend(fontsize=9)

    return fig, axes.squeeze() if not has_spectra else axes
