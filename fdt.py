"""Fake-data test (FDT) helpers: response matrix construction and fake-data histogram building.

These utilities handle the cafpyana-free layer of an FDT workflow. The WienerSVD
unfolding call and any generator-level plotting remain in the notebook, since they
cross the cafpyana boundary.
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from collections.abc import Callable

from .utils import get_hist1d, get_hist2d, ensure_lexsorted, bin_geometry
from .classes import VariableConfig, SystematicsOutput
from .syst import key_in_allowed, decompose_cov
from .analysis import integrated_flux, NTARGETS
from .funcs import chi_squared, format_chisq_label

__all__ = [
    'UnfoldInput',
    'get_response_matrix',
    'make_fake_data_hists',
    'plot_unfolded_result',
    'run_random_background_fdt',
    'plot_random_background_fdt',
]


_DEFAULT_TRUTH_COLORS = ["C0", "yellowgreen", "C3", "C4", "C5"]


def _xsec_ylabel(var: VariableConfig) -> str:
    """dσ/d<var> axis label in cm²·nucleon⁻¹ units."""
    plot_math = var.var_plot_name.strip("$")
    unit_part = rf"\,\mathrm{{{var.var_unit}}}^{{-1}}" if var.var_unit else ""
    return (
        rf"$d\sigma / d{plot_math}$ "
        rf"$[\mathrm{{cm}}^2{unit_part}\,\mathrm{{nucleon}}^{{-1}}]$"
    )


# ---------------------------------------------------------------------------


@dataclass
class UnfoldInput:
    """Pre-built inputs to a WienerSVD unfolding call.

    Stores the fixed components (response matrix, CV signal prediction,
    per-key systematic covariance matrices, and the sample's nominal POT)
    that are expensive to compute and do not change between fake-data test
    iterations. The measurement and covariance key selection are deferred
    to ``unfold()``, which is called once per FDT iteration.

    All internal arrays are in absolute event-count units at ``mcbnb_pot``
    (``weights_mc`` summed, no flux division), consistent with
    :class:`~nueana.classes.SystematicsOutput` and
    :func:`make_fake_data_hists`.  Cross-section unit conversion is deferred
    to ``unfold()`` via ``xsec_scale``, whose default resolves to true
    cm²·nucleon⁻¹ using the stored ``mcbnb_pot``.

    Parameters
    ----------
    response : np.ndarray, shape (n_reco, n_true)
        Response matrix from get_response_matrix.
    cv_signal : np.ndarray, shape (n_true,)
        Central-value signal prediction in absolute event-count units at mcbnb_pot.
    syst_covs : dict of str -> np.ndarray
        Per-key covariance matrices in event-count² units at mcbnb_pot, from
        syst_output.xsec_syst_dict.
    mcbnb_pot : float
        Nominal POT of the MC sample; used to compute the default xsec_scale
        in :meth:`unfold` so the unfolded result is in cm²·nucleon⁻¹.
    syst_covs_1bin : dict of str -> np.ndarray or None
        Pre-built 1×1 covariance matrices for single-bin unfolding, keyed by
        the same names as ``syst_covs``.  For xsec (GENIE) knobs these are
        derived from the response-matrix single-bin variance computed in
        ``get_syst_hists`` — not from ``sum(syst_covs[k])``, which would
        integrate the multi-bin smeared covariance and give a different result.
        Populated automatically by :meth:`build` from ``single_bin_unc`` in
        ``xsec_syst_dict``.  When None, :meth:`integrate` falls back to
        ``sum(C)`` for all keys.
    """
    response:       np.ndarray
    cv_signal:      np.ndarray
    syst_covs:      dict
    mcbnb_pot:      float
    syst_covs_1bin: dict | None = None

    def unfold(
        self,
        wienersvd_fn: Callable,
        measure: np.ndarray,
        allowed_keys: tuple[str, ...] | None = None,
        extra_cov: np.ndarray | None = None,
        xsec_scale: float | None = None,
        total_cov: np.ndarray | None = None,
        c_type: int = 2,
        norm_type: float = 0.5,
    ) -> dict:
        """Run WienerSVD unfolding.

        Parameters
        ----------
        wienersvd_fn : callable
            The WienerSVD function imported from cafpyana in the notebook.
        measure : np.ndarray, shape (n_reco,)
            Background-subtracted measurement in absolute event-count units at mcbnb_pot.
        allowed_keys : tuple of str or None, optional
            Systematic categories to include (e.g. ``('GENIE', 'MCstat')``). None
            includes all. Ignored when ``total_cov`` is provided.
        extra_cov : np.ndarray or None, optional
            Extra covariance in event-count² units added after summing syst_covs.
            Use for data/fake-data statistical uncertainty.
        xsec_scale : float, optional
            Conversion from event-count units to cross-section units. Defaults to
            ``1 / (integrated_flux * mcbnb_pot * NTARGETS)`` (true cm²·nucleon⁻¹).
            Pass ``1.0`` to keep event-count units. Stamped on the returned dict.
        total_cov : np.ndarray or None, optional
            Pre-built covariance replacing syst_covs summation (e.g. ``ccbc_cov['cov_ms_ms']``).
        c_type : int, optional
            WienerSVD smoothness matrix type. Default 2 (2nd derivative).
        norm_type : float, optional
            WienerSVD signal normalisation exponent. Default 0.5.

        Returns
        -------
        dict
            WienerSVD output with keys 'unfold', 'AddSmear', 'WF', 'UnfoldCov',
            'CovRotation', and 'xsec_scale'.  Note: 'WF' is a 1-D array of
            per-singular-value filter factors, not a 2-D operator.  Use
            'CovRotation' to propagate covariances through the unfolding.
        """
        if xsec_scale is None:
            xsec_scale = 1.0 / (integrated_flux * self.mcbnb_pot * NTARGETS)
        n_bins = self.cv_signal.shape[0]
        measure = np.asarray(measure)
        if measure.shape != (self.response.shape[0],):
            raise ValueError(
                f"measure has shape {measure.shape} but response expects "
                f"({self.response.shape[0]},) reco bins"
            )
        if total_cov is not None:
            cov = total_cov
        else:
            cov = np.zeros((n_bins, n_bins))
            for k, c in self.syst_covs.items():
                if key_in_allowed(k, allowed_keys):
                    cov += c
        if extra_cov is not None:
            cov = cov + extra_cov
        cov = cov * xsec_scale ** 2
        result = wienersvd_fn(
            Response=self.response,
            Signal=self.cv_signal * xsec_scale,
            Measure=measure * xsec_scale,
            Covariance=cov,
            C_type=c_type,
            Norm_type=norm_type,
        )
        result['xsec_scale'] = xsec_scale
        return result

    @classmethod
    def build(
        cls,
        var: VariableConfig,
        reco_df: pd.DataFrame,
        true_df: pd.DataFrame,
        syst_output: SystematicsOutput,
    ) -> UnfoldInput:
        """Construct an UnfoldInput from a VariableConfig and SystematicsOutput.

        Builds the expensive fixed components once (response matrix, CV signal,
        full per-key covariance dict). Key selection and xsec unit conversion
        are deferred to unfold() at call time.

        Parameters
        ----------
        var : VariableConfig
            Specifies which variable to unfold. Supplies column keys and bins
            for the response matrix and cv_signal histogram.
        reco_df : pd.DataFrame
            Full selected sample with weights_mc and signal columns.
            Signal events (signal == 0) are used for the response matrix.
        true_df : pd.DataFrame
            Truth-level signal-only sample with weights_mc column.
        syst_output : SystematicsOutput
            Output of get_total_cov for this variable. Provides mcbnb_pot used
            to build cv_signal. All entries in xsec_syst_dict are stored
            in event-count² units; key filtering happens in unfold().

        Returns
        -------
        UnfoldInput
        """
        true_df = ensure_lexsorted(true_df, axis=1)

        response = get_response_matrix(reco_df, true_df, var)

        cv_signal = get_hist1d(
            data=true_df[var.var_nu_col],
            bins=var.bins,
            weights=true_df.weights_mc,
        )

        syst_covs = {
            k: entry['cov']
            for k, entry in syst_output.xsec_syst_dict.items()
        }

        # For xsec (GENIE) knobs, single_bin_unc was computed by running the
        # response-matrix method at n_bins=1 in get_syst_hists — this is the
        # correct integrated variance, not sum(multi-bin cov).
        syst_covs_1bin = {
            k: np.array([[entry['single_bin_unc'] ** 2]])
            for k, entry in syst_output.xsec_syst_dict.items()
            if 'single_bin_unc' in entry
        } or None

        return cls(
            response=response,
            cv_signal=cv_signal,
            syst_covs=syst_covs,
            mcbnb_pot=syst_output.mcbnb_pot,
            syst_covs_1bin=syst_covs_1bin,
        )

    def integrate(self) -> UnfoldInput:
        """Return a 1-bin UnfoldInput for single-bin (integrated) unfolding.

        Collapses the multi-bin system to a single true bin (total signal) and
        a single reco bin (total measurement).  The 1×1 response is the total
        signal efficiency::

            eff = sum(R @ cv_signal) / sum(cv_signal)

        For each key, the 1×1 covariance is taken from ``syst_covs_1bin`` when
        available (correct for xsec/GENIE knobs, where the response-matrix
        single-bin variance differs from ``sum(C)``).  Keys absent from
        ``syst_covs_1bin`` fall back to ``1ᵀ C 1 = sum(C)``.

        The returned object can be passed directly to :meth:`unfold` with a
        1-element ``measure`` array and a 1×1 ``total_cov``.
        """
        total_true = float(np.sum(self.cv_signal))
        eff = float(np.sum(self.response @ self.cv_signal)) / total_true
        covs_1bin = {}
        for k, c in self.syst_covs.items():
            if self.syst_covs_1bin is not None and k in self.syst_covs_1bin:
                covs_1bin[k] = self.syst_covs_1bin[k]
            else:
                covs_1bin[k] = np.array([[np.sum(c)]])
        return UnfoldInput(
            response=np.array([[eff]]),
            cv_signal=np.array([total_true]),
            syst_covs=covs_1bin,
            mcbnb_pot=self.mcbnb_pot,
        )


def get_response_matrix(
    reco_df: pd.DataFrame,
    true_df: pd.DataFrame,
    var: VariableConfig,
) -> np.ndarray:
    """Build a response (smearing) matrix normalized column-wise by the truth histogram.

    Parameters
    ----------
    reco_df : pd.DataFrame
        Selected signal events with a weights_mc column. The reco and truth
        columns are read from var.var_evt_reco_col and var.var_evt_truth_col.
    true_df : pd.DataFrame
        Truth-level signal-only sample (e.g. mcsig_df). The truth column is
        read from var.var_nu_col. weights_mc is used if present, otherwise
        uniform weights of 1.0.
    var : VariableConfig
        Variable configuration supplying column keys and bin edges.
        Only variables with a VariableConfig (i.e. cross-section extraction
        variables) should be passed here.

    Returns
    -------
    np.ndarray, shape (n_bins, n_bins)
        R[i, j] = fraction of true-bin-j events reconstructed in reco-bin-i.
    """
    # filter to signal events first — signal is a flat top-level column so
    # no lexsort is needed to access it, and sorting the smaller result is cheaper
    reco_df  = reco_df[reco_df.signal == 0]
    reco_df  = ensure_lexsorted(reco_df, axis=1)
    true_df  = ensure_lexsorted(true_df, axis=1)

    truth_hist = get_hist1d(
        data=true_df[var.var_nu_col],
        bins=var.bins,
        weights=true_df.weights_mc,
    )
    smearing = get_hist2d(
        weights=reco_df.weights_mc,
        x=reco_df[var.var_evt_reco_col],
        y=reco_df[var.var_evt_truth_col],
        bins=var.bins,
    )
    response = np.divide(
        smearing,
        truth_hist,
        out=np.zeros_like(smearing),
        where=truth_hist != 0,
    )
    return response


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


def plot_unfolded_result(
    result: dict,
    var: VariableConfig,
    truths: dict[str, np.ndarray],
    ax: plt.Axes | None = None,
    truth_colors: dict[str, str] | None = None,
    chisq_label_fmt: Callable[[str, float, int], str] | None = None,
    show_norm_band: bool = True,
    norm_band_ref: str | None = None,
    norm_band_label: str = "norm. uncertainty",
    data_label: str = "unfolded (fake) data",
    ylabel: str | None = None,
    stat_cov: np.ndarray | None = None,
) -> dict:
    """Plot an unfolded measurement with smeared-truth overlays and a normalisation band.

    Parameters
    ----------
    result : dict
        WienerSVD output with 'unfold', 'UnfoldCov', 'AddSmear', 'WF', 'xsec_scale'.
        Truths must be in absolute event-count units at mcbnb_pot (not pre-scaled).
    var : VariableConfig
        Provides bin edges, display labels, and axis label pieces.
    truths : dict of {label: np.ndarray}
        Truth histograms to overlay. Keys become legend labels with chi^2 annotation.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. Created if None.
    truth_colors : dict of {label: color}, optional
        Per-label colour overrides; unspecified labels cycle through _DEFAULT_TRUTH_COLORS.
    chisq_label_fmt : callable, optional
        ``f(label, chisq, ndof) -> str``. Defaults to :func:`~nueana.funcs.format_chisq_label`.
    show_norm_band : bool, optional
        Decompose UnfoldCov and draw the normalisation component as a gray fill. Default True.
    norm_band_ref : str, optional
        Truth key used as reference for decompose_cov. Default is the first key in truths.
    norm_band_label : str, optional
        Legend label for the norm band. Pass empty string to hide from legend.
    data_label : str, optional
        Legend label for the unfolded errorbar.
    ylabel : str, optional
        Y-axis label. None derives a dσ/d<var> label; empty string skips it.
    stat_cov : np.ndarray or None, optional
        Pre-unfolding statistical covariance in event-count² units at mcbnb_pot (e.g.
        ``data_stat_energy`` passed as ``extra_cov`` to ``unfold()``).  When provided,
        the errorbar is split: an outer bar (stat+syst, larger caps, thin line)
        and an inner bar (stat-only, smaller caps, carries the legend label).
        When None a single total-error bar is drawn.

    Returns
    -------
    dict with keys ``ax``, ``chisq`` ({label: float}), ``ndof``, ``cov_norm``, ``cov_shape``.
    """
    if not truths:
        raise ValueError("plot_unfolded_result requires at least one truth hist.")

    xsec_scale = result['xsec_scale']
    unfold     = result['unfold']
    cov        = result['UnfoldCov']
    smear      = result['AddSmear']

    bins             = var.bins
    nbins            = len(bins) - 1
    centers, widths  = bin_geometry(var)

    if ax is None:
        _, ax = plt.subplots()
    if chisq_label_fmt is None:
        chisq_label_fmt = format_chisq_label
    if truth_colors is None:
        truth_colors = {}

    # Unfolded data point — one bar (total) or two bars (stat / stat+syst)
    y          = unfold / widths
    total_err  = np.sqrt(np.clip(np.diag(cov), 0.0, None)) / widths
    if stat_cov is not None:
        # CovRotation is the full (n_true x n_reco) unfolding operator; WF is
        # only the 1D diagonal factors in SVD space and cannot be used directly.
        cov_rot = np.asarray(result['CovRotation'])
        stat_unfolded_cov = cov_rot @ (np.asarray(stat_cov) * xsec_scale ** 2) @ cov_rot.T
        stat_err = np.sqrt(np.clip(np.diag(stat_unfolded_cov), 0.0, None)) / widths
        # Outer bar: total (stat+syst), thin line with caps
        ax.errorbar(centers, y, yerr=total_err,
                    fmt='k.', capsize=4, lw=1, capthick=1)
        # Inner bar: stat-only, thicker line — dominates the overlap region
        # so the subpixel misalignment between the two lw values is invisible
        ax.errorbar(centers, y, yerr=stat_err,
                    fmt='k.', capsize=2, lw=1, capthick=1, label=data_label)
    else:
        ax.errorbar(centers, y, yerr=total_err, fmt='k.', label=data_label)

    # Truth overlays
    chisqs: dict[str, float] = {}
    for i, (label, truth_hist) in enumerate(truths.items()):
        smeared = smear @ truth_hist * xsec_scale
        chisq   = chi_squared(diff=smeared - unfold, cov=cov)
        chisqs[label] = float(chisq)
        color = truth_colors.get(
            label,
            _DEFAULT_TRUTH_COLORS[i % len(_DEFAULT_TRUTH_COLORS)],
        )
        ax.stairs(
            smeared / widths, bins, lw=1.5, color=color, alpha=0.9,
            label=chisq_label_fmt(label, chisq, nbins),
        )

    cov_norm = cov_shape = None
    if show_norm_band:
        ref_key   = norm_band_ref if norm_band_ref is not None else next(iter(truths))
        ref_smear = smear @ truths[ref_key] * xsec_scale
        cov_norm, cov_shape = decompose_cov(cov, ref_smear)
        ax.stairs(
            np.diag(np.sqrt(cov_norm)) / widths, bins,
            fill=True, color='gray', alpha=0.3,
            label=norm_band_label if norm_band_label else None,
        )

    ax.set_xticks(bins)
    ax.set_xticklabels(var.bin_labels)
    ax.set_xlabel(var.var_labels[0],fontsize=8)
    if ylabel is None:
        ylabel = _xsec_ylabel(var,)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)
    ax.legend(frameon=False,fontsize=7)

    return {
        "ax":        ax,
        "chisq":     chisqs,
        "ndof":      nbins,
        "cov_norm":  cov_norm,
        "cov_shape": cov_shape,
    }


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
