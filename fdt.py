"""Fake-data test (FDT) helpers: response matrix construction and fake-data histogram building.

These utilities handle the cafpyana-free layer of an FDT workflow. The WienerSVD
unfolding call and any generator-level plotting remain in the notebook, since they
cross the cafpyana boundary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from collections.abc import Callable

from .utils import get_hist1d, get_hist2d, ensure_lexsorted
from .classes import VariableConfig, SystematicsOutput
from .syst import key_in_allowed, decompose_cov
from .analysis import integrated_flux, NTARGETS
from .funcs import chi_squared

__all__ = [
    'UnfoldInput',
    'get_response_matrix',
    'make_fake_data_hists',
    'plot_unfolded_result',
]


_DEFAULT_TRUTH_COLORS = ["C0", "yellowgreen", "C3", "C4", "C5"]


def _default_chisq_label(label: str, chisq: float, ndof: int) -> str:
    base = r"$A_C \otimes$ " + label + "\n" + rf"$\chi^2$/dof={chisq:.1f}/{ndof}"
    try:
        from scipy.stats import chi2 as _chi2
        pval = 1.0 - _chi2.cdf(chisq, df=ndof)
        return base + f", $p$={pval:.2g}"
    except Exception:
        return base


# ---------------------------------------------------------------------------


@dataclass
class UnfoldInput:
    """Pre-built inputs to a WienerSVD unfolding call.

    Stores the fixed components (response matrix, CV signal prediction, and
    per-key systematic covariance matrices) that are expensive to compute and
    do not change between fake-data test iterations. The measurement and
    covariance key selection are deferred to ``unfold()``, which is called
    once per FDT iteration.

    All internal arrays are in flux-averaged event-rate units
    (``weights_mc / (integrated_flux * mcbnb_pot)``), consistent with
    :class:`~nueana.classes.SystematicsOutput` and
    :func:`make_fake_data_hists`.  Cross-section unit conversion is deferred
    to ``unfold()`` via ``xsec_scale``, so the same object can be reused
    across FDT iterations and the final data unfolding.

    Parameters
    ----------
    response : np.ndarray, shape (n_reco, n_true)
        Response matrix from get_response_matrix.
    cv_signal : np.ndarray, shape (n_true,)
        Central-value signal prediction in flux-averaged units.
    syst_covs : dict of str -> np.ndarray
        Per-key covariance matrices in flux-averaged squared units, from
        syst_output.xsec_syst_dict.
    """
    response:  np.ndarray
    cv_signal: np.ndarray
    syst_covs: dict

    def unfold(
        self,
        wienersvd_fn: Callable,
        measure: np.ndarray,
        allowed_keys: tuple[str, ...] | None = None,
        extra_cov: np.ndarray | None = None,
        xsec_scale: float = 1.0 / NTARGETS,
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
            Background-subtracted measurement in flux-averaged units (output
            of :func:`make_fake_data_hists`).
        allowed_keys : tuple of str or None, optional
            Categories to include, matched via the same classification logic as
            syst.py (e.g. ``('GENIE', 'MCstat')``). Each key in syst_covs is
            classified into a category ('GENIE', 'Flux', 'MCstat', 'DetVar',
            'Geant4') and included only if its category appears in allowed_keys.
            GENIE aliases (SBNNuSyst, SuSAv2) are handled correctly. None
            (default) includes all keys. Ignored when total_cov is provided.
        extra_cov : np.ndarray or None, optional
            Additional covariance matrix in flux-averaged squared units, added
            after summing syst_covs and applying xsec_scale². Use for
            fake-data or data statistical uncertainty.
        xsec_scale : float, optional
            Scale factor applied to convert from flux-averaged event-rate units
            to cross-section units. Applied as ``xsec_scale`` to Signal and
            Measure and ``xsec_scale²`` to the covariance. Defaults to
            ``1 / NTARGETS`` (cross-section per nucleon). Pass ``1.0`` to keep
            flux-averaged event-rate units.
        total_cov : np.ndarray or None, optional
            Pre-built covariance matrix in flux-averaged squared units that
            replaces the syst_covs summation entirely. Use to pass
            ``ccbc_dict['cov_ms_ms']`` directly — its allowed-key selection
            was already applied inside ``get_ccbc_cov``. allowed_keys is
            ignored.
        c_type : int, optional
            WienerSVD smoothness matrix type (0=unit, 1=1st deriv, 2=2nd deriv,
            3=3rd deriv). Default 2.
        norm_type : float, optional
            WienerSVD signal normalisation exponent. Default 0.5.

        Returns
        -------
        dict
            WienerSVD output dict with keys 'unfold', 'AddSmear', 'WF',
            'UnfoldCov', 'CovRotation'.
        """
        n_bins = self.cv_signal.shape[0]
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
        return wienersvd_fn(
            Response=self.response,
            Signal=self.cv_signal * xsec_scale,
            Measure=measure * xsec_scale,
            Covariance=cov,
            C_type=c_type,
            Norm_type=norm_type,
        )

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
            Output of get_total_cov for this variable. Provides mcbnb_pot for
            flux-averaging cv_signal. All entries in xsec_syst_dict are stored
            in flux-averaged squared units; key filtering happens in unfold().

        Returns
        -------
        UnfoldInput
        """
        if syst_output.mcbnb_pot is None:
            raise ValueError(
                "syst_output.mcbnb_pot is None — cannot flux-average cv_signal. "
                "Pass mcbnb_pot to get_total_cov when building syst_output."
            )
        flux_norm = integrated_flux * syst_output.mcbnb_pot
        true_df = ensure_lexsorted(true_df, axis=1)

        response = get_response_matrix(reco_df, true_df, var)

        cv_signal = get_hist1d(
            data=true_df[var.var_nu_col],
            bins=var.bins,
            weights=true_df.weights_mc / flux_norm,
        )

        syst_covs = {
            k: entry['cov']
            for k, entry in syst_output.xsec_syst_dict.items()
        }

        return cls(
            response=response,
            cv_signal=cv_signal,
            syst_covs=syst_covs,
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
    mcbnb_pot: float,
    side_df: pd.DataFrame | None = None,
    ccbc_cov: dict | None = None,
    side_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    """Build background-subtracted fake-data and modified signal histograms for an FDT.

    All returned histograms are in flux-averaged event-rate units
    (``weights_mc / (integrated_flux * mcbnb_pot)``), consistent with the
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
    mcbnb_pot : float
        Total POT of the MC sample (e.g. from
        ``syst_output.mcbnb_pot``).  Used to flux-average all histograms.
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
        Background-subtracted fake measurement in flux-averaged units.
    fd_true : np.ndarray
        Modified truth-level signal histogram in flux-averaged units.
    fd_ns : np.ndarray
        Total (signal + background) fake-data signal-region histogram in
        flux-averaged units, before background subtraction.  Plays the role
        of observed data (``D_S``) in :func:`~nueana.ccbc.plot_ccbc_fd_comparison`.
    fd_nc : np.ndarray or None
        Control-region fake-data histogram in flux-averaged units, or
        ``None`` when ``side_df`` is not provided.
    fd_Bs : np.ndarray
        True reweighted background histogram in flux-averaged units — the
        background-event (``signal != 0``) contribution under the FDT weights.
        Distinct from the CV background (no reweighting) and from the
        CCBC-constrained prediction; use as the "ground-truth background" overlay
        in :func:`~nueana.ccbc.plot_ccbc_fd_comparison`.
    """
    flux_norm = integrated_flux * mcbnb_pot

    reco_df = ensure_lexsorted(reco_df, axis=1)
    true_df = ensure_lexsorted(true_df, axis=1)

    reco_weights = reco_df.weights_mc.values.copy() / flux_norm
    reco_weights[reco_mask] *= weight

    true_weights = true_df.weights_mc.values.copy() / flux_norm
    true_weights[true_mask] *= weight

    fd_ns = get_hist1d(
        data=reco_df[var.var_evt_reco_col], bins=var.bins, weights=reco_weights,
    )
    fd_true = get_hist1d(
        data=true_df[var.var_nu_col], bins=var.bins, weights=true_weights,
    )
    fd_Bs = get_hist1d(
        data=reco_df[var.var_evt_reco_col], bins=var.bins,
        weights=np.where(reco_df.signal.values != 0, reco_weights, 0),
    )

    if side_df is not None and ccbc_cov is not None:
        from .ccbc import get_constrained_background
        side_df = ensure_lexsorted(side_df, axis=1)
        side_weights = side_df.weights_mc.values.copy() / flux_norm
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
        weights=backgr_df.weights_mc / flux_norm,
    )
    fd_meas = fd_ns - cv_backgr_hist
    return fd_meas, fd_true, fd_ns, None, fd_Bs


# ---------------------------------------------------------------------------


def plot_unfolded_result(
    result: dict,
    var: VariableConfig,
    truths: dict[str, np.ndarray],
    ax: plt.Axes | None = None,
    xsec_scale: float = 1.0 / NTARGETS,
    truth_colors: dict[str, str] | None = None,
    chisq_label_fmt: Callable[[str, float, int], str] | None = None,
    show_norm_band: bool = True,
    norm_band_ref: str | None = None,
    norm_band_label: str = "norm. uncertainty",
    data_label: str = "unfolded (fake) data",
    ylabel: str | None = None,
) -> dict:
    """Plot an unfolded measurement with smeared-truth overlays, chi^2 labels,
    and a normalisation-uncertainty band.

    Encapsulates the cell 13/14 pattern: errorbar of the unfolded result
    divided by display bin widths, one stairs per truth (each scaled by
    ``AddSmear @ truth * xsec_scale`` and divided by widths) with a chi^2
    annotation in the legend, plus an optional gray fill_between for the
    normalisation-only component of UnfoldCov.

    All histograms are plotted on ``var.bins`` positions but divided by
    ``np.diff(var.bin_labels)`` for the y-axis density (so an overflow bin
    with a display label far past its edge appears at its natural width on
    the legend axis).

    Parameters
    ----------
    result : dict
        WienerSVD output containing ``'unfold'``, ``'UnfoldCov'``, ``'AddSmear'``.
    var : VariableConfig
        Provides bin edges, display labels, and axis label pieces.
    truths : dict of {label: np.ndarray}
        At least one truth-bin histogram to overlay, keyed by legend label.
        Each truth is scaled by ``AddSmear @ truth * xsec_scale`` and rendered
        as a stairs plot. The keys are also used as the chi^2 legend label
        prefix via ``chisq_label_fmt``.
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes to draw on. A new figure+axes is created if None.
    xsec_scale : float, optional
        Scale factor applied to each ``AddSmear @ truth`` to bring it into the
        same units as ``result['unfold']``. Defaults to ``1 / NTARGETS``
        (cross-section per nucleon, matching the default in
        :meth:`UnfoldInput.unfold`).
    truth_colors : dict of {label: color}, optional
        Per-label colour overrides. Unspecified labels fall back to
        ``_DEFAULT_TRUTH_COLORS`` cycled in dict-insertion order.
    chisq_label_fmt : callable, optional
        ``f(label, chisq, ndof) -> legend_str``. Defaults to a two-line
        format with ``A_C ⊗`` prefix, chi^2/dof, and p-value.
    show_norm_band : bool, optional
        If True, decompose ``UnfoldCov`` around the first truth's smeared
        prediction (or ``norm_band_ref``) and draw the diagonal-sqrt of the
        normalisation component as a gray fill. Default True.
    norm_band_ref : str, optional
        Truth key whose smeared prediction is used as the reference for
        ``decompose_cov``. Default is the first key in ``truths``.
    norm_band_label : str, optional
        Legend label for the normalisation-uncertainty band. Pass ``None`` (or
        an empty string) to keep the band but omit it from the legend.
    data_label : str, optional
        Legend label for the unfolded errorbar.
    ylabel : str, optional
        Y-axis label. ``None`` (default) derives a differential cross-section
        label from ``var`` (``d sigma/d<var> [cm^2 / <unit> / nucleon]``,
        assuming ``xsec_scale = 1 / NTARGETS``). Pass a string to override, or
        an empty string to skip setting the label.

    Returns
    -------
    dict with keys:
        ``ax``       — the matplotlib Axes object.
        ``chisq``    — ``{truth_label: float}`` chi^2 values.
        ``ndof``     — number of bins (used as the chi^2 ndof).
        ``cov_norm`` — normalisation-component covariance (None if disabled).
        ``cov_shape``— shape-component covariance (None if disabled).
    """
    if not truths:
        raise ValueError("plot_unfolded_result requires at least one truth hist.")

    unfold = result['unfold']
    cov    = result['UnfoldCov']
    smear  = result['AddSmear']

    bins       = var.bins
    bin_labels = var.bin_labels
    centers    = 0.5 * (bins[:-1] + bins[1:])
    widths     = np.diff(bin_labels)
    nbins      = len(bins) - 1

    if ax is None:
        _, ax = plt.subplots()
    if chisq_label_fmt is None:
        chisq_label_fmt = _default_chisq_label
    if truth_colors is None:
        truth_colors = {}

    # Unfolded data point with diagonal errors
    ax.errorbar(
        centers, unfold / widths,
        yerr=np.sqrt(np.diag(cov)) / widths,
        fmt='k.', label=data_label,
    )

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
            smeared / widths, bins, lw=2, color=color,
            label=chisq_label_fmt(label, chisq, nbins),
        )

    cov_norm = cov_shape = None
    if show_norm_band:
        ref_key   = norm_band_ref if norm_band_ref is not None else next(iter(truths))
        ref_smear = smear @ truths[ref_key] * xsec_scale
        cov_norm, cov_shape = decompose_cov(cov, ref_smear)
        ax.stairs(
            np.diag(np.sqrt(cov_norm)) / widths, bins,
            fill=True, color='gray', alpha=0.5,
            label=norm_band_label if norm_band_label else None,
        )

    ax.set_xticks(bins)
    ax.set_xticklabels(bin_labels)
    ax.set_xlabel(var.var_labels[0],fontsize=12)
    if ylabel is None:
        plot_math = var.var_plot_name.strip("$")
        unit_part = (
            rf"\,\mathrm{{{var.var_unit}}}^{{-1}}" if var.var_unit else ""
        )
        ylabel = (
            rf"$d\sigma / d{plot_math}$ "
            rf"$[\mathrm{{cm}}^2{unit_part}\,\mathrm{{nucleon}}^{{-1}}]$"
        )
    if ylabel:
        ax.set_ylabel(ylabel,fontsize=12)
    ax.legend()

    return {
        "ax":        ax,
        "chisq":     chisqs,
        "ndof":      nbins,
        "cov_norm":  cov_norm,
        "cov_shape": cov_shape,
    }
