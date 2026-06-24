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
]


_DEFAULT_TRUTH_COLORS = ["C0", "yellowgreen", "C3", "C4", "C5"]


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
    """
    response:  np.ndarray
    cv_signal: np.ndarray
    syst_covs: dict
    mcbnb_pot: float
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
            'CovRotation', and 'xsec_scale'.
        """
        if xsec_scale is None:
            xsec_scale = 1.0 / (integrated_flux * self.mcbnb_pot * NTARGETS)
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

        return cls(
            response=response,
            cv_signal=cv_signal,
            syst_covs=syst_covs,
            mcbnb_pot=syst_output.mcbnb_pot,
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
) -> dict:
    """Plot an unfolded measurement with smeared-truth overlays and a normalisation band.

    Parameters
    ----------
    result : dict
        WienerSVD output with 'unfold', 'UnfoldCov', 'AddSmear', 'xsec_scale'.
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
    ax.set_xticklabels(var.bin_labels)
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
