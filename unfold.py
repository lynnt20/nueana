"""WienerSVD unfolding helpers: response matrix, UnfoldInput container, and result plotting.

These utilities cover the cafpyana-free layer of the unfolding workflow. The
WienerSVD function itself is imported from cafpyana in the notebook.
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
    'plot_unfolded_result',
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
