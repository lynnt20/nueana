"""Control-Region Background Constraint (CCBC) covariance and diagnostics.

CCBC uses correlated universe histograms from the sideband (control region) to
constrain background uncertainty in the signal region. The workflow is:

  1. Call ``get_total_cov`` four times — once each for the background-only sample
     (Bs), the full signal-region sample (ns), the signal-only sample (Ps), and
     the control-region sample (nc).
  2. Pass those four outputs to ``get_ccbc_cov`` to obtain all block covariance
     matrices and the constraint-reduced versions.
  3. Use ``plot_ccbc_constraint`` and ``plot_ccbc_blocks`` for diagnostics.
"""
from __future__ import annotations

import numpy as np
from collections.abc import Sequence
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import seaborn as sns

from .classes import SystematicsOutput, VariableConfig
from .funcs import get_corr_from_cov, get_fractional_covariance

__all__ = [
    "nonsymmetric_cov",
    "get_ccbc_cov",
    "get_chisq_diff",
    "get_chisq",
    "plot_ccbc_constraint",
    "plot_ccbc_blocks",
]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _repeat(arr: np.ndarray) -> np.ndarray:
    """Prepend arr[0] so step-plot fill_between covers the leftmost bin edge."""
    return np.append(arr[0], arr)


def _construct_full_cov(
    left: np.ndarray,
    right: np.ndarray,
    lr: np.ndarray,
) -> np.ndarray:
    """Assemble a symmetric block covariance from two auto-cov blocks and one cross block."""
    return np.block([[left, lr], [lr.T, right]])


def _annotate_block_axes(axes, nbins: int, var: str) -> None:
    """Draw the B_S / n_C dividing line and region labels on block-matrix heatmap axes."""
    for ax in axes:
        ax.plot([nbins, nbins], [0, 2 * nbins], "w-", lw=2)
        ax.plot([0, 2 * nbins], [nbins, nbins], "w-", lw=2)
        ax.invert_yaxis()
        tick_labels = list(range(nbins)) * 2
        ax.set_xticklabels(tick_labels, size=8)
        ax.set_yticklabels(tick_labels, size=8)
        for offset, label in [
            (-60, f"$B_S$ {var} bins"),
            (60,  f"$n_C$ {var} bins"),
        ]:
            ax.annotate(
                label, xy=(0.25, 0.5),
                xytext=(-ax.yaxis.labelpad + 2, offset),
                xycoords=ax.yaxis.label, textcoords="offset points",
                size=10, ha="right", va="center", rotation=90,
            )
        for offset, label in [
            (-20, f"$B_S$ {var} bins"),
            (95,  f"$n_C$ {var} bins"),
        ]:
            ax.annotate(
                label, xy=(0.25, 0.5),
                xytext=(-ax.xaxis.labelpad + offset, -5),
                xycoords=ax.xaxis.label, textcoords="offset points",
                size=10, ha="right", va="center",
            )


# ---------------------------------------------------------------------------
# Core covariance functions
# ---------------------------------------------------------------------------


def nonsymmetric_cov(
    left_univ: np.ndarray,
    left_cv: np.ndarray,
    right_univ: np.ndarray,
    right_cv: np.ndarray,
    ddof: int = 0,
) -> np.ndarray:
    """Cross-covariance between two sets of universe histograms.

    Computes ``(left_diff @ right_diff.T) / (N_univ - ddof)`` where
    ``left_diff[i, u] = left_univ[i, u] - left_cv[i]``.  The result is an
    ``(nbins_left, nbins_right)`` matrix that is not generally symmetric, which
    is why a standard ``np.cov`` call cannot be used here.

    Parameters
    ----------
    left_univ : shape (nbins_left, nuniv)
    left_cv   : shape (nbins_left,)
    right_univ: shape (nbins_right, nuniv)
    right_cv  : shape (nbins_right,)
    ddof      : delta degrees of freedom; 0 (default) gives the MLE estimator.

    Returns
    -------
    np.ndarray, shape (nbins_left, nbins_right)
    """
    left_univ  = np.asarray(left_univ)
    right_univ = np.asarray(right_univ)
    left_cv    = np.asarray(left_cv)
    right_cv   = np.asarray(right_cv)

    if left_univ.ndim != 2 or right_univ.ndim != 2:
        raise ValueError("Universe arrays must be 2-D with shape (nbins, nuniv)")
    if left_univ.shape[1] != right_univ.shape[1]:
        raise ValueError("left_univ and right_univ must have the same nuniv")
    if left_univ.shape[0] != left_cv.shape[0]:
        raise ValueError("left_cv length must match left_univ nbins")
    if right_univ.shape[0] != right_cv.shape[0]:
        raise ValueError("right_cv length must match right_univ nbins")

    nuniv = left_univ.shape[1]
    norm  = nuniv - ddof
    if norm <= 0:
        raise ValueError(f"Invalid normalization: nuniv={nuniv}, ddof={ddof}")

    left_diff  = left_univ  - left_cv[:, None]
    right_diff = right_univ - right_cv[:, None]
    return (left_diff @ right_diff.T) / norm


def get_ccbc_cov(
    Bs_output: SystematicsOutput,
    Ps_output: SystematicsOutput,
    nc_output: SystematicsOutput,
    allowed_keys: Sequence[str] = ("GENIE", "Flux", "Geant4", "MCstat"),
    rcond: float = 1e-10,
) -> dict[str, np.ndarray]:
    """Assemble CCBC block covariance matrices and apply the sideband constraint.

    Iterates over all universe systematics in ``nc_output.rate_syst_dict`` whose
    key contains any string from ``allowed_keys``, accumulates six covariance
    blocks via :func:`nonsymmetric_cov`, then computes the constraint-reduced
    background and total-rate covariances.

    The constraint formula is:

    .. math::

        C_{B_S B_S}^{\\rm constr} = C_{B_S B_S}
            - C_{B_S n_C}\\, C_{n_C n_C}^{+}\\, C_{B_S n_C}^T

        C_{P_S B_S}^{\\rm constr} = C_{P_S B_S}
            - C_{P_S n_C}\\, C_{n_C n_C}^{+}\\, C_{B_S n_C}^T

        C_{m_S m_S} = C_{P_S P_S}
            + C_{P_S B_S}^{\\rm constr}
            + (C_{P_S B_S}^{\\rm constr})^T
            + C_{B_S B_S}^{\\rm constr}

    where :math:`C^+` denotes the Moore–Penrose pseudoinverse.

    Parameters
    ----------
    Bs_output : SystematicsOutput
        ``get_total_cov`` result for background-only signal-region events
        (``event_type='background'``).
    Ps_output : SystematicsOutput
        ``get_total_cov`` result for signal-only signal-region events
        (``event_type='signal'``, ``xsec_inputs`` required so that
        ``xsec_syst_dict`` is populated).
    nc_output : SystematicsOutput
        ``get_total_cov`` result for control-region events
        (``select_region='control'``).
    allowed_keys : sequence of str
        A universe key is included when any element of this sequence appears as
        a substring of the key string. Defaults to GENIE, Flux, Geant4, MCstat.
    rcond : float
        Regularisation cut-off for ``np.linalg.pinv`` applied to
        ``cov_nc_nc``.  Default 1e-10.

    Returns
    -------
    dict with keys:
        ``cov_Bs_Bs``         — pre-constraint background auto-covariance
        ``cov_Bs_Bs_constr``  — post-constraint background auto-covariance
        ``cov_ns_ns``         — pre-constraint total-rate covariance
        ``cov_ms_ms``         — post-constraint total-rate covariance
        ``cov_Ps_Ps``         — signal-component auto-covariance
        ``cov_Ps_Bs``         — signal × background cross-covariance
        ``cov_Ps_Bs_constr``  — post-constraint signal × background cross-covariance
        ``cov_Bs_nc``         — background × control cross-covariance
        ``cov_Ps_nc``         — signal × control cross-covariance
        ``cov_nc_nc``         — control-region auto-covariance
    """
    nbins  = len(nc_output.rate_hist_cv)
    zeros  = lambda: np.zeros((nbins, nbins))
    cov_Bs_nc = zeros(); cov_nc_nc = zeros(); cov_Ps_nc = zeros()
    cov_Ps_Bs = zeros(); cov_Bs_Bs = zeros(); cov_Ps_Ps = zeros()

    for key in nc_output.rate_syst_dict:
        if not any(k in key for k in allowed_keys):
            continue
        if key not in Bs_output.rate_syst_dict:
            continue
        if Ps_output.xsec_syst_dict is None or key not in Ps_output.xsec_syst_dict:
            continue

        Bs_h  = Bs_output.rate_syst_dict[key]["hists"]
        nc_h  = nc_output.rate_syst_dict[key]["hists"]
        Ps_h  = Ps_output.xsec_syst_dict[key]["hists"]

        Bs_cv = Bs_output.rate_hist_cv
        nc_cv = nc_output.rate_hist_cv
        Ps_cv = Ps_output.rate_hist_cv

        cov_Bs_nc += nonsymmetric_cov(Bs_h, Bs_cv, nc_h, nc_cv)
        cov_Ps_nc += nonsymmetric_cov(Ps_h, Ps_cv, nc_h, nc_cv)
        cov_Ps_Bs += nonsymmetric_cov(Ps_h, Ps_cv, Bs_h, Bs_cv)
        cov_nc_nc += nonsymmetric_cov(nc_h, nc_cv, nc_h, nc_cv)
        cov_Bs_Bs += nonsymmetric_cov(Bs_h, Bs_cv, Bs_h, Bs_cv)
        cov_Ps_Ps += nonsymmetric_cov(Ps_h, Ps_cv, Ps_h, Ps_cv)

    cov_nc_nc_sym    = (cov_nc_nc + cov_nc_nc.T) / 2
    pinv_nc          = np.linalg.pinv(cov_nc_nc_sym, rcond=rcond)
    cov_Bs_Bs_constr = cov_Bs_Bs  - cov_Bs_nc @ pinv_nc @ cov_Bs_nc.T
    cov_Ps_Bs_constr = cov_Ps_Bs  - cov_Ps_nc @ pinv_nc @ cov_Bs_nc.T
    cov_ms_ms        = cov_Ps_Ps  + cov_Ps_Bs_constr + cov_Ps_Bs_constr.T + cov_Bs_Bs_constr
    cov_ns_ns        = cov_Ps_Ps  + cov_Ps_Bs        + cov_Ps_Bs.T        + cov_Bs_Bs

    return {
        "cov_Bs_Bs":         cov_Bs_Bs,
        "cov_Bs_Bs_constr":  cov_Bs_Bs_constr,
        "cov_ns_ns":         cov_ns_ns,
        "cov_ms_ms":         cov_ms_ms,
        "cov_Ps_Ps":         cov_Ps_Ps,
        "cov_Ps_Bs":         cov_Ps_Bs,
        "cov_Ps_Bs_constr":  cov_Ps_Bs_constr,
        "cov_Bs_nc":         cov_Bs_nc,
        "cov_Ps_nc":         cov_Ps_nc,
        "cov_nc_nc":         cov_nc_nc,
    }


# ---------------------------------------------------------------------------
# Chi-squared helpers
# ---------------------------------------------------------------------------


def get_chisq_diff(diff: np.ndarray, cov: np.ndarray) -> float:
    """Chi-squared statistic: ``diff.T @ inv(cov) @ diff``.

    Parameters
    ----------
    diff : np.ndarray, shape (n,)
    cov  : np.ndarray, shape (n, n) — must be invertible

    Returns
    -------
    float
    """
    return float(diff @ np.linalg.inv(cov) @ diff)


def get_chisq(
    syst_result: SystematicsOutput,
    scale: float,
    data_hist: np.ndarray,
) -> float:
    """Chi-squared comparing a scaled CV rate histogram to a data histogram.

    Covariance is ``rate_cov * scale^2 + diag(data_hist)`` — the second term
    adds Poisson data statistics assuming ``data_hist`` is in event counts.

    Parameters
    ----------
    syst_result : SystematicsOutput
    scale       : float  — converts rate_hist_cv to event-count units
    data_hist   : np.ndarray, shape (nbins,) — observed event counts

    Returns
    -------
    float
    """
    cv  = syst_result.rate_hist_cv * scale
    cov = syst_result.rate_cov * scale ** 2 + np.diag(data_hist)
    return get_chisq_diff(cv - data_hist, cov)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_ccbc_constraint(
    cov_list: list[dict],
    cov_stat: dict,
    Bs_hist: np.ndarray,
    ns_hist: np.ndarray,
    var_config: VariableConfig,
    list_keys: list[str],
    colors: list[str],
) -> tuple[plt.Figure, np.ndarray]:
    """Four-row panel comparing pre/post-constraint uncertainty for each systematic group.

    Rows from top to bottom:
        1. B_S event rate with pre- and post-constraint error bars
        2. B_S fractional uncertainty ratio bands
        3. n_S (total) event rate with pre- and post-constraint error bars
        4. n_S fractional uncertainty ratio bands

    A hatched band showing the MC-stat-only uncertainty from ``cov_stat`` is
    overlaid on rows 2 and 4 for reference.

    All histogram and covariance inputs must be in the same event-count units
    (i.e. already scaled by the appropriate flux × POT factor).

    Parameters
    ----------
    cov_list   : list of dicts returned by :func:`get_ccbc_cov`, one per column.
                 Each dict must contain ``cov_Bs_Bs``, ``cov_Bs_Bs_constr``,
                 ``cov_ns_ns``, ``cov_ms_ms`` (in event-count^2 units).
    cov_stat   : single :func:`get_ccbc_cov` dict computed with MCstat only,
                 used for the stat-only hatch overlay.
    Bs_hist    : CV background histogram in event-count units.
    ns_hist    : CV total event-rate histogram in event-count units.
    var_config : VariableConfig for bin edges and axis labels.
    list_keys  : column titles, one per entry in ``cov_list``.
    colors     : post-constraint highlight colour per entry in ``cov_list``.

    Returns
    -------
    fig, axes  where axes has shape (4, ncols).
    """
    ncols   = len(cov_list)
    bins    = var_config.bins
    centers = var_config.bin_centers

    stat_ratio_Bs = np.sqrt(np.diag(cov_stat["cov_Bs_Bs"])) / Bs_hist
    stat_ratio_ns = np.sqrt(np.diag(cov_stat["cov_ns_ns"])) / ns_hist

    fig = plt.figure(figsize=(4 * ncols, 8))
    gs  = GridSpec(4, ncols, height_ratios=[4, 1, 4, 1], hspace=0.4)

    axes_Bs_main  = [fig.add_subplot(gs[0, i]) for i in range(ncols)]
    axes_Bs_ratio = [fig.add_subplot(gs[1, i]) for i in range(ncols)]
    axes_ns_main  = [fig.add_subplot(gs[2, i]) for i in range(ncols)]
    axes_ns_ratio = [fig.add_subplot(gs[3, i]) for i in range(ncols)]

    for i, (cov, color, key) in enumerate(zip(cov_list, colors, list_keys)):
        pre_cov_Bs  = cov["cov_Bs_Bs"]
        post_cov_Bs = cov["cov_Bs_Bs_constr"]
        pre_cov_ns  = cov["cov_ns_ns"]
        post_cov_ns = cov["cov_ms_ms"]

        pre_ratio_Bs  = np.sqrt(np.diag(pre_cov_Bs))  / Bs_hist
        post_ratio_Bs = np.sqrt(np.diag(post_cov_Bs)) / Bs_hist
        pre_ratio_ns  = np.sqrt(np.diag(pre_cov_ns))  / ns_hist
        post_ratio_ns = np.sqrt(np.diag(post_cov_ns)) / ns_hist

        # Row 1: B_S main
        axes_Bs_main[i].stairs(Bs_hist, bins, color="black", label=r"$B_S^{CV}$")
        axes_Bs_main[i].errorbar(centers, Bs_hist, yerr=np.sqrt(np.diag(pre_cov_Bs)),
                                 fmt=".", capsize=3, color="black", label="Pre-constraint")
        axes_Bs_main[i].errorbar(centers, Bs_hist, yerr=np.sqrt(np.diag(post_cov_Bs)),
                                 fmt=".", capsize=3, color=color,  label="Post-constraint")
        axes_Bs_main[i].fill_between(
            x=bins,
            y1=_repeat(Bs_hist - np.sqrt(np.diag(cov_stat["cov_Bs_Bs"]))),
            y2=_repeat(Bs_hist + np.sqrt(np.diag(cov_stat["cov_Bs_Bs"]))),
            step="pre", facecolor="none",
            edgecolor=mpl.colors.to_rgba("gray", 0.5), lw=0, hatch="////",
            label="MC stat only",
        )
        axes_Bs_main[i].set_title(key, fontsize=12)
        axes_Bs_main[i].set_xlabel(var_config.var_labels[1], fontsize=10)
        axes_Bs_main[i].set_xticks(bins)
        axes_Bs_main[i].set_xticklabels(var_config.bin_labels, fontsize=8)

        # Row 2: B_S ratio
        axes_Bs_ratio[i].fill_between(bins, 1 - _repeat(pre_ratio_Bs),
                                      1 + _repeat(pre_ratio_Bs),
                                      step="pre", color="gray", alpha=0.3)
        axes_Bs_ratio[i].fill_between(bins, 1 - _repeat(post_ratio_Bs),
                                      1 + _repeat(post_ratio_Bs),
                                      step="pre", color=color, alpha=0.5)
        axes_Bs_ratio[i].fill_between(bins, 1 - _repeat(stat_ratio_Bs),
                                      1 + _repeat(stat_ratio_Bs),
                                      step="pre", facecolor="none",
                                      edgecolor="gray", hatch="//",
                                      label="MC stat only")
        axes_Bs_ratio[i].axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.7)
        axes_Bs_ratio[i].set_xticks(bins)
        axes_Bs_ratio[i].set_xticklabels(var_config.bin_labels, fontsize=8)

        # Row 3: n_S main
        axes_ns_main[i].stairs(ns_hist, bins, color="black", label=r"$n_S^{CV}$")
        axes_ns_main[i].errorbar(centers, ns_hist, yerr=np.sqrt(np.diag(pre_cov_ns)),
                                 fmt=".", capsize=3, color="black", label="Pre-constraint")
        axes_ns_main[i].errorbar(centers, ns_hist, yerr=np.sqrt(np.diag(post_cov_ns)),
                                 fmt=".", capsize=3, color=color,  label="Post-constraint")
        axes_ns_main[i].fill_between(
            x=bins,
            y1=_repeat(ns_hist - np.sqrt(np.diag(cov_stat["cov_ns_ns"]))),
            y2=_repeat(ns_hist + np.sqrt(np.diag(cov_stat["cov_ns_ns"]))),
            step="pre", facecolor="none",
            edgecolor=mpl.colors.to_rgba("gray", 0.5), lw=0, hatch="////",
            label="MC stat only",
        )
        axes_ns_main[i].set_xlabel(var_config.var_labels[1], fontsize=10)
        axes_ns_main[i].set_xticks(bins)
        axes_ns_main[i].set_xticklabels(var_config.bin_labels, fontsize=8)

        # Row 4: n_S ratio
        axes_ns_ratio[i].fill_between(bins, 1 - _repeat(pre_ratio_ns),
                                      1 + _repeat(pre_ratio_ns),
                                      step="pre", color="gray", alpha=0.3)
        axes_ns_ratio[i].fill_between(bins, 1 - _repeat(post_ratio_ns),
                                      1 + _repeat(post_ratio_ns),
                                      step="pre", color=color, alpha=0.5)
        axes_ns_ratio[i].fill_between(bins, 1 - _repeat(stat_ratio_ns),
                                      1 + _repeat(stat_ratio_ns),
                                      step="pre", facecolor="none",
                                      edgecolor="gray", hatch="//",
                                      label="MC stat only")
        axes_ns_ratio[i].axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.7)
        axes_ns_ratio[i].set_xticks(bins)
        axes_ns_ratio[i].set_xticklabels(var_config.bin_labels, fontsize=8)
        axes_ns_ratio[i].set_xlabel(var_config.var_labels[1])

        axes_Bs_main[i].legend(fontsize=9.5)
        axes_ns_main[i].legend(fontsize=9.5)
        for ax in [axes_Bs_main[i], axes_Bs_ratio[i],
                   axes_ns_main[i], axes_ns_ratio[i]]:
            ax.tick_params(axis="y", labelsize=8)

    axes_Bs_main[0].set_ylabel(r"$B_S$ events")
    axes_Bs_ratio[0].set_ylabel("Ratio")
    axes_ns_main[0].set_ylabel(r"$n_S$ events")
    axes_ns_ratio[0].set_ylabel("Ratio")

    all_axes = np.array([axes_Bs_main, axes_Bs_ratio,
                         axes_ns_main, axes_ns_ratio])
    return fig, all_axes


def plot_ccbc_blocks(
    Bs_output: SystematicsOutput,
    nc_output: SystematicsOutput,
    allowed_keys: Sequence[str],
    var: str = "",
    axes: np.ndarray | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Heatmaps of the (B_S, n_C) block covariance and correlation matrices.

    Accumulates the B_S × B_S, n_C × n_C, and B_S × n_C cross-covariance
    blocks from universe histograms matching ``allowed_keys``, assembles the
    2-block matrix, then renders side-by-side seaborn heatmaps of the fractional
    covariance and the linear correlation coefficient matrix.

    Parameters
    ----------
    Bs_output   : SystematicsOutput for the signal-region background sample.
    nc_output   : SystematicsOutput for the control-region sample.
    allowed_keys: systematic key substrings to include (e.g. ``["GENIE"]``).
    var         : short label appended to axis annotations, e.g. ``"energy"``.
    axes        : optional (2,) array of pre-created Axes; a new figure is
                  created if not provided.

    Returns
    -------
    fig, axes   where axes has shape (2,).
    """
    nbins = len(Bs_output.rate_hist_cv)
    zeros = lambda: np.zeros((nbins, nbins))
    cov_Bs_nc = zeros(); cov_nc_nc = zeros(); cov_Bs_Bs = zeros()

    for key in nc_output.rate_syst_dict:
        if not any(k in key for k in allowed_keys):
            continue
        if key not in Bs_output.rate_syst_dict:
            continue

        Bs_h  = Bs_output.rate_syst_dict[key]["hists"]
        nc_h  = nc_output.rate_syst_dict[key]["hists"]
        Bs_cv = Bs_output.rate_hist_cv
        nc_cv = nc_output.rate_hist_cv

        cov_Bs_nc += nonsymmetric_cov(Bs_h, Bs_cv, nc_h, nc_cv)
        cov_Bs_Bs += nonsymmetric_cov(Bs_h, Bs_cv, Bs_h, Bs_cv)
        cov_nc_nc += nonsymmetric_cov(nc_h, nc_cv, nc_h, nc_cv)

    combined_hist     = np.concatenate((Bs_output.rate_hist_cv, nc_output.rate_hist_cv))
    combined_cov      = _construct_full_cov(cov_Bs_Bs, cov_nc_nc, cov_Bs_nc)
    combined_frac_cov = get_fractional_covariance(combined_cov, combined_hist)
    combined_corr     = get_corr_from_cov(combined_cov)

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        plt.subplots_adjust(wspace=0.3)
    else:
        fig = axes[0].get_figure()

    sns.heatmap(combined_frac_cov, cmap="mako", ax=axes[0],
                cbar_kws={"label": "Fractional Covariance"})
    sns.heatmap(combined_corr, cmap="Spectral", ax=axes[1],
                annot=True, fmt=".2f", annot_kws={"fontsize": 8},
                cbar_kws={"label": "Correlation"}, vmin=-1, vmax=1)

    _annotate_block_axes(axes, nbins=nbins, var=var)
    return fig, axes
