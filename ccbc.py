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

import warnings
import numpy as np
from collections.abc import Sequence
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import seaborn as sns

from .classes import SystematicsOutput, VariableConfig
from .funcs import get_corr_from_cov, get_fractional_covariance
from .syst import key_in_allowed

__all__ = [
    "nonsymmetric_cov",
    "get_ccbc_cov",
    "get_constrained_background",
    "get_chisq_diff",
    "get_chisq",
    "plot_ccbc_summary",
    "plot_ccbc_fd_comparison",
    "plot_ccbc_constraint",
    "plot_ccbc_blocks",
    "plot_ccbc_key_correlations",
]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _warn_key_mismatch(ref_keys: set, other_keys: set, ref_name: str, other_name: str) -> None:
    only_ref   = ref_keys - other_keys
    only_other = other_keys - ref_keys
    if only_ref or only_other:
        msg = f"Systematic universe key mismatch between {ref_name} and {other_name}."
        if only_ref:
            msg += f"\n  Only in {ref_name}: {sorted(only_ref)}"
        if only_other:
            msg += f"\n  Only in {other_name}: {sorted(only_other)}"
        warnings.warn(msg, stacklevel=3)

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


def _iter_shared_keys(nc_dict, Bs_dict, allowed_keys, extra_dict=None):
    """Yield keys present in nc_dict and Bs_dict that pass key_in_allowed.

    When extra_dict is provided, also requires key in extra_dict.
    """
    for key in nc_dict:
        if not key_in_allowed(key, allowed_keys):
            continue
        if key not in Bs_dict:
            continue
        if extra_dict is not None and key not in extra_dict:
            continue
        yield key


def _rescale_detvar(h, cv, output, key):
    """Rescale DetVar hists from events/sample_pot to events at mcbnb_pot."""
    if key_in_allowed(key, ("DetVar",)):
        return h * output.mcbnb_pot, cv * output.mcbnb_pot
    return h, cv


def _norm_pct(cov: np.ndarray, hist: np.ndarray) -> float:
    """Fractional normalization uncertainty in percent: sqrt(sum(cov)) / sum(hist) * 100."""
    return float(np.sqrt(np.sum(cov)) / np.sum(hist) * 100)


def _chisq_str(chisq: float, sub: str, ndof: int) -> str:
    """Format χ²/dof legend label with subscript notation and p-value when scipy is available."""
    base = rf"$\chi^2_{{\rm {sub}}}$/dof = {chisq:.1f}/{ndof}"
    try:
        from scipy.stats import chi2 as _chi2
        p = 1.0 - _chi2.cdf(chisq, df=ndof)
        return base + rf", $p$ = {p:.2g}"
    except Exception:
        return base


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
    ns_output: SystematicsOutput | None = None,
    allowed_keys: Sequence[str] = ("GENIE", "Flux", "Geant4", "MCstat"),
    rcond: float = 1e-10,
    data_stat_nc: np.ndarray | None = None,
    data_stat_ns: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Assemble CCBC block covariance matrices and apply the sideband constraint.

    The constraint formula is:

    .. math::

        C_{B_S B_S}^{\\rm constr} = C_{B_S B_S}
            - C_{B_S n_C}\\, C_{n_C n_C}^{+}\\, C_{B_S n_C}^T

        C_{m_S m_S} = C_{P_S P_S}
            + C_{P_S B_S}^{\\rm constr}
            + (C_{P_S B_S}^{\\rm constr})^T
            + C_{B_S B_S}^{\\rm constr}

    P_S uses ``xsec_syst_dict`` (response-matrix universe histograms); B_S and
    n_C use ``rate_syst_dict``.  DetVar entries are rescaled from events/sample_pot
    to events at mcbnb_pot before covariance accumulation.

    Parameters
    ----------
    Bs_output : SystematicsOutput
        Background-only signal-region sample (``event_type='background'``).
    Ps_output : SystematicsOutput
        Signal-only signal-region sample (``xsec_syst_dict`` must be populated).
    nc_output : SystematicsOutput
        Control-region sample (``select_region='control'``).
    ns_output : SystematicsOutput, optional
        Total signal-region sample used to cross-check cov_ns_ns consistency.
        A warning is raised when the relative element-wise difference exceeds 1e-6.
    allowed_keys : sequence of str
        Keys included when any element appears as a substring. Default: GENIE, Flux,
        Geant4, MCstat.
    rcond : float
        Regularisation cut-off for ``np.linalg.pinv(cov_nc_nc)``. Default 1e-10.
    data_stat_nc : np.ndarray, shape (nbins, nbins), optional
        Poisson data-statistical covariance on n_C, added to cov_nc_nc before the
        pseudoinverse to soften the constraint. Typical form: ``np.diag(nc_raw_counts)``.
    data_stat_ns : np.ndarray, shape (nbins, nbins), optional
        Poisson data-statistical covariance on n_S, added to cov_ns_ns and cov_ms_ms.

    Returns
    -------
    dict
        ``cov_Bs_Bs``, ``cov_Bs_Bs_constr``, ``cov_ns_ns``, ``cov_ms_ms``,
        ``cov_Ps_Ps``, ``cov_Ps_Bs``, ``cov_Ps_Bs_constr``, ``cov_Bs_nc``,
        ``cov_Ps_nc``, ``cov_nc_nc`` — covariance blocks in event-count² units.
        ``norm_cov_Bs_Bs``, ``norm_cov_Bs_Bs_constr``, ``norm_cov_ns_ns``,
        ``norm_cov_ms_ms`` — scalar variable-independent normalization variances.
        ``pinv_nc_nc`` — pseudoinverse of symmetrised cov_nc_nc (stat-augmented if given).
        ``Bs_output``, ``ns_output``, ``nc_output`` — pass-throughs for downstream functions.
    """
    nc_keys = set(nc_output.rate_syst_dict)
    _warn_key_mismatch(nc_keys, set(Bs_output.rate_syst_dict), "nc_output", "Bs_output")
    if Ps_output.xsec_syst_dict is not None:
        _warn_key_mismatch(nc_keys, set(Ps_output.xsec_syst_dict), "nc_output", "Ps_output.xsec_syst_dict")
    if ns_output is not None and ns_output.xsec_syst_dict is not None:
        _warn_key_mismatch(nc_keys, set(ns_output.xsec_syst_dict), "nc_output", "ns_output.xsec_syst_dict")

    nbins  = len(nc_output.rate_hist_cv)
    zeros  = lambda: np.zeros((nbins, nbins))
    cov_Bs_nc = zeros(); cov_nc_nc = zeros(); cov_Ps_nc = zeros()
    cov_Ps_Bs = zeros(); cov_Bs_Bs = zeros(); cov_Ps_Ps = zeros()

    # Scalar (1-bin) rate-only accumulators for variable-independent norm percentages.
    norm_Ps_Ps = 0.0;  norm_Ps_Bs = 0.0;  norm_Ps_nc = 0.0

    Ps_xsec = Ps_output.xsec_syst_dict if Ps_output.xsec_syst_dict is not None else {}
    for key in _iter_shared_keys(nc_output.rate_syst_dict, Bs_output.rate_syst_dict,
                                  allowed_keys, extra_dict=Ps_xsec):
        Bs_h  = Bs_output.rate_syst_dict[key]["hists"]
        nc_h  = nc_output.rate_syst_dict[key]["hists"]
        Ps_h  = Ps_xsec[key]["hists"]

        # DetVar keys store their own CV to capture only the detector variation
        # effect; GENIE/Flux/MCstat fall back to rate_hist_cv.
        Bs_cv = np.asarray(Bs_output.rate_syst_dict[key].get("hist_cv", Bs_output.rate_hist_cv))
        nc_cv = np.asarray(nc_output.rate_syst_dict[key].get("hist_cv", nc_output.rate_hist_cv))
        Ps_cv = np.asarray(Ps_xsec[key].get("hist_cv", Ps_output.rate_hist_cv))

        Bs_h, Bs_cv = _rescale_detvar(Bs_h, Bs_cv, Bs_output, key)
        nc_h, nc_cv = _rescale_detvar(nc_h, nc_cv, nc_output, key)
        Ps_h, Ps_cv = _rescale_detvar(Ps_h, Ps_cv, Ps_output, key)

        cov_Bs_nc += nonsymmetric_cov(Bs_h, Bs_cv, nc_h, nc_cv)
        cov_Ps_nc += nonsymmetric_cov(Ps_h, Ps_cv, nc_h, nc_cv)
        cov_Ps_Bs += nonsymmetric_cov(Ps_h, Ps_cv, Bs_h, Bs_cv)
        cov_nc_nc += nonsymmetric_cov(nc_h, nc_cv, nc_h, nc_cv)
        cov_Bs_Bs += nonsymmetric_cov(Bs_h, Bs_cv, Bs_h, Bs_cv)
        cov_Ps_Ps += nonsymmetric_cov(Ps_h, Ps_cv, Ps_h, Ps_cv)

        # Scalar (1-bin) accumulators for variable-independent normalization percentages.
        # Ps uses xsec_syst_dict universe histograms (response-matrix-weighted) so that
        # norm_cov_ns_ns / norm_cov_ms_ms reflect the cross-section-level uncertainty,
        # not just the raw rate uncertainty.  The CV denominator is always rate_hist_cv
        # (not xsec_hist_cv, which is background-subtracted) so the fraction is relative
        # to the total observed rate.  Bs and nc use rate_syst_dict histograms throughout.
        _nuniv = np.asarray(Ps_h).shape[1]
        Ps_d = np.asarray(Ps_h).sum(axis=0) - float(np.asarray(Ps_cv).sum())
        Bs_d = np.asarray(Bs_h).sum(axis=0) - float(np.asarray(Bs_cv).sum())
        nc_d = np.asarray(nc_h).sum(axis=0) - float(np.asarray(nc_cv).sum())
        norm_Ps_Ps += float(np.dot(Ps_d, Ps_d)) / _nuniv
        norm_Ps_Bs += float(np.dot(Ps_d, Bs_d)) / _nuniv
        norm_Ps_nc += float(np.dot(Ps_d, nc_d)) / _nuniv

    cov_nc_nc_sym    = (cov_nc_nc + cov_nc_nc.T) / 2
    # Data stats on n_C soften the constraint: a noisier sideband
    # measurement should be trusted less when correcting B_S.
    cov_nc_nc_pinv_input = cov_nc_nc_sym + (data_stat_nc if data_stat_nc is not None else 0.0)
    pinv_nc          = np.linalg.pinv(cov_nc_nc_pinv_input, rcond=rcond)
    cov_Bs_Bs_constr = cov_Bs_Bs  - cov_Bs_nc @ pinv_nc @ cov_Bs_nc.T
    cov_Ps_Bs_constr = cov_Ps_Bs  - cov_Ps_nc @ pinv_nc @ cov_Bs_nc.T
    cov_ms_ms        = cov_Ps_Ps  + cov_Ps_Bs_constr + cov_Ps_Bs_constr.T + cov_Bs_Bs_constr
    cov_ns_ns        = cov_Ps_Ps  + cov_Ps_Bs        + cov_Ps_Bs.T        + cov_Bs_Bs
    if data_stat_ns is not None:
        cov_ms_ms = cov_ms_ms + data_stat_ns
        cov_ns_ns = cov_ns_ns + data_stat_ns

    # Scalar (1-bin) CCBC constraint — variable-independent normalization percentages.
    # Bs and nc blocks: summing a rate covariance matrix = scalar rate variance.
    norm_Bs_Bs = float(np.sum(cov_Bs_Bs))
    norm_Bs_nc = float(np.sum(cov_Bs_nc))
    norm_nc_nc = float(np.sum(cov_nc_nc))
    if norm_nc_nc > 0:
        norm_Bs_Bs_constr = norm_Bs_Bs - norm_Bs_nc ** 2 / norm_nc_nc
        norm_Ps_Bs_constr = norm_Ps_Bs - norm_Ps_nc * norm_Bs_nc / norm_nc_nc
    else:
        norm_Bs_Bs_constr = norm_Bs_Bs
        norm_Ps_Bs_constr = norm_Ps_Bs
    norm_ms_ms = norm_Ps_Ps + 2.0 * norm_Ps_Bs_constr + norm_Bs_Bs_constr
    norm_ns_ns = norm_Ps_Ps + 2.0 * norm_Ps_Bs        + norm_Bs_Bs

    # Cross-check cov_ns_ns against ns_output.xsec_syst_dict if provided.
    # The identity holds when ns_h_xsec = Ps_h_xsec + Bs_h_rate for each key,
    # which is guaranteed when the background component of ns_output.xsec_syst_dict
    # (a plain bincount over background events) equals Bs_output.rate_syst_dict.
    # The CV used on both sides is rate_hist_cv so the algebraic identity
    # cov(Ps_diff + Bs_diff) = cov_Ps_Ps + cross_terms + cov_Bs_Bs holds exactly.
    if ns_output is not None:
        if ns_output.xsec_syst_dict is None:
            warnings.warn(
                "ns_output provided but xsec_syst_dict is None; "
                "skipping cov_ns_ns cross-check.",
                stacklevel=2,
            )
        else:
            cov_ns_ns_direct = zeros()
            for key in _iter_shared_keys(nc_output.rate_syst_dict, Bs_output.rate_syst_dict,
                                          allowed_keys, extra_dict=ns_output.xsec_syst_dict):
                ns_h      = ns_output.xsec_syst_dict[key]["hists"]
                ns_cv_key = np.asarray(ns_output.xsec_syst_dict[key].get("hist_cv", ns_output.rate_hist_cv))
                ns_h, ns_cv_key = _rescale_detvar(ns_h, ns_cv_key, ns_output, key)
                cov_ns_ns_direct += nonsymmetric_cov(ns_h, ns_cv_key, ns_h, ns_cv_key)

            # Compare against the syst-only part of cov_ns_ns; data_stat_ns is
            # added externally and is not built from universe histograms.
            cov_ns_ns_syst = cov_ns_ns - (data_stat_ns if data_stat_ns is not None else 0.0)
            scale    = np.max(np.abs(cov_ns_ns_syst))
            rel_diff = np.max(np.abs(cov_ns_ns_syst - cov_ns_ns_direct)) / scale if scale > 0 else 0.0
            if rel_diff > 1e-6:
                warnings.warn(
                    f"cov_ns_ns cross-check failed (max relative diff = {rel_diff:.2e}). "
                    "The on-the-fly combination of Ps_xsec and Bs_rate universe histograms "
                    "differs from ns_output.xsec_syst_dict. "
                    "Verify that ns_output uses the same events as Ps_output + Bs_output.",
                    stacklevel=2,
                )

    return {
        "cov_Bs_Bs":             cov_Bs_Bs,
        "cov_Bs_Bs_constr":      cov_Bs_Bs_constr,
        "cov_ns_ns":             cov_ns_ns,
        "cov_ms_ms":             cov_ms_ms,
        "cov_Ps_Ps":             cov_Ps_Ps,
        "cov_Ps_Bs":             cov_Ps_Bs,
        "cov_Ps_Bs_constr":      cov_Ps_Bs_constr,
        "cov_Bs_nc":             cov_Bs_nc,
        "cov_Ps_nc":             cov_Ps_nc,
        "cov_nc_nc":             cov_nc_nc,
        # Scalar (1-bin) rate-only variances — variable-independent norm percentages.
        "norm_cov_Bs_Bs":        norm_Bs_Bs,
        "norm_cov_Bs_Bs_constr": norm_Bs_Bs_constr,
        "norm_cov_ns_ns":        norm_ns_ns,
        "norm_cov_ms_ms":        norm_ms_ms,
        # Pass inputs through so downstream functions can access them without
        # the caller having to re-supply them.
        "Bs_output":             Bs_output,
        "ns_output":             ns_output,
        "nc_output":             nc_output,
        "pinv_nc_nc":            pinv_nc,
    }


# ---------------------------------------------------------------------------
# Constrained background prediction
# ---------------------------------------------------------------------------


def get_constrained_background(
    ccbc_cov: dict,
    fd_nc_hist: np.ndarray,
) -> np.ndarray:
    """Shift the CV background prediction using the CCBC linear predictor.

    Applies the standard CCBC constraint update:

    .. math::

        \\hat{B}_S = B_S^{\\rm CV}
            + C_{B_S n_C}\\, C_{n_C n_C}^{+}\\, (n_C^{\\rm FD} - n_C^{\\rm CV})

    Parameters
    ----------
    ccbc_cov : dict
        Output of :func:`get_ccbc_cov`.  Must contain ``"Bs_output"``,
        ``"nc_output"``, ``"cov_Bs_nc"``, and ``"pinv_nc_nc"``.
    fd_nc_hist : np.ndarray, shape (nbins,)
        Fake-data control-region histogram in absolute event-count units at
        mcbnb_pot, matching ``nc_output.rate_hist_cv`` (i.e. ``weights_mc``-weighted).

    Returns
    -------
    np.ndarray, shape (nbins,)
        Constrained background prediction.  May be negative in extreme
        fake-data scenarios; clamp or inspect as needed.
    """
    cv_Bs = np.asarray(ccbc_cov["Bs_output"].rate_hist_cv)
    cv_nc = np.asarray(ccbc_cov["nc_output"].rate_hist_cv)
    shift = ccbc_cov["cov_Bs_nc"] @ ccbc_cov["pinv_nc_nc"] @ (np.asarray(fd_nc_hist) - cv_nc)
    return cv_Bs + shift


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
    category_dict_Bs: dict | None = None,
    category_dict_ns: dict | None = None,
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
    cov_list         : list of dicts returned by :func:`get_ccbc_cov`, one per column.
                       Each dict must contain ``cov_Bs_Bs``, ``cov_Bs_Bs_constr``,
                       ``cov_ns_ns``, ``cov_ms_ms`` (in event-count^2 units).
    cov_stat         : single :func:`get_ccbc_cov` dict computed with MCstat only,
                       used for the stat-only hatch overlay.
    Bs_hist          : CV background histogram in event-count units.
    ns_hist          : CV total event-rate histogram in event-count units.
    var_config       : VariableConfig for bin edges and axis labels.
    list_keys        : column titles, one per entry in ``cov_list``.  Each key is
                       looked up in the two category dicts to obtain post-constraint
                       highlight colours.
    category_dict_Bs : category style dict used to colour the B_S rows.  Keys are
                       systematic group names; each value must have a ``'color'`` entry.
                       Defaults to ``analysis.category_dict_control``.
    category_dict_ns : category style dict used to colour the n_S rows.
                       Defaults to ``analysis.category_dict_signal``.

    Returns
    -------
    fig, axes  where axes has shape (4, ncols).
    """
    if category_dict_Bs is None or category_dict_ns is None:
        from .analysis import category_dict_control, category_dict_signal
        if category_dict_Bs is None:
            category_dict_Bs = category_dict_control
        if category_dict_ns is None:
            category_dict_ns = category_dict_signal

    ncols      = len(cov_list)
    bins       = var_config.bins
    centers    = var_config.bin_centers
    colors_Bs  = [category_dict_Bs.get(k, {}).get("color", "C0") for k in list_keys]
    colors_ns  = [category_dict_ns.get(k, {}).get("color", "C0") for k in list_keys]

    stat_ratio_Bs = np.sqrt(np.diag(cov_stat["cov_Bs_Bs"])) / Bs_hist
    stat_ratio_ns = np.sqrt(np.diag(cov_stat["cov_ns_ns"])) / ns_hist

    fig = plt.figure(figsize=(4 * ncols, 8))
    gs  = GridSpec(4, ncols, height_ratios=[4, 1, 4, 1], hspace=0.4)

    axes_Bs_main  = [fig.add_subplot(gs[0, i]) for i in range(ncols)]
    axes_Bs_ratio = [fig.add_subplot(gs[1, i]) for i in range(ncols)]
    axes_ns_main  = [fig.add_subplot(gs[2, i]) for i in range(ncols)]
    axes_ns_ratio = [fig.add_subplot(gs[3, i]) for i in range(ncols)]

    for i, (cov, color_Bs, color_ns, key) in enumerate(zip(cov_list, colors_Bs, colors_ns, list_keys)):
        pre_cov_Bs  = cov["cov_Bs_Bs"]
        post_cov_Bs = cov["cov_Bs_Bs_constr"]
        pre_cov_ns  = cov["cov_ns_ns"]
        post_cov_ns = cov["cov_ms_ms"]

        pre_ratio_Bs  = np.sqrt(np.diag(pre_cov_Bs))  / Bs_hist
        post_ratio_Bs = np.sqrt(np.diag(post_cov_Bs)) / Bs_hist
        pre_ratio_ns  = np.sqrt(np.diag(pre_cov_ns))  / ns_hist
        post_ratio_ns = np.sqrt(np.diag(post_cov_ns)) / ns_hist

        label_pre_Bs  = f"Pre-constraint ({_norm_pct(pre_cov_Bs,  Bs_hist):.1f}%)"
        label_post_Bs = f"Post-constraint ({_norm_pct(post_cov_Bs, Bs_hist):.1f}%)"
        label_pre_ns  = f"Pre-constraint ({_norm_pct(pre_cov_ns,  ns_hist):.1f}%)"
        label_post_ns = f"Post-constraint ({_norm_pct(post_cov_ns, ns_hist):.1f}%)"

        # Row 1: B_S main
        axes_Bs_main[i].stairs(Bs_hist, bins, color="black", label=r"$B_S^{CV}$")
        axes_Bs_main[i].errorbar(centers, Bs_hist, yerr=np.sqrt(np.diag(pre_cov_Bs)),
                                 fmt=".", capsize=3, color="black", label=label_pre_Bs)
        axes_Bs_main[i].errorbar(centers, Bs_hist, yerr=np.sqrt(np.diag(post_cov_Bs)),
                                 fmt=".", capsize=3, color=color_Bs, label=label_post_Bs)
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
                                      step="pre", color=color_Bs, alpha=0.5)
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
                                 fmt=".", capsize=3, color="black", label=label_pre_ns)
        axes_ns_main[i].errorbar(centers, ns_hist, yerr=np.sqrt(np.diag(post_cov_ns)),
                                 fmt=".", capsize=3, color=color_ns, label=label_post_ns)
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
                                      step="pre", color=color_ns, alpha=0.5)
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


def plot_ccbc_summary(
    ccbc_cov: dict,
    cov_stat: dict,
    Bs_hist: np.ndarray,
    ns_hist: np.ndarray,
    var_config: VariableConfig,
    color_Bs: str = "C1",
    color_ns: str = "C0",
    axes: np.ndarray | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Two-panel summary of total pre/post-constraint uncertainty reduction.

    Shows the total (all-group summed) pre- and post-constraint uncertainties
    for the background-only (B_S) and total signal-region (n_S) samples.  A
    2 × 2 GridSpec places B_S on the left and n_S on the right, with the main
    histogram row on top and a fractional-uncertainty ratio panel below.

    Complementary to :func:`plot_ccbc_constraint`, which shows per-systematic-
    group breakdowns.  All histogram and covariance inputs must be in the same
    event-count units (i.e. already scaled by the appropriate flux × POT factor).

    The pre-constraint covariances are taken from ``ccbc_cov["Bs_output"].rate_cov``
    and ``ccbc_cov["ns_output"].rate_cov`` — the full systematics outputs stored
    by :func:`get_ccbc_cov` — rather than from the universe-systematic-only blocks
    in ``ccbc_cov``.  The scale factor is inferred from ``Bs_hist`` vs
    ``Bs_output.rate_hist_cv``.  When ``ccbc_cov["ns_output"]`` is ``None``
    (i.e. ``ns_output`` was not passed to :func:`get_ccbc_cov`), the pre-constraint
    n_S covariance falls back to ``ccbc_cov["cov_ns_ns"]``.

    Parameters
    ----------
    ccbc_cov : dict returned by :func:`get_ccbc_cov`
        Must contain ``cov_Bs_Bs_constr``, ``cov_ms_ms``, ``Bs_output``, and
        optionally ``ns_output``.
    cov_stat : dict
        Same structure as ``ccbc_cov`` but computed with MC-stat only, for
        the stat-only hatch overlay.
    Bs_hist : np.ndarray, shape (nbins,)
        CV background histogram in event-count units.
    ns_hist : np.ndarray, shape (nbins,)
        CV total event-rate histogram in event-count units.
    var_config : VariableConfig
        Bin edges and axis labels.
    color_Bs : str, default ``"C1"``
        Post-constraint colour for the background-only (B_S) panels.
    color_ns : str, default ``"C0"``
        Post-constraint colour for the signal+background (n_S) panels.
    axes : (2, 2) array-like of Axes, optional
        Pre-existing axes in row-major order
        ``[[bs_main, ns_main], [bs_ratio, ns_ratio]]``.
        A new figure is created if not provided.

    Returns
    -------
    fig : plt.Figure
    axes : np.ndarray, shape (2, 2)
        ``axes[0, 0]`` — B_S main panel
        ``axes[0, 1]`` — n_S main panel
        ``axes[1, 0]`` — B_S ratio panel
        ``axes[1, 1]`` — n_S ratio panel
    """
    bins       = var_config.bins
    centers    = var_config.bin_centers
    Bs_output  = ccbc_cov["Bs_output"]
    ns_output  = ccbc_cov["ns_output"]

    # Infer the scale factor from the already-scaled Bs_hist.
    scale_sq = (np.sum(Bs_hist) / np.sum(Bs_output.rate_hist_cv)) ** 2

    pre_cov_Bs  = ccbc_cov["cov_Bs_Bs"] * scale_sq
    post_cov_Bs = ccbc_cov["cov_Bs_Bs_constr"] * scale_sq
    pre_cov_ns  = ccbc_cov["cov_ns_ns"] * scale_sq
    post_cov_ns = ccbc_cov["cov_ms_ms"] * scale_sq

    pre_err_Bs  = np.sqrt(np.diag(pre_cov_Bs))
    post_err_Bs = np.sqrt(np.diag(post_cov_Bs))
    pre_err_ns  = np.sqrt(np.diag(pre_cov_ns))
    post_err_ns = np.sqrt(np.diag(post_cov_ns))

    stat_err_Bs = np.sqrt(np.diag(cov_stat["cov_Bs_Bs"] * scale_sq))
    stat_err_ns = np.sqrt(np.diag(cov_stat["cov_ns_ns"] * scale_sq))

    pre_ratio_Bs  = pre_err_Bs  / Bs_hist
    post_ratio_Bs = post_err_Bs / Bs_hist
    pre_ratio_ns  = pre_err_ns  / ns_hist
    post_ratio_ns = post_err_ns / ns_hist
    stat_ratio_Bs = stat_err_Bs / Bs_hist
    stat_ratio_ns = stat_err_ns / ns_hist

    label_pre_Bs  = f"Pre-constraint ({_norm_pct(pre_cov_Bs,  Bs_hist):.1f}%)"
    label_post_Bs = f"Post-constraint ({_norm_pct(post_cov_Bs, Bs_hist):.1f}%)"
    label_pre_ns  = f"Pre-constraint ({_norm_pct(pre_cov_ns,  ns_hist):.1f}%)"
    label_post_ns = f"Post-constraint ({_norm_pct(post_cov_ns, ns_hist):.1f}%)"

    if axes is None:
        fig = plt.figure(figsize=(10, 5))
        gs  = GridSpec(2, 2, height_ratios=[4, 1], hspace=0.2, wspace=0.25)
        ax_bs_main  = fig.add_subplot(gs[0, 0])
        ax_ns_main  = fig.add_subplot(gs[0, 1])
        ax_bs_ratio = fig.add_subplot(gs[1, 0])
        ax_ns_ratio = fig.add_subplot(gs[1, 1])
        axes = np.array([[ax_bs_main, ax_ns_main], [ax_bs_ratio, ax_ns_ratio]])
    else:
        axes = np.asarray(axes)
        fig  = axes[0, 0].get_figure()
        ax_bs_main,  ax_ns_main  = axes[0, 0], axes[0, 1]
        ax_bs_ratio, ax_ns_ratio = axes[1, 0], axes[1, 1]

    # B_S main panel
    ax_bs_main.stairs(Bs_hist, bins, color="black", label=r"$B_S^{CV}$")
    ax_bs_main.errorbar(centers, Bs_hist, yerr=pre_err_Bs,
                        fmt=".", capsize=3, color="black", label=label_pre_Bs)
    ax_bs_main.errorbar(centers, Bs_hist, yerr=post_err_Bs,
                        fmt=".", capsize=3, color=color_Bs, label=label_post_Bs)
    ax_bs_main.fill_between(
        x=bins,
        y1=_repeat(Bs_hist - stat_err_Bs),
        y2=_repeat(Bs_hist + stat_err_Bs),
        step="pre", facecolor="none",
        edgecolor=mpl.colors.to_rgba("gray", 0.5), lw=0, hatch="////",
        label="MC stat only",
    )
    ax_bs_main.set_title("Background only")
    ax_bs_main.set_ylabel("Events")
    ax_bs_main.legend(fontsize=9)
    ax_bs_main.set_xticks(bins)
    ax_bs_main.set_xticklabels(var_config.bin_labels, fontsize=8)

    # n_S main panel
    ax_ns_main.stairs(ns_hist, bins, color="black", label=r"$n_S^{CV}$")
    ax_ns_main.errorbar(centers, ns_hist, yerr=pre_err_ns,
                        fmt=".", capsize=3, color="black", label=label_pre_ns)
    ax_ns_main.errorbar(centers, ns_hist, yerr=post_err_ns,
                        fmt=".", capsize=3, color=color_ns, label=label_post_ns)
    ax_ns_main.fill_between(
        x=bins,
        y1=_repeat(ns_hist - stat_err_ns),
        y2=_repeat(ns_hist + stat_err_ns),
        step="pre", facecolor="none",
        edgecolor=mpl.colors.to_rgba("gray", 0.5), lw=0, hatch="////",
        label="MC stat only",
    )
    ax_ns_main.set_title("Signal + Background")
    ax_ns_main.set_ylabel("Events")
    ax_ns_main.legend(fontsize=9)
    ax_ns_main.set_xticks(bins)
    ax_ns_main.set_xticklabels(var_config.bin_labels, fontsize=8)

    # B_S ratio panel
    ax_bs_ratio.fill_between(bins, 1 - _repeat(pre_ratio_Bs), 1 + _repeat(pre_ratio_Bs),
                              step="pre", color="gray", alpha=0.3)
    ax_bs_ratio.fill_between(bins, 1 - _repeat(post_ratio_Bs), 1 + _repeat(post_ratio_Bs),
                              step="pre", color=color_Bs, alpha=0.5)
    ax_bs_ratio.fill_between(bins, 1 - _repeat(stat_ratio_Bs), 1 + _repeat(stat_ratio_Bs),
                              step="pre", facecolor="none", edgecolor="gray", hatch="//")
    ax_bs_ratio.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.7)
    ax_bs_ratio.set_ylabel("Ratio")
    ax_bs_ratio.set_xlabel(var_config.var_labels[1])
    ax_bs_ratio.set_xticks(bins)
    ax_bs_ratio.set_xticklabels(var_config.bin_labels, fontsize=8)

    # n_S ratio panel
    ax_ns_ratio.fill_between(bins, 1 - _repeat(pre_ratio_ns), 1 + _repeat(pre_ratio_ns),
                              step="pre", color="gray", alpha=0.3)
    ax_ns_ratio.fill_between(bins, 1 - _repeat(post_ratio_ns), 1 + _repeat(post_ratio_ns),
                              step="pre", color=color_ns, alpha=0.5)
    ax_ns_ratio.fill_between(bins, 1 - _repeat(stat_ratio_ns), 1 + _repeat(stat_ratio_ns),
                              step="pre", facecolor="none", edgecolor="gray", hatch="//")
    ax_ns_ratio.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.7)
    ax_ns_ratio.set_ylabel("Ratio")
    ax_ns_ratio.set_xlabel(var_config.var_labels[1])
    ax_ns_ratio.set_xticks(bins)
    ax_ns_ratio.set_xticklabels(var_config.bin_labels, fontsize=8)

    # Shared y-limits across the two columns
    y0 = min(ax_bs_main.get_ylim()[0],  ax_ns_main.get_ylim()[0])
    y1 = max(ax_bs_main.get_ylim()[1],  ax_ns_main.get_ylim()[1])
    ax_bs_main.set_ylim(y0, y1)
    ax_ns_main.set_ylim(y0, y1)

    r0 = min(ax_bs_ratio.get_ylim()[0], ax_ns_ratio.get_ylim()[0])
    r1 = max(ax_bs_ratio.get_ylim()[1], ax_ns_ratio.get_ylim()[1])
    ax_bs_ratio.set_ylim(r0, r1)
    ax_ns_ratio.set_ylim(r0, r1)

    for ax in axes.flat:
        ax.tick_params(axis="y", labelsize=8)

    return fig, axes


def plot_ccbc_fd_comparison(
    ccbc_cov: dict,
    fd_nc_hist: np.ndarray,
    fd_ns_hist: np.ndarray,
    fd_Bs_hist: np.ndarray,
    ns_hist: np.ndarray,
    var_config: VariableConfig,
    axes: np.ndarray | None = None,
    color_pre: str = "C3",
    color_post: str = "C0",
) -> tuple[plt.Figure, np.ndarray, dict]:
    """Two-panel comparison of unconstrained and constrained predictions vs fake data.

    Left panel shows the unconstrained total signal-region prediction
    (labelled :math:`n_S`); right panel shows the CCBC-constrained prediction
    (labelled :math:`m_S`).  Each panel contains:

    * A filled ±1σ systematic uncertainty band around the prediction.
    * A dash-dot line for the background prediction (B_S^CV on the left;
      B̂_S on the right).
    * A dashed gray line for the true reweighted background (``fd_Bs_hist``).
    * Errorbar points for the fake-data total (``fd_ns_hist``), labelled :math:`D_S`.
    * A chi-squared text annotation placed outside the legend.

    All inputs must be in **absolute event-count units at mcbnb_pot**
    (``weights_mc`` summed, no flux division), consistent with
    :class:`~nueana.classes.SystematicsOutput` and the outputs of
    :func:`~nueana.fdt.make_fake_data_hists`.

    Parameters
    ----------
    ccbc_cov : dict
        Output of :func:`get_ccbc_cov`.  Must contain ``Bs_output``,
        ``cov_Bs_nc``, ``pinv_nc_nc``, ``cov_ns_ns``, and ``cov_ms_ms``.
    fd_nc_hist : np.ndarray, shape (nbins,)
        Fake-data control-region histogram in absolute event-count units at mcbnb_pot.
        Passed to :func:`get_constrained_background` to derive B̂_S.
        Use ``fd_nc`` from :func:`~nueana.fdt.make_fake_data_hists`.
    fd_ns_hist : np.ndarray, shape (nbins,)
        Fake-data total signal-region histogram in absolute event-count units at mcbnb_pot
        (before background subtraction).  Plotted as :math:`D_S`.
        Use ``fd_ns`` from :func:`~nueana.fdt.make_fake_data_hists`.
    fd_Bs_hist : np.ndarray, shape (nbins,)
        True reweighted background histogram in absolute event-count units at mcbnb_pot.
        Plotted as the "true background" reference line.
        Use ``fd_Bs`` from :func:`~nueana.fdt.make_fake_data_hists`.
    ns_hist : np.ndarray, shape (nbins,)
        CV total signal-region prediction (P_S + B_S) in absolute event-count units at
        mcbnb_pot (e.g. ``ns_output.rate_hist_cv``).
    var_config : VariableConfig
        Bin edges and axis labels.
    axes : array-like of two Axes, optional
        Pre-existing (2,) axes array.  A new figure with ``sharey=True`` is
        created if not provided.
    color_pre : str, default ``"C3"``
        Colour for the unconstrained prediction and its band.
    color_post : str, default ``"C0"``
        Colour for the constrained prediction and its band.

    Returns
    -------
    fig : plt.Figure
    axes : np.ndarray, shape (2,)
        ``axes[0]`` — unconstrained (:math:`n_S`) panel
        ``axes[1]`` — constrained (:math:`m_S`) panel
    result : dict
        ``chisq_pre``      — χ² of the unconstrained prediction vs ``fd_ns_hist``
        ``chisq_post``     — χ² of the constrained prediction vs ``fd_ns_hist``
        ``ndof``           — number of bins used as the χ² degrees of freedom
        ``constrained_ns`` — constrained total prediction in absolute event-count units
        ``constrained_Bs`` — constrained background prediction in absolute event-count units
    """
    Bs_cv          = np.asarray(ccbc_cov["Bs_output"].rate_hist_cv)
    constrained_Bs = get_constrained_background(ccbc_cov, fd_nc_hist)
    constrained_ns = ns_hist + (constrained_Bs - Bs_cv)

    cov_pre  = np.asarray(ccbc_cov["cov_ns_ns"])
    cov_post = np.asarray(ccbc_cov["cov_ms_ms"])
    err_pre  = np.sqrt(np.diag(cov_pre))
    err_post = np.sqrt(np.diag(cov_post))

    chisq_pre  = get_chisq_diff(fd_ns_hist - ns_hist,        cov_pre)
    chisq_post = get_chisq_diff(fd_ns_hist - constrained_ns, cov_post)
    ndof = len(ns_hist)

    bins    = var_config.bins
    centers = var_config.bin_centers

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    else:
        axes = np.asarray(axes)
        fig  = axes[0].get_figure()

    panels = [
        (axes[0], ns_hist,        err_pre,  Bs_cv,          r"$B_S^{\rm CV}$", chisq_pre,  color_pre,  "unconstrained", r"$n_S = \phi_S^{\rm CV} + B_S^{\rm CV}$", "pre"),
        (axes[1], constrained_ns, err_post, constrained_Bs, r"$B_S^{\rm constr}$",    chisq_post, color_post,   "constrained", r"$m_S = \phi_S^{\rm CV} + B_S^{\rm constr}$",   "post"),
    ]
    for ax, pred, err, Bs_pred, Bs_label, chisq, color, title, label, sub in panels:
        ax.set_title(title, fontsize=12)
        ax.stairs(pred, bins, color=color, lw=1.5, label=label)
        ax.fill_between(
            bins, _repeat(pred - err), _repeat(pred + err),
            step="pre", color=color, alpha=0.25, label=r"MC stat.+syst.",
        )
        ax.stairs(Bs_pred,    bins, color=color,  lw=1.2, ls="-.", label=Bs_label)
        ax.stairs(fd_Bs_hist, bins, color="gray", lw=1.2, ls="--", label=r"true $B_S$")
        ax.errorbar(centers, fd_ns_hist, fmt="ko", ms=5, label=r"$D_S$")
        ax.set_xlabel(var_config.var_labels[1], )
        ax.set_xticks(bins)
        ax.set_xticklabels(var_config.bin_labels, )
        ax.tick_params(axis="y", labelsize=8)
        ax.legend(fontsize=9)

    axes[0].set_ylabel("Events")
    axes[1].tick_params(axis="y", labelleft=True)

    # Anchor chi-squared annotations just below each legend using the rendered
    # bounding box — mirrors the plot_mc_data pattern in plotting.py.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, (_, _, _, _, _, chisq, _, _, _, sub) in zip(axes, panels):
        leg = ax.get_legend()
        if leg is not None:
            bb           = leg.get_window_extent(renderer).transformed(ax.transAxes.inverted())
            anchor_right = bb.x0 > 0.5
            ann_x        = bb.x1 if anchor_right else bb.x0
            ann_y        = bb.y0
            ann_ha       = "right" if anchor_right else "left"
        else:
            ann_x, ann_y, ann_ha = 0.97, 0.55, "right"
        ax.annotate(
            _chisq_str(chisq, sub, ndof),
            xy=(ann_x, ann_y),
            xycoords=ax.transAxes,
            xytext=(0, -4),
            textcoords="offset points",
            ha=ann_ha, va="top", 
        )

    return fig, axes, {
        "chisq_pre":      chisq_pre,
        "chisq_post":     chisq_post,
        "ndof":           ndof,
        "constrained_ns": constrained_ns,
        "constrained_Bs": constrained_Bs,
    }


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
    _warn_key_mismatch(set(nc_output.rate_syst_dict), set(Bs_output.rate_syst_dict), "nc_output", "Bs_output")

    nbins = len(Bs_output.rate_hist_cv)
    zeros = lambda: np.zeros((nbins, nbins))
    cov_Bs_nc = zeros(); cov_nc_nc = zeros(); cov_Bs_Bs = zeros()

    for key in _iter_shared_keys(nc_output.rate_syst_dict, Bs_output.rate_syst_dict, allowed_keys):
        Bs_h  = Bs_output.rate_syst_dict[key]["hists"]
        nc_h  = nc_output.rate_syst_dict[key]["hists"]
        Bs_cv = np.asarray(Bs_output.rate_syst_dict[key].get("hist_cv", Bs_output.rate_hist_cv))
        nc_cv = np.asarray(nc_output.rate_syst_dict[key].get("hist_cv", nc_output.rate_hist_cv))
        Bs_h, Bs_cv = _rescale_detvar(Bs_h, Bs_cv, Bs_output, key)
        nc_h, nc_cv = _rescale_detvar(nc_h, nc_cv, nc_output, key)
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
                annot=True, fmt=".2f", annot_kws={"fontsize": 8},
                cbar_kws={"label": "Fractional Covariance"})
    sns.heatmap(combined_corr, cmap="Spectral", ax=axes[1],
                annot=True, fmt=".2f", annot_kws={"fontsize": 8},
                cbar_kws={"label": "Correlation"}, vmin=-1, vmax=1)

    _annotate_block_axes(axes, nbins=nbins, var=var)
    return fig, axes


def plot_ccbc_key_correlations(
    Bs_output: SystematicsOutput,
    nc_output: SystematicsOutput,
    allowed_keys: Sequence[str] = ("GENIE",),
    sort_by: str = "correlation",
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes, object]:
    """Per-key normalization-level correlation between signal-region background and sideband.

    For each systematic key that matches ``allowed_keys`` and is present in both
    outputs, sums universe histogram fluctuations across bins to obtain a scalar
    per universe, then computes the Pearson correlation between B_S and n_C
    across universes.  A correlation near +1 means the knob shifts both regions
    together (CCBC can constrain it); near 0 means the two regions respond
    independently (CCBC provides no constraint).

    Parameters
    ----------
    Bs_output : SystematicsOutput
        Signal-region background-only systematics output.
    nc_output : SystematicsOutput
        Sideband (control-region) systematics output.
    allowed_keys : sequence of str, default ``("GENIE",)``
        Key substrings to include — passed to :func:`~nueana.syst.key_in_allowed`.
    sort_by : ``"correlation"`` or ``"unc_norm_Bs"``
        Bar ordering. ``"correlation"`` (default) puts the least-correlated
        keys first, making it easy to spot knobs that won't be constrained.
        ``"unc_norm_Bs"`` orders by the B_S normalization uncertainty, putting
        the most impactful knobs first.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw into. A new figure is created if not provided.

    Returns
    -------
    fig : plt.Figure
    ax  : plt.Axes
    df  : pd.DataFrame
        Per-key table with columns ``key``, ``correlation``, ``unc_norm_Bs``,
        ``unc_norm_nc``.  Useful for further inspection.
    """
    import pandas as pd

    N_Bs = float(np.sum(Bs_output.rate_hist_cv))
    N_nc = float(np.sum(nc_output.rate_hist_cv))

    records = []
    for key in _iter_shared_keys(nc_output.rate_syst_dict, Bs_output.rate_syst_dict, allowed_keys):
        Bs_h  = np.asarray(Bs_output.rate_syst_dict[key]["hists"])  # (nbins, nuniv)
        nc_h  = np.asarray(nc_output.rate_syst_dict[key]["hists"])
        Bs_cv = np.asarray(Bs_output.rate_syst_dict[key].get("hist_cv", Bs_output.rate_hist_cv))
        nc_cv = np.asarray(nc_output.rate_syst_dict[key].get("hist_cv", nc_output.rate_hist_cv))
        # Correlation is scale-invariant; rescaling is needed only so unc_norm_Bs is dimensionless.
        Bs_h, Bs_cv = _rescale_detvar(Bs_h, Bs_cv, Bs_output, key)
        nc_h, nc_cv = _rescale_detvar(nc_h, nc_cv, nc_output, key)

        # scalar fluctuation per universe: sum bins, subtract CV total
        delta_Bs = Bs_h.sum(axis=0) - Bs_cv.sum()   # shape (nuniv,)
        delta_nc = nc_h.sum(axis=0) - nc_cv.sum()

        corr = float(np.corrcoef(delta_Bs, delta_nc)[0, 1]) if delta_Bs.std() > 0 and delta_nc.std() > 0 else 0.0

        cov_Bs = nonsymmetric_cov(Bs_h, Bs_cv, Bs_h, Bs_cv)
        cov_nc = nonsymmetric_cov(nc_h, nc_cv, nc_h, nc_cv)
        unc_norm_Bs = float(np.sqrt(max(np.sum(cov_Bs), 0.0))) / N_Bs if N_Bs > 0 else 0.0
        unc_norm_nc = float(np.sqrt(max(np.sum(cov_nc), 0.0))) / N_nc if N_nc > 0 else 0.0

        records.append({"key": key, "correlation": corr,
                        "unc_norm_Bs": unc_norm_Bs, "unc_norm_nc": unc_norm_nc})

    import pandas as pd
    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("No keys matched allowed_keys in both Bs_output and nc_output.")

    ascending = sort_by == "correlation"
    df = df.sort_values(sort_by, ascending=ascending).reset_index(drop=True)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, max(4, 0.3 * len(df))))
    else:
        fig = ax.get_figure()

    colors = [plt.cm.RdYlGn(0.1 + 0.8 * (c + 1) / 2) for c in df["correlation"]]
    bars = ax.barh(df["key"], df["correlation"], color=colors, edgecolor="none")

    # annotate each bar with unc_norm_Bs
    for bar, unc in zip(bars, df["unc_norm_Bs"]):
        x = bar.get_width()
        ax.text(
            x + 0.01 if x >= 0 else x - 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{unc:.1%}",
            va="center", ha="left" if x >= 0 else "right", fontsize=7, color="dimgray",
        )

    ax.axvline(0, color="k", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlim(-1.1, 1.3)
    ax.set_xlabel(r"Pearson $\rho$  (B$_S$ vs $n_C$, normalization)")
    ax.set_title(f"Per-key B$_S$–$n_C$ correlation  [{', '.join(allowed_keys)}]")
    ax.tick_params(axis="y", labelsize=8)

    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=mpl.colors.Normalize(-1, 1))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=r"$\rho$", fraction=0.03, pad=0.02)

    return fig, ax, df
