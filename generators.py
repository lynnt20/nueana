"""Generator comparison utilities for external neutrino event generators.

Handles loading FlatTree ROOT files from GENIE, NEUT, NuWro, GiBUU and
similar generators, applying flux weighting to combine nu/nubar samples,
and computing cross-section-weighted histograms comparable to an SBND
unfolded result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .classes import VariableConfig
from .utils import get_hist1d, bin_geometry

__all__ = [
    'DEFAULT_BRANCHES',
    'load_flux_weights',
    'load_generator',
    'make_generator_hist',
    'smear_generator_hist',
    'plot_generator_comparison',
]

DEFAULT_BRANCHES = [
    "cc", "PDGLep", "ELep", "CosLep", "Enu_true", "Q2", "q0", "Mode",
    "Weight", "fScaleFactor",
]


# ---------------------------------------------------------------------------
# Flux weighting
# ---------------------------------------------------------------------------


def load_flux_weights(flux_file: str) -> tuple[float, float]:
    """Load nue/nuebar flux normalization weights from a ROOT flux file.

    Reads the "flux_sbnd_nue" and "flux_sbnd_anue" histograms and returns
    weights that normalize each species to the combined total flux.  The
    product ``Weight * fScaleFactor * flux_weight`` gives ``xsec_weight``
    in cm²/nucleon.

    Parameters
    ----------
    flux_file : str
        Path to the ROOT file containing "flux_sbnd_nue" and
        "flux_sbnd_anue" histograms (integrated flux in cm⁻² POT⁻¹).

    Returns
    -------
    nue_weight : float
        Scale factor for electron-neutrino events.
    anue_weight : float
        Scale factor for electron-antineutrino events.
    """
    try:
        import uproot
    except ImportError as exc:
        raise ImportError(
            "uproot is required for load_flux_weights. "
            "Install it with: pip install uproot"
        ) from exc

    with uproot.open(flux_file) as f:
        nue_flux_vals,  _ = f["flux_sbnd_nue"].to_numpy()
        anue_flux_vals, _ = f["flux_sbnd_anue"].to_numpy()

    nue_flux   = float(nue_flux_vals.sum())
    anue_flux  = float(anue_flux_vals.sum())
    total_flux = nue_flux + anue_flux
    return nue_flux / total_flux, anue_flux / total_flux


# ---------------------------------------------------------------------------
# Generator loading
# ---------------------------------------------------------------------------


def _default_signal_fn(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df.cc == 1) & (df.PDGLep.abs() == 11) & (df.ELep > 0.5)]


def load_generator(
    nue_file: str,
    nuebar_file: str,
    samples_dir: str,
    flux_weights: tuple[float, float],
    branches: list[str] | None = None,
    signal_fn=None,
) -> pd.DataFrame:
    """Load and combine nu/nubar FlatTree ROOT files into a signal DataFrame.

    Reads ``FlatTree_VARS`` from each file, concatenates them, attaches
    ``flux_weight`` and ``xsec_weight`` columns, then applies ``signal_fn``.

    Parameters
    ----------
    nue_file : str
        Filename (relative to ``samples_dir``) of the nue FlatTree ROOT file.
    nuebar_file : str
        Filename (relative to ``samples_dir``) of the nuebar FlatTree ROOT file.
    samples_dir : str
        Directory containing the FlatTree ROOT files.
    flux_weights : (float, float)
        ``(nue_weight, anue_weight)`` as returned by :func:`load_flux_weights`.
    branches : list of str, optional
        Branches to read from FlatTree_VARS. Defaults to :data:`DEFAULT_BRANCHES`.
    signal_fn : callable, optional
        ``df -> df`` filter applied after weight columns are added.
        Default: CC events with PDGLep == ±11 and ELep > 0.5 GeV.

    Returns
    -------
    pd.DataFrame
        Signal-filtered DataFrame with added columns:

        - ``flux_weight``: per-event flux normalization factor
        - ``xsec_weight``: ``Weight * fScaleFactor * flux_weight`` [cm²/nucleon]
    """
    try:
        import uproot
    except ImportError as exc:
        raise ImportError(
            "uproot is required for load_generator. "
            "Install it with: pip install uproot"
        ) from exc

    if branches is None:
        branches = DEFAULT_BRANCHES
    if signal_fn is None:
        signal_fn = _default_signal_fn

    nue_weight, anue_weight = flux_weights

    dfs = []
    for fname in [nue_file, nuebar_file]:
        t = uproot.open(samples_dir + fname)["FlatTree_VARS"]
        dfs.append(t.arrays(branches, library="pd"))

    df = pd.concat(dfs, ignore_index=True)
    df['flux_weight'] = np.where(df.PDGLep == 11, nue_weight, anue_weight)
    df['xsec_weight'] = df.Weight * df.fScaleFactor * df.flux_weight
    return signal_fn(df)


# ---------------------------------------------------------------------------
# Histogram building
# ---------------------------------------------------------------------------


def make_generator_hist(
    gen_df: pd.DataFrame,
    col: str,
    var: VariableConfig,
    weights_col: str = 'xsec_weight',
) -> np.ndarray:
    """Build a cross-section histogram from a generator DataFrame.

    Parameters
    ----------
    gen_df : pd.DataFrame
        Generator DataFrame as returned by :func:`load_generator`.
    col : str
        Column name to histogram (e.g. ``'ELep'``, ``'CosLep'``).
    var : VariableConfig
        Provides bin edges.  Out-of-range events are folded into edge bins
        (overflow=True, matching the nueana default).
    weights_col : str, optional
        Weight column. Default: ``'xsec_weight'``.

    Returns
    -------
    np.ndarray
        Histogram in xsec units [cm²/nucleon per bin], not yet divided by
        bin width.  Pass to :func:`smear_generator_hist` or directly to
        :func:`plot_generator_comparison`.
    """
    return get_hist1d(
        data=gen_df[col],
        bins=var.bins,
        weights=gen_df[weights_col],
    )


# ---------------------------------------------------------------------------
# Smearing
# ---------------------------------------------------------------------------


def smear_generator_hist(
    gen_hist: np.ndarray,
    add_smear: np.ndarray,
) -> np.ndarray:
    """Apply the WienerSVD AddSmear matrix to a generator histogram.

    Generator histograms are already in cross-section units, so no
    ``xsec_scale`` conversion is needed — unlike the event-count-unit truths
    used in :func:`~nueana.fdt.plot_unfolded_result`.

    Parameters
    ----------
    gen_hist : np.ndarray, shape (n_bins,)
        Generator cross-section histogram in xsec units [cm²/nucleon per bin].
    add_smear : np.ndarray, shape (n_bins, n_bins)
        ``AddSmear`` matrix from the WienerSVD output dict.

    Returns
    -------
    np.ndarray, shape (n_bins,)
        Smeared histogram in the same xsec units.
    """
    return add_smear @ np.asarray(gen_hist, dtype=float)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_generator_comparison(
    gen_hists: dict[str, np.ndarray],
    var: VariableConfig,
    ax: plt.Axes | None = None,
    smear: np.ndarray | None = None,
    ylabel: str | None = None,
    divide_by_width: bool = True,
) -> plt.Axes:
    """Plot differential cross-section histograms for multiple generators.

    Parameters
    ----------
    gen_hists : dict of {str: np.ndarray}
        Mapping from generator label to cross-section histogram (in xsec
        units, not yet divided by bin width).  Build with
        :func:`make_generator_hist`.
    var : VariableConfig
        Provides bin edges, bin labels, and axis label pieces.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  Created if None.
    smear : np.ndarray or None, optional
        ``AddSmear`` matrix from WienerSVD.  When provided, each histogram is
        passed through :func:`smear_generator_hist` before plotting.
    ylabel : str or None, optional
        Y-axis label.  Defaults to a ``dσ/d<var>`` label derived from ``var``.
    divide_by_width : bool, default True
        Divide each histogram by bin widths to produce a differential
        cross-section.  Pass False to plot raw bin integrals.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()

    _, widths = bin_geometry(var)

    for label, hist in gen_hists.items():
        if smear is not None:
            hist = smear_generator_hist(hist, smear)
        y = hist / widths if divide_by_width else hist
        ax.stairs(y, var.bins, lw=1.5, label=label)

    ax.set_xticks(var.bins)
    ax.set_xticklabels(var.bin_labels)
    ax.set_xlabel(var.var_labels[0], fontsize=12)

    if ylabel is None:
        plot_math = var.var_plot_name.strip("$")
        unit_part = (
            rf"\,\mathrm{{{var.var_unit}}}^{{-1}}" if var.var_unit else ""
        )
        ylabel = (
            rf"$d\sigma / d{plot_math}$ "
            rf"$[\mathrm{{cm}}^2{unit_part}\,\mathrm{{nucleon}}^{{-1}}]$"
        )
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend()

    return ax
