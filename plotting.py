"""Plotting helpers for nueana: stacked MC, PDG/mode breakdowns, and data overlays.

Functions
---------
plot_var : unified stacked histogram — signal types, PDG, interaction mode, or generic.
data_plot_overlay : data points with Poisson errors for overlaying on MC stacks.
plot_mc_data : combined MC+data figure with ratio subplot and chi-sq annotation.

All functions accept plain and MultiIndex DataFrames. Style and display options can
be bundled into a :class:`~nueana.classes.PlottingConfig` instance and passed as
``config``; keyword arguments take priority over the config.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import pandas as pd
import warnings
from dataclasses import fields as _dc_fields
try:
    from scipy.stats import chi2 as chi2_dist
except Exception:
    chi2_dist = None

__all__ = ['annotate_internal', 'plot_var', 'plot_var_pdg', 'data_plot_overlay', 'plot_mc_data']

from .constants import (signal_dict, signal_categories,
                        generic_dict, generic_categories,
                        pdg_categories,
                        mode_dict, mode_categories)
from .utils import ensure_lexsorted
from .syst import get_syst
from .histogram import get_hist1d
from .classes import PlottingConfig

def annotate_internal(ax):
    """Stamp 'SBND Internal' in the upper-left corner of *ax*."""
    ax.annotate("SBND Internal", xy=(0.0, 1.02), xycoords='axes fraction', ha='left',color='gray',fontweight='bold')

def plot_var(df: pd.DataFrame,
             var: tuple | str,
             bins: np.ndarray,
             ax = None,
             config: PlottingConfig | None = None,
             **kwargs,
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plot a variable as stacked histograms, selectable by category type.

    Category mode is controlled by ``generic``, ``pdg``, and ``mode`` (checked in
    that priority order):

    - default (all False): stack by interaction type (``signal_categories``).
    - ``pdg=True``: stack by leading-particle PDG code (``pdg_categories``).
    - ``mode=True``: stack by GENIE interaction mode (``mode_categories``).
    - ``generic=True``: stack by broad class — CC nu, NC nu, non-FV, dirt, cosmic
      (``generic_categories``).

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    var : tuple | str
        Column name (or multi-index tuple) to histogram.
    bins : np.ndarray
        Bin edges for the histogram.
    ax : matplotlib.axes.Axes, optional
        Axis to draw on. If None the current axis is used.
    config : PlottingConfig, optional
        Style/display options bundled into a dataclass. Keyword arguments take
        priority over any field set in ``config``.
    xlabel : str, optional
        X axis label. Defaults to the variable name when empty.
    ylabel : str, optional
        Y axis label. Defaults to 'Counts' when empty.
    title : str, optional
        Plot title. Defaults to the variable name when empty.
    counts : bool, default False
        If True, append event counts to legend labels.
    percents : bool, default False
        If True, append percentage-of-total to legend labels.
    scale : float, default 1.0
        Scale factor applied to all histogram bins (and error arrays).
    normalize : bool, default False
        If True, normalize histograms so the integral equals 1 (uses bin widths).
    mult_factor : float, default 1.0
        Extra multiplicative factor applied to the first category only. Intended for
        quick visual scaling; error propagation is not adjusted.
    cut_val : list of float, optional
        x-values at which to draw vertical dashed cut lines.
    plot_err : bool, default True
        If True, draw MC error bands (stat and/or syst).
    systs : True | np.ndarray | None, default None
        Controls how uncertainties are computed and displayed:

        - ``True``: read universe columns from ``df`` via :func:`~nueana.syst.get_syst`.
          If an MCstat universe is present the combined stat+syst band is drawn; otherwise
          stat and syst bands are drawn separately.
        - ``np.ndarray``: treat as a pre-computed ``(n_bins, n_bins)`` total covariance
          matrix (stat already included). A single combined band is drawn.
        - ``None`` (default): MC stat error only (diagonal, sum-of-weights-squared).
    pdg : bool, default False
        Stack by PDG code rather than signal type.
    pdg_col : tuple | str, default 'pfp_shw_truth_p_pdg'
        Column containing the PDG code per particle (used when ``pdg=True``).
    mode : bool, default False
        Stack by GENIE interaction mode.
    hatch : list of str, optional
        Hatch pattern per category (must match number of categories).
    bin_labels : list of str, optional
        Custom tick labels placed at each bin edge.
    generic : bool, default False
        Stack by broad category (CC nu, NC nu, non-FV, dirt, cosmic).
    overflow : bool, default True
        If True, fold values above ``bins[-1]`` into the last bin.
    legend_kwargs : dict, optional
        Forwarded to ``ax.legend()``, overriding the defaults
        ``{ncol: 2, loc: 'upper right'}``.

    Returns
    -------
    bins : np.ndarray
        The input bin edges (unchanged).
    steps : np.ndarray, shape (n_categories, len(bins))
        Cumulative step values per category used for the filled polygons.
    total_err : np.ndarray, shape (n_bins,)
        Per-bin total uncertainty (sqrt of diagonal of ``total_cov``).
    syst_dict : dict
        Per-systematic covariance matrices keyed by systematic name, plus
        ``'__total_cov__'`` holding the full ``(n_bins, n_bins)`` combined
        covariance (stat + syst, scaled).
    """
    _p = {f.name: getattr(config, f.name) for f in _dc_fields(config)} if config is not None else {}
    _p.update(kwargs)
    xlabel        = _p.get('xlabel', '')
    ylabel        = _p.get('ylabel', '')
    title         = _p.get('title', '')
    counts        = _p.get('counts', False)
    percents      = _p.get('percents', False)
    scale         = _p.get('scale', 1.0)
    normalize     = _p.get('normalize', False)
    mult_factor   = _p.get('mult_factor', 1.0)
    cut_val       = _p.get('cut_val', None)
    plot_err      = _p.get('plot_err', True)
    systs         = _p.get('systs', None)
    pdg           = _p.get('pdg', False)
    pdg_col       = _p.get('pdg_col', 'pfp_shw_truth_p_pdg')
    mode          = _p.get('mode', False)
    hatch         = _p.get('hatch', None)
    bin_labels    = _p.get('bin_labels', None)
    generic       = _p.get('generic', False)
    overflow      = _p.get('overflow', True)
    legend_kwargs = _p.get('legend_kwargs', None)
    if isinstance(df, pd.DataFrame):
        df = ensure_lexsorted(df, axis=0)
        df = ensure_lexsorted(df, axis=1)
    
    weight = False
    for col in df.columns:
        if "weights_mc" in "".join(list(col)):
          weight=True
          break
    
    if ax is None: ax = plt.gca()
    if generic:   categories = generic_categories
    elif pdg:     categories = pdg_categories
    elif mode:    categories = mode_categories
    else:         categories = signal_categories
    ncategories = len(categories)
    if hatch == None: hatch = [""]*ncategories
    alpha = 0.25 if pdg else 0.4
    
    hists       = np.zeros((ncategories,len(bins)-1)) # this is for storing the histograms
    steps       = np.zeros((ncategories,len(bins))) # this is for plotting
    bin_widths  = np.diff(bins)
    
    stats       = np.zeros(len(bins)-1)
    stats_err   = np.zeros(len(bins)-1)
    systs_err   = np.zeros(len(bins)-1)
    total_cov   = np.zeros((len(bins)-1, len(bins)-1))
    
    # Check if systs is provided as array (already includes stats)
    systs_is_array = isinstance(systs, np.ndarray)

    if (pdg==False) & (mode==False):
        for i, (key, entry) in enumerate(categories.items()):
            this_cat = entry["value"]
            hists[i] = get_hist1d(data=df[df.signal==this_cat][var],
                                  weights=df[df.signal==this_cat]['weights_mc'] if weight else None,
                                  bins=bins, overflow=overflow)
            
    elif mode:
        this_nu    = df[df.slc.truth.genie_mode == df.slc.truth.genie_mode]
        this_other = df[df.slc.truth.genie_mode != df.slc.truth.genie_mode]
        for i, (key, entry) in enumerate(categories.items()):
            if entry["value"] is not None:
                this_cat = entry["value"]
                hists[i] = get_hist1d(data=df[df.slc.truth.genie_mode==this_cat][var],
                                      weights=df[df.slc.truth.genie_mode==this_cat]['weights_mc'] if weight else None,
                                      bins=bins, overflow=overflow)
                this_nu = this_nu[this_nu.slc.truth.genie_mode != this_cat]
            elif entry["filter"] == "other_nu":
                hists[i] = get_hist1d(data=this_nu[var],
                                      weights=this_nu['weights_mc'] if weight else None,
                                      bins=bins, overflow=overflow)
            elif entry["filter"] == "non_nu":
                hists[i] = get_hist1d(data=this_other[var],
                                      weights=this_other['weights_mc'] if weight else None,
                                      bins=bins, overflow=overflow)
    else:
        process_col = tuple(list(pdg_col)[:-1] + ['start_process']) 
        # other_df stores any particles that we don't specify the pdg of
        this_nu_df      = df[df.signal <  signal_dict['cosmic']]#.sort_index()
        this_cosmic_df  = df[df.signal == signal_dict['cosmic']]#.sort_index()
        this_offbeam_df = df[df.signal == signal_dict['offbeam']]#.sort_index()
        # really only want to see electrons that are
        # primaries from a FV neutrino interaction
        where_notprim = ((abs(this_nu_df[pdg_col])==11) & 
                          (this_nu_df[process_col] != 0)) 
        this_notprim_df   = this_nu_df[where_notprim]
        this_nu_df         = this_nu_df[~where_notprim]
        this_other         = this_nu_df.copy()
        
        _pdg_populations = {
            "notprim": this_notprim_df,
            "cosmic":   this_cosmic_df,
            "offbeam":  this_offbeam_df,
        }
        for i, (key, entry) in enumerate(categories.items()):
            if entry["pdg"] is not None:
                pdg_value = entry["pdg"]
                pdg_df = this_nu_df[abs(this_nu_df[pdg_col])==pdg_value].sort_index()
                hists[i] = get_hist1d(data=pdg_df[var],
                                      weights=pdg_df['weights_mc'] if weight else None,
                                      bins=bins, overflow=overflow)
                this_other = this_other[abs(this_other[pdg_col])!=pdg_value]
            else:
                filt = entry["filter"]
                pop = _pdg_populations.get(filt, this_other if filt == "other_nu" else None)
                if pop is not None and len(pop) != 0:
                    hists[i] = get_hist1d(data=pop[var],
                                          weights=pop['weights_mc'] if weight else None,
                                          bins=bins, overflow=overflow)
    
    # ! THIS ASSUMES that the PDG of interest and the signal type of interest are both index 0
    # ! e.g. for nueCC (signal==0), e- is the first entry in the pdg_dict
    hists    *= scale 
    hists[0] = mult_factor*hists[0]

    # storing the sum of each category in case we want to display it
    hist_counts = np.sum(hists,axis=1)

    # --- Systematics ---
    # Three cases, resolved before the plot loop:
    #   2D array  → caller provides the full (stat+syst) covariance; stat is included.
    #   True      → inherit universe columns from df; detect whether MCstat is present.
    #   None/else → no systematics; only MC stat error is shown.
    systs_is_array = isinstance(systs, np.ndarray)

    if systs_is_array:
        # Case 2: full covariance supplied by caller — must be exactly 2D.
        if systs.ndim != 2 or systs.shape != (len(bins)-1, len(bins)-1):
            raise ValueError(
                "systs must be a 2D covariance matrix of shape "
                f"({len(bins)-1}, {len(bins)-1}); got {systs.shape}"
            )
        total_cov = np.array(systs, dtype=float, copy=True)
        systs_arr = np.sqrt(np.clip(np.diag(total_cov), a_min=0.0, a_max=None))
        syst_dict = {}
        calc_separate_mcstat = False

    elif systs is True:
        # Case 1: inherit systematics from universe columns in the dataframe.
        found_systs = any("univ_" in "_".join(list(col)) for col in df.columns)
        if not found_systs:
            print("systs=True but no universe columns found; computing stat error only")
            syst_dict = {}
            systs_arr = np.zeros(len(bins)-1)
            calc_separate_mcstat = True
        else:
            syst_dict = get_syst(reco_df=df, reco_var=var, bins=bins, scale=False)
            has_mcstat = any(str(k).lower() == 'mcstat' for k in syst_dict)
            for key in syst_dict:
                total_cov += syst_dict[key]['cov']
            systs_arr = np.sqrt(np.clip(np.diag(total_cov), a_min=0.0, a_max=None))
            calc_separate_mcstat = not has_mcstat

    else:
        # Case 3: systs=None — no systematics; only MC stat error is shown.
        syst_dict = {}
        systs_arr = np.zeros(len(bins)-1)
        calc_separate_mcstat = True

    # MC stat variance — added when not already folded into the syst covariance.
    # For weighted MC the per-bin variance is sum(w^2); unweighted reduces to Poisson N.
    if calc_separate_mcstat:
        stats_var = get_hist1d(data=df[var],
                               weights=np.square(df['weights_mc']) if weight else None,
                               bins=bins, overflow=overflow)
        stats_err = np.sqrt(stats_var) * scale
        total_cov += np.diag(stats_var)
    else:
        stats_err = np.zeros(len(bins)-1)

    systs_err = systs_arr * scale
    total_cov = total_cov * (scale ** 2)
    if normalize:
        total_integral = np.sum(hists * bin_widths)
        hists = hists / total_integral
        if calc_separate_mcstat:
            stats_err = stats_err / total_integral
        systs_err = systs_err / total_integral
        total_cov = total_cov / (total_integral ** 2)
        
    for i, (key, entry) in enumerate(categories.items()):
        color      = entry["color"]
        plot_label = entry.get("label", key)
        if (mult_factor!= 1.0) & (i==0): plot_label +=  f" [x{mult_factor}]"
        if counts: plot_label += f" ({int(hist_counts[i]):,})" if hist_counts[i] < 1e6 else f"({hist_counts[i]:.2e}"
        if percents: plot_label += f" ({hist_counts[i]/np.sum(hist_counts)*100:.1f}%)"
        bottom=steps[i-1] if i>0 else 0
        # steps needs the first entry to be repeated!
        steps[i] = np.insert(hists[i],obj=0,values=hists[i][0]) + bottom; 
        ax.fill_between(bins, bottom, steps[i], step="pre", 
                         facecolor=mpl.colors.to_rgba(color,alpha),
                         edgecolor=mpl.colors.to_rgba(color,1.0),  
                         lw=1.5, 
                         hatch=hatch[i],zorder=(ncategories-i),label=plot_label)
    
    if plot_err:
        systs_options = {"step": "pre", "color": mpl.colors.to_rgba("gray", alpha=0.75),
                         "lw": 0.0, "facecolor": "none", "hatch": "xxx",
                         "zorder": ncategories + 1}
        stats_options = {"step": "pre", "color": mpl.colors.to_rgba("gray", alpha=0.9),
                         "lw": 0.0, "facecolor": "none", "hatch": "....",
                         "zorder": ncategories + 1}

        has_systs = np.any(systs_arr > 0)
        # fill_between needs the first bin edge repeated
        _systs = np.append(systs_err[0], systs_err)
        _stats = np.append(stats_err[0], stats_err)

        if has_systs and not calc_separate_mcstat:
            # Cases 2 and 1+mcstat: stat is already folded into the covariance → combined band.
            ax.fill_between(bins,
                            steps[-1] - _systs, steps[-1] + _systs,
                            **systs_options, label="MC stat.+syst.")
        elif has_systs and calc_separate_mcstat:
            # Case 1 without mcstat: draw syst and stat bands separately.
            min_syst = steps[-1] - _systs
            pls_syst = steps[-1] + _systs
            ax.fill_between(bins, min_syst, pls_syst, **systs_options, label="MC syst.")
            ax.fill_between(bins, min_syst - _stats, min_syst, **stats_options, label="MC stat.")
            ax.fill_between(bins, pls_syst, pls_syst + _stats, **stats_options)
        else:
            # Case 3: no systematics — stat error only.
            ax.fill_between(bins,
                            steps[-1] - _stats, steps[-1] + _stats,
                            **stats_options, label="MC stat.")

    cut_line_zorder = ncategories + 2
    if cut_val != None:
        for i in range(len(cut_val)):
            ax.axvline(cut_val[i],lw=2,color="gray",linestyle="--",zorder=cut_line_zorder)
    
    total_err = np.sqrt(np.clip(np.diag(total_cov), a_min=0.0, a_max=None))
    syst_dict['__total_cov__'] = total_cov

    ax.set_xlabel('_'.join(var)) if xlabel == "" else ax.set_xlabel(xlabel)
    ax.set_ylabel("Counts")      if ylabel == "" else ax.set_ylabel(ylabel)
    ax.set_title ('_'.join(var)) if title  == "" else ax.set_title (title)
    annotate_internal(ax)
    
    if bin_labels is not None:
        ax.set_xticks(bins)
        ax.set_xticklabels(bin_labels)
    
    # Apply legend with custom kwargs
    default_legend_kwargs = {'ncol': 2, 'loc': 'upper right'}
    if legend_kwargs:
        default_legend_kwargs.update(legend_kwargs)
    legend = ax.legend(**default_legend_kwargs)
    legend.set_zorder(cut_line_zorder + 1)

    return bins, steps, total_err, syst_dict

def plot_var_pdg(**args):
    """Backward-compatible wrapper: calls :func:`plot_var` with ``pdg=True``.

    All keyword arguments are forwarded unchanged. See :func:`plot_var` for the
    full parameter list and the 4-tuple return value.
    """
    return plot_var(pdg=True,**args)

def data_plot_overlay(df: pd.DataFrame,
                      var: str | tuple,
                      bins: list[float] | np.ndarray,
                      ax = None,
                      normalize: bool = False,
                      overflow: bool = True) -> tuple[np.ndarray, np.ndarray, object]:
    """Overlay data as points with Poisson errors on an axis.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the data to plot.
    var : str | tuple
        Column name (or multi-index tuple) to histogram.
    bins : array-like
        Bin edges for the histogram.
    ax : matplotlib.axes.Axes, optional
        Axis to draw on. If None the current axis is used.
    normalize : bool, default False
        If True, normalize the histogram by its integral (uses bin widths).
    overflow : bool, default True
        If True, fold values above ``bins[-1]`` into the last bin.

    Returns
    -------
    hist : np.ndarray
        Per-bin counts (or normalized values).
    errors : np.ndarray
        Per-bin Poisson errors (sqrt of raw counts, then rescaled if normalized).
    plot : matplotlib.Artist
        The object returned by ``ax.errorbar``.
    """
    if ax is None:
        ax = plt.gca()

    if isinstance(df, pd.DataFrame):
        df = ensure_lexsorted(df, axis=0)
        df = ensure_lexsorted(df, axis=1)

    hist = get_hist1d(data=df[var], bins=bins, overflow=overflow)
    errors = np.sqrt(hist)
    bin_widths = np.diff(bins)

    label = "data" 
    label += f" ({np.sum(hist,dtype=int):,})" if np.sum(hist) < 1e6 else f"({np.sum(hist):.2e})"
    
    if normalize:
        total_integral = np.sum(hist * bin_widths)
        hist = hist / total_integral
        errors = errors / total_integral
    
    bin_centers = 0.5*(bins[1:] + bins[:-1])
    plot = ax.errorbar(bin_centers, hist, yerr=errors, fmt='.',color='black',zorder=1e3,label=label)
    return hist, errors, plot

def plot_mc_data(mc_df: pd.DataFrame,
                 data_df: pd.DataFrame,
                 var: str | tuple,
                 bins: list[float] | np.ndarray,
                 bin_labels: list[str] | None = None,
                 figsize: tuple[int, int] = (7, 6),
                 ratio_min: float = 0.0,
                 ratio_max: float = 2.0,
                 annot: bool = True,
                 savefig: str = "",
                 config: PlottingConfig | None = None,
                 **kwargs) -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    """Create a combined MC stack + data overlay plot with a data/MC ratio subplot.

    Calls :func:`plot_var` for the MC stack and :func:`data_plot_overlay` for the
    data points, then draws a ratio panel and annotates with the integrated Data/MC
    ratio and a chi-squared goodness-of-fit test.

    Parameters
    ----------
    mc_df : pandas.DataFrame
        MC dataframe passed to :func:`plot_var`.
    data_df : pandas.DataFrame
        Observed-data dataframe passed to :func:`data_plot_overlay`.
    var : str | tuple
        Column (or multi-index tuple) to histogram.
    bins : array-like
        Bin edges for the histograms.
    bin_labels : list of str, optional
        Custom tick labels placed at each bin edge on both axes.
    figsize : tuple, default (7, 6)
        Figure size passed to ``plt.figure``.
    ratio_min, ratio_max : float, default (0.0, 2.0)
        y-axis limits for the ratio subplot.
    annot : bool, default True
        If True, annotate the main axis with the integrated Data/MC ratio and
        the chi-squared / p-value.
    savefig : str, optional
        If non-empty, save the figure to this path with ``bbox_inches='tight'``.
    config : PlottingConfig, optional
        Style/display options. Keyword arguments take priority.
    **kwargs
        Forwarded to :func:`plot_var` (e.g. ``scale``, ``pdg``, ``xlabel``,
        ``systs``, ``hatch``, ``normalize``, ``legend_kwargs``).

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax_main : matplotlib.axes.Axes
        The upper (MC stack + data) axis.
    ax_sub : matplotlib.axes.Axes
        The lower (data/MC ratio) axis.
    mc_dict : dict
        The syst dict returned by :func:`plot_var`, including ``'__total_cov__'``.
    """
    _p = {f.name: getattr(config, f.name) for f in _dc_fields(config)} if config is not None else {}
    _p.update(kwargs)
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 1, height_ratios=[6, 1], hspace=0.4)
    ax_main = fig.add_subplot(gs[0])
    ax_sub = fig.add_subplot(gs[1])

    data_args = dict(df=data_df, var=var, bins=bins, ax=ax_main, normalize=_p.get('normalize', False), overflow=_p.get('overflow', True))
    mc_args   = dict(df=mc_df, var=var, bins=bins, ax=ax_main, config=config, **kwargs)

    data_hist, data_err, data_plot = data_plot_overlay(**data_args)
    mc_bins, mc_steps, mc_err, mc_dict = plot_var(**mc_args)
    
    xmin, xmax = ax_main.get_xlim()
    
    # plot the ratio
    mc_tot = mc_steps[-1][1:]  # last step contains the total MC counts
    fig.canvas.draw()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",message="invalid value encountered in divide")
        # ratio is (data bin content) / (mc bin content)
        ratio = data_hist / mc_tot
        # error in ratio is just (data error) / (mc bin content)
        ratio_err = data_err / mc_tot
        # error in shading should just be (mc error) / (mc bin content)
        mc_contribution = mc_err/mc_tot
        # shading is around unity    
        ps_err = 1 + np.append(mc_contribution[0],mc_contribution)
        ms_err = 1 - np.append(mc_contribution[0],mc_contribution)

    nbins = len(bins)-1
    mc_total_cov = mc_dict.get('__total_cov__') if isinstance(mc_dict, dict) else None
        
    bin_centers = 0.5 * (mc_bins[1:] + mc_bins[:-1])
    
    ax_sub.errorbar(bin_centers, ratio, yerr=ratio_err, fmt='s', markersize=3,color='black', zorder=1e3, label='data/MC ratio')
    # fill_between needs last entry to be repeated 
    ax_sub.fill_between(mc_bins,ms_err, ps_err, step="pre", color=mpl.colors.to_rgba("gray", alpha=0.4), lw=0.0, label='MC err.')
    
    ax_sub.axhline(1, color='red', linestyle='--', linewidth=1, zorder=0,label="y=1.0")
    ax_sub.set_xlim(xmin, xmax)
    ax_sub.set_ylim(ratio_min, ratio_max)
    ax_sub.set_ylabel("Data/MC")
    ax_sub.legend(loc='upper center', bbox_to_anchor=(0.5, 1.5),
                  ncol=3,fontsize='small',frameon=False)
    cut_val = _p.get('cut_val', None)
    if cut_val is not None:
        for cut in cut_val:
            # ax_main.axvline(cut, color='black', linestyle='--', linewidth=2, alpha=0.5, zorder=1e2)
            ax_sub.axvline (cut, color='black', linestyle='--', linewidth=2, alpha=0.5, zorder=1e2)

    total_data = np.sum(data_hist)
    total_mc   = np.sum(mc_tot)
    total_ratio = total_data / total_mc

    # MC covariance matrix — full 2D when cases 1/2 were used in plot_var,
    # diagonal stat-only for case 3.
    has_full_cov = isinstance(mc_total_cov, np.ndarray) and mc_total_cov.shape == (nbins, nbins)
    mc_cov = mc_total_cov if has_full_cov else np.diag(np.square(mc_err))

    # Combined covariance for chi-sq: data (Poisson diagonal) + MC.
    data_cov   = np.diag(np.square(data_err))
    counts_cov = data_cov + mc_cov

    # Integrated ratio uncertainty.
    # Cases 1 & 2: propagate full covariance — sigma_mc = sqrt(sum(mc_cov)) * R / total_mc.
    # Case 3: same formula, but mc_cov is diagonal so sum(mc_cov) = sum of stat variances.
    total_ratio_mc_err   = np.sqrt(np.sum(mc_cov)) * (total_ratio / total_mc)
    total_ratio_data_err = np.sqrt(total_data) / total_mc
    total_ratio_err      = np.sqrt(total_ratio_data_err**2 + total_ratio_mc_err**2)

    valid = np.isfinite(data_hist) & np.isfinite(mc_tot)
    ndf     = nbins
    chi2    = np.nan
    p_value = np.nan
    if np.count_nonzero(valid) > 0:
        delta   = data_hist[valid] - mc_tot[valid]
        cov_sel = counts_cov[np.ix_(valid, valid)]
        try:
            chi2 = float(delta.T @ np.linalg.pinv(cov_sel) @ delta)
            if chi2_dist is not None and np.isfinite(chi2):
                p_value = float(chi2_dist.sf(chi2, df=ndf))
        except np.linalg.LinAlgError:
            chi2    = np.nan
            p_value = np.nan

    fig.canvas.draw()
    legend_loc  = str((_p.get('legend_kwargs') or {}).get('loc', '')).lower()
    main_legend = ax_main.get_legend()

    if main_legend is not None:
        renderer   = fig.canvas.get_renderer()
        legend_box = main_legend.get_window_extent(renderer).transformed(ax_main.transAxes.inverted())
        ann_fontsize = main_legend.get_texts()[0].get_fontsize() if main_legend.get_texts() else 'small'
    else:
        legend_box, ann_fontsize = None, 'small'

    if 'right' in legend_loc:
        anchor_right = True
    elif 'left' in legend_loc or 'center' in legend_loc:
        anchor_right = False
    else:
        anchor_right = legend_box is not None and legend_box.x0 > 0.5

    if legend_box is not None:
        ann_x, ann_y = (legend_box.x1 if anchor_right else legend_box.x0), legend_box.y0
    else:
        ann_x, ann_y = (0.98, 0.98) if anchor_right else (0.02, 0.98)
    ann_ha = 'right' if anchor_right else 'left'

    if annot:
        ax_main.annotate(rf"$\Sigma$ Data/MC = {total_ratio:.2f} $\pm$ {total_ratio_err:.2f}",
                        xy=(ann_x, ann_y),
                        xycoords=ax_main.transAxes,
                        xytext=(0, -6),
                        textcoords='offset points',
                        ha=ann_ha, va='top', fontsize=ann_fontsize)
        
        ax_main.annotate(rf"$\chi^2$/ndf = {chi2:.1f}/{ndf}, $p$ = {p_value:.2g}",
                        xy=(ann_x, ann_y),
                        xycoords=ax_main.transAxes,
                        xytext=(0, -20),
                        textcoords='offset points',
                        ha=ann_ha, va='top', fontsize=ann_fontsize)

    if bin_labels is not None:
        ax_main.set_xticks(bins)
        ax_main.set_xticklabels(bin_labels)
        ax_sub.set_xticks(bins)
        ax_sub.set_xticklabels(bin_labels)
    annotate_internal(ax_main)

    if savefig!="":
        plt.savefig(savefig,bbox_inches='tight')
    
    return fig, ax_main, ax_sub, mc_dict