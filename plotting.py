"""Useful plotting helpers for nueana: stacked MC, PDG breakdowns, and data overlays.

This module provides:
- plot_var: unified function to plot either signal-type stacks or PDG-type stacks.
- data_plot_overlay: draw data points with Poisson errors on top of MC stacks.
- plot_mc_data: convenience function that builds an MC+data figure with ratio subplot.

All functions accept both plain and MultiIndex DataFrames (the code will attempt to
ensure lexsorted axes via ``ensure_lexsorted`` imported from ``.utils``).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import pandas as pd
import warnings

from .constants import signal_dict, signal_labels, pdg_dict, signal_colors
from .utils import ensure_lexsorted
from .syst import *

def plot_var(df: pd.DataFrame | list[pd.DataFrame],
             var: tuple | str,
             bins: np.ndarray,
             ax = None,
             xlabel: str = "",
             ylabel: str = "",
             title: str = "",
             counts: bool = False,
             scale: float | list[float] | None = None,
             normalize: bool = False,
             mult_factor: float = 1.0,
             cut_val: list[float] | None = None,
             plot_err: bool = True,
             systs: bool = False,
             pdg: bool = False,
             pdg_col: tuple | str = 'pfp_shw_truth_p_pdg',
             hatch: list[str] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plot a variable as stacked histograms for signal categories or PDG types.

    This function supports two modes controlled by ``pdg``:
    - pdg=False (default): stack by interaction type using ``signal_dict``.
    - pdg=True: stack by particle PDG using ``pdg_dict``; adds 'cosmic' and
      'other' as the last two categories.

    Parameters
    ----------
    df : pandas.DataFrame or list[pandas.DataFrame]
        Input dataframe(s). If a single DataFrame is provided it will be wrapped in a list.
    var : tuple | str
        Column name (or multi-index tuple) to histogram.
    bins : np.ndarray
        Bin edges for the histogram.
    ax : matplotlib.axes.Axes, optional
        Axis to draw on. If None the current axis is used.
    xlabel : str, optional
        X axis label. Defaults to the variable name when empty.
    ylabel : str, optional
        Y axis label. Defaults to 'Counts' when empty.
    title : str, optional
        Plot title. Defaults to the variable name when empty.
    counts : bool, default False
        If True, append event counts to legend labels.
    scale : list[float], optional
        Per-DataFrame scale factors. If None, all scales are 1.0. Length must equal number of
        input DataFrames.
    normalize : bool, default False
        If True, normalize histograms so integral equals 1 (uses bin widths from ``bins``).
    mult_factor : float, default 1.0
        Multiplicative factor applied to the first category (index 0). Intended for quick
        visual scaling only; error propagation is not adjusted at all. 
    cut_val : list, optional
        List of x-values at which to draw vertical cut lines.
    plot_err : bool, default True
        If True, draw MC statistical (and optional systematic) error bands.
    systs : bool, default False
        if True, calculates and plots systematic uncertainties. 
    pdg : bool, default False
        When True, split histograms by PDG (uses ``pdg_col``). Otherwise split by signal type.
    pdg_col : tuple | str, default 'pfp_shw_truth_p_pdg'
        Column (or multi-index tuple) containing the PDG code per particle (used when ``pdg``
        is True).
    hatch : list, optional
        Optional hatch patterns per category.

    Returns
    -------
    bins, steps, total_err
        - bins: the input bin edges
        - steps: array of cumulative step values used for plotting (shape (n_categories, len(bins)))
        - total_err: combined stat + syst per bin (length = n_bins)
    """
    if (type(df) is not list): df = [df]
    if (type(scale)==float or type(scale)==np.float64): scale = [scale]
    if scale == None: scale = list(np.ones(len(df)))
    assert (len(scale) == len(df))

    # Apply to each provided dataframe (both row-index and columns)
    for ii, this_df in enumerate(df):
        if isinstance(this_df, pd.DataFrame):
            this_df = ensure_lexsorted(this_df, axis=0)
            this_df = ensure_lexsorted(this_df, axis=1)
            df[ii] = this_df
    
    colors = signal_colors
    if ax is None: ax = plt.gca()
    ncategories = len(pdg_dict)+2 if pdg else len(signal_dict)
    if hatch == None: hatch = [""]*ncategories
    alpha = 0.25 if pdg else 0.4
    
    bin_steps   = bins[1:]    
    hists   = np.zeros((ncategories,len(bin_steps))) # this is for storing the histograms
    steps   = np.zeros((ncategories,len(bins))) # this is for plotting
    
    stats = np.zeros((len(bin_steps),len(df)))
    stats_err   = np.zeros(len(bin_steps))
    systs_err   = np.zeros(len(bin_steps))

    df_counter = 0
    if pdg==False: 
        for this_df, this_scale in zip(df,scale):
            for i, entry in enumerate(signal_dict):
                this_signal_val = signal_dict[entry]
                hist, bin_edges = np.histogram(this_df[this_df.signal==this_signal_val][var],bins=bins)
                stats[:,df_counter] += hist
                hists[i] = hists[i] + this_scale*hist
            df_counter += 1

    else: 
        # other_df is going to store any particles that we don't specify the pdg of
        other_df = []
        cosmic_df = []
        for this_df, this_scale in zip(df,scale):
            this_other = this_df.copy().sort_index() # for storing anything left over 
            this_nu_df = this_df[this_df.signal < signal_dict['cosmic']].sort_index()
            this_cosmic_df =  this_df[this_df.signal == signal_dict['cosmic']].sort_index()
            for i, key in enumerate(list(pdg_dict.keys())):
                pdg_value = pdg_dict[key]['pdg']
                pdg_df = this_nu_df[abs(this_nu_df[pdg_col])==pdg_value].sort_index()
                hist, _____ = np.histogram(pdg_df[var],bins=bins)
                stats[:,df_counter] += hist
                hists[i] = hists[i] + this_scale*hist
                # remove the "good pdg" from other_df
                this_other = this_other[abs(this_other[pdg_col])!=pdg_value]
            other_df.append(this_other)
            cosmic_df.append(this_cosmic_df)
            df_counter+=1
        # scaling the special dfs (other, cosmic) given multiple input
        for idx in range(len(other_df)):
            this_scale = scale[idx]
            this_other = other_df[idx]
            this_cosmic = cosmic_df[idx]
            if len(this_other)!=0: 
                hist, _____ = np.histogram(this_other[var],bins=bins)
                hists[-1] = hists[-1] + this_scale*hist
            if len(this_cosmic)!=0: 
                hist, _____ = np.histogram(this_cosmic[var],bins=bins)
                hists[-2] = hists[-2] + this_scale*hist 
    
    # ! THIS ASSUMES that the PDG of interest and the signal type of interest are both index 0
    # ! e.g. for nueCC (signal==0), e- is the first entry in the pdg_dict
    hists[0] = mult_factor*hists[0]

    # storing the sum of each category in case we want to display it
    hist_counts = np.sum(hists,axis=1)

    # check if systematic cols are inside the df
    bool found_systs = False
    for col in df[0].columns:
        if "univ_" in list(col):
            found_systs = True
            break
    if (systs==True) & (found_systs): 
        # ! TODO now hardcoded for the first entry
        this_mc = df[0]
        this_sig = this_mc[this_mc.signal < signal_dict['cosmic']]
        syst_dict = get_syst(indf=this_sig,var=var,bins=bins)
        total_cov = np.zeros((len(bins)-1,len(bins)-1))
        for key in syst_dict.keys():
            total_cov += syst_dict[key][1]
        systs_arr = np.reshape(np.sqrt(np.diag(total_cov)),(-1,1))
    else:
        if (systs==True) & (found_systs=False):
            print("can't find universes in the input df, ignoring systematic error bars")
            systs=False
        systs_arr = np.reshape(np.zeros(len(bins)-1),(-1,1))
        
    # statistical error is only on the input statistics, and then we rescale 
    # if we're using multiple input mc that has different scalings, 
    # also need to rescale the syst cov matrices
    for i in range(len(df)):
        stats_err += np.sqrt(stats[:,i])*scale[i]
        systs_err += systs_arr[:,i]*scale[i]

    if normalize:
        norm_factor = np.sum(hists) * np.diff(bins)
        hists /= norm_factor
        stats_err /= norm_factor
        systs_err /= norm_factor    
        
    for i in range(ncategories):
        plot_label = (list(pdg_dict.keys())+['cosmic']+['other'])[i] if pdg else signal_labels[i]
        if (mult_factor!= 1.0) & (i==0): plot_label +=  f" [x{mult_factor}]"
        if counts: plot_label += f" ({int(hist_counts[i]):,})" if hist_counts[i] < 1e6 else f"({hist_counts[i]:.2e}"
        
        bottom=steps[i-1] if i>0 else 0
        # steps needs the first entry to be repeated!
        steps[i] = np.insert(hists[i],obj=0,values=hists[i][0]) + bottom; 
        ax.fill_between(bins, bottom, steps[i], step="pre", 
                         facecolor=mpl.colors.to_rgba(colors[i],alpha) if plot_label.find('cosmic') else mpl.colors.to_rgba(colors[-1],alpha),
                         edgecolor=mpl.colors.to_rgba(colors[i],1.0)   if plot_label.find('cosmic') else mpl.colors.to_rgba(colors[-1],1.0),
                         lw=1.5, 
                         hatch=hatch[i],zorder=(ncategories-i),label=plot_label)
    if plot_err: 
        stats_options = {"step":"pre", "color":mpl.colors.to_rgba("gray", alpha=0.9),
                         "lw":0.0,"facecolor":"none","hatch":"....",
                         "zorder":ncategories+1}
        systs_options = {"step":"pre", "color":mpl.colors.to_rgba("gray", alpha=0.8),
                         "lw":0.0,"facecolor":"none","hatch":"xxx",
                         "zorder":ncategories+1}
        # fill_between needs the *first* entry to be repeated...
        if systs:
            min_systs_err = steps[-1]     - np.append(systs_err[0],systs_err)
            pls_systs_err = steps[-1]     + np.append(systs_err[0],systs_err)
            min_stats_err = min_systs_err - np.append(stats_err[0],stats_err)
            pls_stats_err = pls_systs_err + np.append(stats_err[0],stats_err)
            ax.fill_between(bins, min_systs_err, pls_systs_err, **systs_options,label="MC syst.")
            ax.fill_between(bins, min_systs_err, min_stats_err, **stats_options,label="MC stat.")
            ax.fill_between(bins, pls_systs_err, pls_stats_err, **stats_options)
        else: 
            min_stats_err = steps[-1] - np.append(stats_err,stats_err[-1])
            pls_stats_err = steps[-1] + np.append(stats_err,stats_err[-1])
            ax.fill_between(bins, min_stats_err, pls_stats_err, **stats_options,label="MC stat.")

    if cut_val != None:
        for i in range(len(cut_val)):
            ax.axvline(cut_val[i],lw=2,color="gray",linestyle="--",zorder=6)
    
    total_err = stats_err + systs_err

    # * if this is a multiindex dataframe, recast `var` 
    # * (since we're done using it) to be nice as a string!
    if type(var)==tuple: var = '_'.join(var)
    ax.set_xlabel(var)      if xlabel == "" else ax.set_xlabel(xlabel)
    ax.set_ylabel("Counts") if ylabel == "" else ax.set_ylabel(ylabel)
    ax.set_title (var)      if title  == "" else ax.set_title (title)
    ax.legend(ncol=2)

    return bins, steps, total_err

def plot_var_pdg(**args):
    """Backward-compatible wrapper for plotting by PDG.

    Parameters
    ----------
    **args : dict
        All keyword arguments are forwarded to :func:`plot_var`. Key arguments are
        documented there; this wrapper simply calls ``plot_var(pdg=True, **args)``.

    Returns
    -------
    tuple
        The same (bins, steps, total_err) tuple returned by :func:`plot_var`.
    """
    return plot_var(pdg=True,**args)

def data_plot_overlay(df: pd.DataFrame,
                      var: str | tuple,
                      bins: list[float] | np.ndarray,
                      ax = None,
                      normalize: bool = False) -> tuple[np.ndarray, np.ndarray, object]:
    """Overlay data as points with Poisson errors on an axis.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the data to plot. ``var`` must be a column name or
        a tuple for MultiIndex columns.
    var : str | tuple
        Column to histogram.
    bins : array-like
        Bin edges for the histogram.
    ax : matplotlib.axes.Axes, optional
        Axis to draw on. If None the current axis is used.
    normalize : bool, default False
        If True, normalize the histogram by its integral (uses bin widths).

    Returns
    -------
    hist, errors, plot
        - hist: per-bin counts (or normalized values)
        - errors: per-bin sqrt(hist) (or normalized errors)
        - plot: the Artist returned by ax.errorbar
    """
    if ax is None:
        ax = plt.gca()

    if isinstance(df, pd.DataFrame):
        df = ensure_lexsorted(df, axis=0)
        df = ensure_lexsorted(df, axis=1)

    hist, edges = np.histogram(df[var], bins=bins)
    errors = np.sqrt(hist)
    label = "data" 
    label += f" ({np.sum(hist,dtype=int):,})" if np.sum(hist) < 1e6 else f"({np.sum(hist):.2e}"
    if normalize:
        total_area = np.sum(hist)*np.diff(edges)
        hist = hist/(total_area)
        errors = errors/(total_area)
    bin_centers = 0.5*(edges[1:] + edges[:-1])
    plot = ax.errorbar(bin_centers, hist, yerr=errors, fmt='.',color='black',zorder=1e3,label=label)
    return hist, errors, plot

def plot_mc_data(mc_dfs: pd.DataFrame | list[pd.DataFrame],
                 data_df: pd.DataFrame,
                 var: str | tuple,
                 bins: list[float] | np.ndarray,
                 scale: float | list[float] | None = None,
                 pdg: bool = False,
                 pdg_col: tuple | str = 'pfp_shw_truth_p_pdg',
                 xlabel: str = "",
                 ylabel: str = "",
                 title:  str = "",
                 counts: bool = False, 
                 normalize: bool = False,
                 systs: np.ndarray = np.array([]),
                 figsize: tuple[int, int] = (7, 6),
                 cut_val: list[float] = [],
                 ratio_min: float = 0.0,
                 ratio_max: float = 2.0,
                 hatch: list[str] | None = None,
                 savefig: str = "") -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    """Create a combined MC stack + data overlay plot with data/MC ratio subplot.

    Parameters
    ----------
    mc_dfs : pandas.DataFrame or list[pandas.DataFrame]
        MC dataframe(s) to be stacked. If a single DataFrame is provided it will be wrapped in a list.
    data_df : pandas.DataFrame
        Dataframe containing observed data to overlay as points with errors.
    var : str | tuple
        Column (or multi-index tuple) to histogram.
    bins : array-like
        Bin edges for the histograms.
    scale : float or list[float], optional
        Scaling factors applied to each MC dataframe. If None, all scales are 1.0.
    pdg : bool, default False
        If True, instruct `plot_var` to split MC by PDG instead of signal type.
    pdg_col : tuple | str, default 'pfp_shw_truth_p_pdg'
        Column (or multi-index tuple) containing the PDG code per particle (used when ``pdg``
        is True).
    xlabel, ylabel, title : str, optional
        Labels and title for the main axis. If blank, sensible defaults are used.
    counts : bool, default False
        If True, append event counts to legend labels.
    normalize : bool, default False
        If True, normalize both MC and data histograms to unit area.
    systs : numpy.ndarray, optional
        Optional systematic contribution per bin per MC input (shape (n_bins, n_dfs)).
    figsize : tuple, default (7, 6)
        Figure size.
    cut_val : list, optional
        x-values at which to draw vertical cut lines on both main and ratio axes.
    ratio_min, ratio_max : float, default (0.0, 2.0)
        y-limits for the ratio subplot.
    hatch : list, optional
        Hatch patterns passed to `plot_var`.
    savefig : str, optional
        If provided, path where the figure will be saved (bbox_inches='tight').

    Returns
    -------
    fig, ax_main, ax_sub
        The created matplotlib Figure and the main and ratio Axes.
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 1, height_ratios=[6, 1], hspace=0.4)
    ax_main = fig.add_subplot(gs[0])
    ax_sub = fig.add_subplot(gs[1])

    data_args = dict(df=data_df, var=var, bins=bins, ax=ax_main, normalize=normalize)
    mc_args   = dict(df=mc_dfs,  var=var, bins=bins, ax=ax_main, normalize=normalize,
                     scale=scale, systs=systs, hatch=hatch, counts=counts,
                     xlabel=xlabel, ylabel=ylabel,title=title)

    data_hist, data_err, data_plot = data_plot_overlay(**data_args)
    mc_bins, mc_steps, mc_err      = plot_var(**mc_args,pdg=pdg,pdg_col=pdg_col)
    
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
        
        # sum_ratio = np.sum(data_hist)/np.sum(mc_tot)
        # sum_syst_err = np.sqrt(np.sum(mc_err**2)) / np.sum(mc_tot)
        # sum_stat_err = np.sqrt(np.sum(data_err**2)) / np.sum(mc_tot)
        
        # chisq = np.sum((data_hist - mc_tot)**2/mc_tot)
        
    bin_centers = 0.5 * (mc_bins[1:] + mc_bins[:-1])
    nbins = len(bins)-1
    
    # ax_main.annotate(r"$\chi^2/$dof"+f"={chisq:.1f}/{int(nbins)}",xycoords='axes fraction',xy=(0.025,0.925))
    # ax_main.annotate(r"$\Sigma$data/$\Sigma$MC"+f"={sum_ratio:.2f}"+r"$\pm$"+f"{sum_syst_err:.2f}(sys.)"+r"$\pm$"+f"{sum_stat_err:.2f}(stat.)" ,
    #                  xycoords='axes fraction',xy=(0.025,0.85))
    
    ax_sub.errorbar(bin_centers, ratio, yerr=ratio_err, fmt='s', markersize=3,color='black', zorder=1e3, label='data/MC ratio')
    # fill_between needs last entry to be repeated 
    ax_sub.fill_between(mc_bins,ms_err, ps_err, step="pre", color=mpl.colors.to_rgba("gray", alpha=0.4), lw=0.0, label='MC err.')
    
    ax_sub.axhline(1, color='red', linestyle='--', linewidth=1, zorder=0,label="y=1.0")
    ax_sub.set_xlim(xmin, xmax)
    ax_sub.set_ylim(ratio_min, ratio_max)
    ax_sub.set_ylabel("Data/MC")
    ax_sub.legend(loc='upper center', bbox_to_anchor=(0.5, 1.5),
                  ncol=3,fontsize='small',frameon=False)
    
    if len(cut_val) > 0:
        for cut in cut_val:
            ax_main.axvline(cut, color='black', linestyle='--', linewidth=2, alpha=0.5, zorder=1e2)
            ax_sub.axvline (cut, color='black', linestyle='--', linewidth=2, alpha=0.5, zorder=1e2)
    
    if savefig!="":
        plt.savefig(savefig,bbox_inches='tight')
    
    return fig, ax_main, ax_sub