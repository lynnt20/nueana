import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import pandas as pd

from .constants import signal_dict, signal_labels, pdg_dict, signal_colors, pdg_colors
from .utils import get_slices, get_evt

def plot_var(df, 
             var: tuple | str, 
             bins: np.ndarray,
             ax = None,
             xlabel: str = "", 
             ylabel: str= "",
             title: str = "",
             counts: bool = False,
             scale: list = None,
             normalize: bool = False,
             mult_factor: float = 1.0,
             cut_val: list = None,
             plot_err: bool = True,
             pdg: bool = False,
             pdg_col: tuple | str = 'pfp_shw_truth_p_pdg',
             hatch: list = None,):
    """
    Plots a variable for each interaction type in a histogram.
    
    Parameters
    ----------
    df: input dataframe or list of dataframes
    var: tuple | str
        variable to be plotted, must be a column in the dataframe
    bins: np.ndarray
        histogram binning
    ax: matplotlib.axes.Axes
        axes to plot on, if None, uses the current axes
    label: str
        x-axis label. If empty, uses the variable name
    ylabel: str
        y-axis label.
    title: str
        title of the plot
    counts: bool
        if True, adds the number of events to the label for the legends
    scale: list of floats
        if specified, scales the histograms from df by the specified values 
    normalize: bool
        if True, normalizes the histogram such that the area under the curve sums to 1
    mult_factor: float
        if specified, multiplies the signal histogram by this factor
    cut_val: list
        if specified, plots vertical lines at the specified values
    plot_err: bool
        if True, adds hatching to plot statistical error
    pdg: bool
        if True, splits by true PDG instead of signal type 
    pdg_col: tuple | str
        The column that stores true pdg on a pfp level. 
        Should be `pfp_shw_truth_p_pdg` for flattened df. 
    hatch: list
        if provided, will add hatching to the histogram
    """
    if (type(df) is not list): df = [df]
    if scale == None: scale = list(np.ones(len(df)))
    if (len(scale) != len(df)): 
        print("Error: scale must be the same length as df")
        return 

        
    # Defensive: ensure DataFrame axes are fully lexsorted when using MultiIndex
    # This avoids pandas PerformanceWarning about indexing past lexsort depth
    def _ensure_lexsorted(frame, axis):
        # axis: 0 -> index, 1 -> columns
        idx = frame.index if axis == 0 else frame.columns
        if isinstance(idx, pd.MultiIndex) and getattr(idx, "lexsort_depth", 0) < idx.nlevels:
            # sort by all levels (returns a new frame)
            return frame.sort_index(axis=axis)
        return frame

    # Apply to each provided dataframe (both row-index and columns)
    for ii, this_df in enumerate(df):
        if isinstance(this_df, pd.DataFrame):
            this_df = _ensure_lexsorted(this_df, axis=0)
            this_df = _ensure_lexsorted(this_df, axis=1)
            df[ii] = this_df
    
    colors = pdg_colors if pdg else signal_colors
    if ax is None: ax = plt.gca()
    ncategories = len(pdg_dict)+2 if pdg else len(signal_dict)
    if hatch == None: hatch = [""]*ncategories
    alpha = 0.4 
    
    bin_steps   = bins[1:]    
    hists   = np.zeros((ncategories,len(bin_steps))) # this is for storing the histograms
    steps   = np.zeros((ncategories,len(bins))) # this is for plotting
    
    binned_stats = np.zeros((len(bin_steps),len(df)))
    binned_err   = np.zeros(len(bin_steps))
    
    df_counter = 0
    if pdg==False: 
        for this_df, this_scale in zip(df,scale):
            for i, entry in enumerate(signal_dict):
                this_signal_val = signal_dict[entry]
                hist, bin_edges = np.histogram(this_df[this_df.signal==this_signal_val][var],bins=bins)
                binned_stats[:,df_counter] += hist
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
                binned_stats[:,df_counter] += hist
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

    # statistical error is only on the input statistics, and the we rescale 
    for i in range(len(bin_steps)): 
        binned_err[i] = np.sqrt(np.sum(binned_stats[i,:]*(np.array(scale)**2)))
    
    if normalize:
        norm_factor = np.sum(hists) * np.diff(bins)
        hists = hists / norm_factor
        binned_err = binned_err / norm_factor
        
    for i in range(ncategories):
        plot_label = (list(pdg_dict.keys())+['cosmic']+['other'])[i] if pdg else signal_labels[i]
        if (mult_factor!= 1.0) & (i==0): plot_label +=  f" [x{mult_factor}]"
        if counts: plot_label += f" ({int(hist_counts[i]):,})" if hist_counts[i] < 1e6 else f"({hist_counts[i]:.2e}"
        
        bottom=steps[i-1] if i>0 else 0
        steps[i] = np.insert(hists[i],obj=0,values=hists[i][0]) + bottom; 

        ax.fill_between(bins, bottom, steps[i], step="pre", 
                         facecolor=mpl.colors.to_rgba(colors[i],alpha),
                         edgecolor=mpl.colors.to_rgba(colors[i],1.0),
                         lw=1.5, 
                         hatch=hatch[i],zorder=(ncategories-i),label=plot_label)
    if plot_err: 
        # fill between needs the last entry to be repeated...
        minus_err = steps[-1] - np.append(binned_err,binned_err[-1])
        plus_err  = steps[-1] + np.append(binned_err,binned_err[-1])
        ax.fill_between(bins, minus_err, plus_err, step="pre", 
                        color=mpl.colors.to_rgba("gray", alpha=0.8),
                        lw=0.0,facecolor="none",
                        hatch="xxxxx",
                        zorder=ncategories+1,label="MC stat. err.")

    if cut_val != None:
        for i in range(len(cut_val)):
            ax.axvline(cut_val[i],lw=2,color="gray",linestyle="--",zorder=6)

    # * if this is a multiindex dataframe, recast `var` 
    # * (since we're done using it) to be nice as a string!
    if type(var)==tuple: var = '_'.join(var)
    ax.set_xlabel(var)      if xlabel == "" else ax.set_xlabel(xlabel)
    ax.set_ylabel("Counts") if ylabel == "" else ax.set_ylabel(ylabel)
    ax.set_title (var)      if title  == "" else ax.set_title (title)
    ax.legend()

    return bins, steps, binned_err

# added for backward compatibility 
def plot_var_pdg(**args):
    return plot_var(pdg=True,**args)

def plot_mc_data(mc_dfs, data_df, var, bins, 
                 scale=None, pdg=False,
                 xlabel="", ylabel="", title="", 
                 normalize=False,
                 figsize = (7, 6),
                 cut_val = [],
                 ratio_min=0.0, ratio_max=2.0,
                 hatch=None,
                 savefig=""):
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 1, height_ratios=[6, 1], hspace=0.4)
    ax_main = fig.add_subplot(gs[0])
    ax_sub = fig.add_subplot(gs[1])

    data_hist, data_err, data_plot= data_plot_overlay(data_df, var, bins, ax=ax_main, normalize=normalize)
    mc_bins, mc_steps, mc_err = plot_var(df=mc_dfs, var=var, bins=bins, scale=scale,ax=ax_main, normalize=normalize,pdg=pdg,
                                         xlabel=xlabel,ylabel=ylabel,title=title,hatch=hatch)
    
    xmin, xmax = ax_main.get_xlim()
    
    # plot the ratio
    mc_tot = mc_steps[-1][1:]  # last step contains the total MC counts
    data_mc_ratio = data_hist / mc_tot
    data_mc_ratio_err = data_mc_ratio * np.sqrt((data_err / data_hist)**2 + (mc_err / mc_tot)**2)
    mc_contribution = mc_err/mc_tot
    bin_centers = 0.5 * (mc_bins[1:] + mc_bins[:-1])
    ax_sub.errorbar(bin_centers, data_mc_ratio, yerr=data_mc_ratio_err, fmt='s', markersize=3,color='black', zorder=1e3, label='data/MC ratio')
    for i in range(len(mc_contribution)):
        ax_sub.fill_between(mc_bins[i:i+2],1 - mc_contribution[i], 1 + mc_contribution[i], step="post",
                            color=mpl.colors.to_rgba("gray", alpha=0.4),
                            lw=0.0, label='MC stat. err.' if i==0 else "")
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