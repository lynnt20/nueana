import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

from .constants import signal_dict, signal_labels, pdg_dict, colors
from .utils import get_slices, get_evt

def plot_var(df, var: str, bins: np.ndarray,
             ax = None,
             label: str = "", 
             ylabel: str= "",
             title: str = "",
             stacked: bool = True,
             count: bool = False,
             scale: list = None,
             normalize: bool = False,
             mult_factor: float = None,
             cut_val: list = None,
             plot_err: bool = True):
    """
    Plots a variable for each interaction type in a histogram.
    
    Parameters
    ----------
    df: input dataframe or list of dataframes
    var: str
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
    stacked: bool
        if True, plots a stacked histogram
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
    """
    if (type(df) is not list): df = [df]
    if scale == None: scale = list(np.ones(len(df)))
    if (len(scale) != len(df)): 
        print("Error: scale must be the same length as df")
        return 
    if mult_factor==None: mult=1.0
    else: mult=mult_factor
    if ax is None:
        ax = plt.gca()
    
    bin_steps   = bins[1:]
    
    hists   = np.zeros((len(signal_dict),len(bin_steps)))
    steps   = np.zeros((len(signal_dict),len(bins))) # for the step plot
    bottom  = np.zeros((len(signal_dict),len(bins))) # for the bar plot 
    order   = np.arange(len(signal_dict),0,step=-1)
    counts  = np.zeros(len(signal_dict))
    labels  = []
    rectangles = []

    binned_stats = np.zeros((len(bin_steps),len(df)))
    binned_err   = np.zeros(len(bin_steps))

    df_counter = 0
    for this_df, this_scale in zip(df,scale):
        for i, entry in enumerate(signal_dict):
            this_signal_val = signal_dict[entry]
            hist, bin_edges = np.histogram(this_df[this_df.signal==this_signal_val][var],bins=bins)
            binned_stats[:,df_counter] += hist
            counts[i] +=  len(this_df[this_df.signal==this_signal_val][var])*this_scale
            hists[i] = hists[i] + this_scale*hist
        df_counter += 1

    
    for i in range(len(bin_steps)): 
        binned_err[i] = np.sqrt(np.sum(binned_stats[i,:]*(np.array(scale)**2)))
        
    hists[0] = mult*hists[0]
    
    if normalize:
        norm_factor = np.sum(hists) * np.diff(bins)
        hists = hists / norm_factor
        binned_err = binned_err / norm_factor
    
    for i, entry in enumerate(signal_dict):
        plot_label = signal_labels[i]
        
        if ((mult_factor!=None) & (i==0)): plot_label = signal_labels[i] + f" [x{mult_factor}]"
        if count==True: 
            this_count = int(counts[i])
            if (this_count < 1e6):
                plot_label = plot_label + f" ({this_count:,})"
            else: 
                plot_label = plot_label + f" ({this_count:.2e})"
        if stacked==False:
            ax.step(bins, np.insert(hists[i],obj=0,values=hists[i][0]),zorder=order[i],color=colors[i])
        if stacked==True: 
            # need to append a value at the beginning so that the step will be plotted across the first bin
            if i==0: 
                steps[i] = np.insert(hists[i],obj=0,values=hists[i][0]);
            else:    
                steps[i] = np.insert(hists[i],obj=0,values=hists[i][0]) + steps[i-1]; 
                bottom[i] = steps[i-1]
            ax.step(bins,steps[i], zorder=order[i],color=colors[i])
            ax.fill_between(bins,bottom[i],steps[i],step="pre",alpha=0.4,color=colors[i],zorder=order[i])
            facecolor = mpl.colors.to_rgba(colors[i], alpha=0.4)
            edgecolor = mpl.colors.to_rgba(colors[i])
            rectangles.append(plt.Rectangle((0, 0), 1, 1, fc=facecolor, edgecolor=edgecolor,linewidth=2))
            labels.append(plot_label)
    
    if plot_err:
        for i in range(len(binned_err)):
            ax.fill_between(bins[i:i+2], steps[-1][i+1]-binned_err[i], steps[-1][i+1]+binned_err[i], step="pre", 
                            color=mpl.colors.to_rgba("gray", alpha=0.8),
                            facecolor="none",
                            hatch="xxxxx",
                            lw=0.0,
                            zorder=np.max(order)+1)
        rectangles.append(plt.Rectangle((0, 0), 1, 1, fc="none", edgecolor=mpl.colors.to_rgba("gray", alpha=0.4), hatch="xxxxx", linewidth=0.0))
        labels.append("MC stat. err.")

    # fix for strange ylim behavior 
    # ymin, ymax = ax.set_ylim()
    if cut_val != None:
        for i in range(len(cut_val)):
            ax.axvline(cut_val[i],lw=2,color="gray",linestyle="--",zorder=6)
    # ax.set_ylim(0,ymax)

    if label == "": ax.set_xlabel(var)
    else: ax.set_xlabel(label)
    if title == "": ax.set_title(var+" distribution (for each interaction type)")
    else: ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(rectangles,labels)
    return bins, steps, binned_err

def plot_var_pdg(df, var: str, bins: np.ndarray,
                 ax= None,
                 label: str = "", title: str = "",
                 scale: list = None,
                 normalize: bool = False,
                 cut_val: list = None,
                 pdg_colors: list = None,
                 pdg_hatch: list = None,
                 plot_err: bool = True):    
    """
    Plots a variable for each pdg in a stacked histogram.
    
    Parameters
    ----------
    df: input dataframe or list of dataframes
    var: str
        variable to be plotted, must be a column in the dataframe
    bins: np.ndarray
        histogram binning
    ax: matplotlib.axes.Axes
        axes to plot on, if None, uses the current axes
    label: str
        x-axis label
    title: str
        title of the plot
    scale: list of floats
        if specified, scales the histograms from df by the specified values 
    normalize: bool
        if True, normalizes the histogram such that the area under the curve sums to 1
    cut_val: list
        if specified, plots vertical lines at the specified values
    pdg_colors: list
        if specified, uses the specified colors for each pdg
    pdg_hatch: list
        if specified, uses the specified hatches for each pdg
    plot_err: bool
        whether to plot the MC statistical err, default True
    """
    if (type(df) is not list): df = [df]
    if scale == None: scale = list(np.ones(len(df)))
    if (len(scale) != len(df)): 
        print("Error: scale must be the same length as df")
        return
    if ax is None:
        ax = plt.gca()
        
    if pdg_colors == None: pdg_colors = colors
    
    bin_steps   = bins[1:]
    
    # add 2 entries for 'other' and 'cosmic'
    hists   = np.zeros((len(pdg_dict)+2,len(bin_steps)))
    steps   = np.zeros((len(pdg_dict)+2,len(bins)))
    order   = np.arange(len(pdg_dict)+2,0,step=-1)
    
    binned_stats = np.zeros((len(bin_steps),len(df)))
    binned_err   = np.zeros(len(bin_steps))

    other_df = []
    cosmic_df = []
    for idx in range(len(df)):
        this_df = df[idx]; this_scale = scale[idx]; this_other = df[idx].copy()
        this_nu_df = df[idx].query(f"signal < " + str(signal_dict['cosmic']))
        this_cosmic_df =  df[idx].query(f"signal == " + str(signal_dict['cosmic']))
        for i, key in enumerate(list(pdg_dict.keys())):
            pdg_value = pdg_dict[key]['pdg']
            pdg_df = this_nu_df[abs(this_nu_df.pfp_shw_truth_p_pdg)==pdg_value]
            hist, _____ = np.histogram(pdg_df[var],bins=bins)
            binned_stats[:,idx] += hist
            hists[i] = hists[i] + this_scale*hist
            # remove the 'good' pdg from
            this_other = this_other[abs(this_other.pfp_shw_truth_p_pdg)!=pdg_value]
        other_df.append(this_other)
        cosmic_df.append(this_cosmic_df)
    
    for i in range(len(bin_steps)): 
        binned_err[i] = np.sqrt(np.sum(binned_stats[i,:]*(np.array(scale)**2)))
        
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

    if normalize:
        norm_factor = np.sum(hists) * np.diff(bins)
        hists = hists / norm_factor
        binned_err = binned_err / norm_factor

    for i in range(len(hists)): 
        plot_label = (list(pdg_dict.keys())+['cosmic']+['other'])[i]
        if i == len(hists)-1: plot_label = "other"
        
        bottom = steps[i-1] if i>0 else 0
        steps[i] = np.insert(hists[i],obj=0,values=hists[i][0]) + bottom; 
        
        ax.step(bins,steps[i], label=plot_label, zorder=(len(hists)-i),color=pdg_colors[i])
        if pdg_hatch == None: pdg_hatch = [""]*len(hists)
        ax.fill_between(bins, bottom, steps[i], step="pre", alpha=0.25, color=pdg_colors[i])
        ax.fill_between(bins, bottom, steps[i], step="pre", 
                         facecolor=mpl.colors.to_rgba(pdg_colors[i],0.0),
                         edgecolor=mpl.colors.to_rgba(pdg_colors[i],0.5), 
                         hatch=pdg_hatch[i])

    if plot_err:
        for i in range(len(binned_err)):
            label = "MC stat. err." if i==0 else ""
            ax.fill_between(bins[i:i+2], steps[-1][i+1]-binned_err[i], steps[-1][i+1]+binned_err[i], step="pre", 
                            color=mpl.colors.to_rgba("gray", alpha=0.8),
                            facecolor="none",
                            hatch="xxxxx",
                            lw=0.0,
                            zorder=np.max(order)+1,
                            label=label)
    
    if cut_val != None:
        for i in range(len(cut_val)):
            ax.axvline(cut_val[i],lw=2,color="gray",linestyle="--",zorder=6)

    if label == "": ax.set_xlabel(var)
    else: ax.set_xlabel(label)
    if title == "": ax.set_title(var+" distribution (separated by particle type)")
    else: ax.set_title(title)

    ax.legend()
    return bins, steps, binned_err


def data_plot_overlay(df, var, bins,ax=None,normalize=False):
    if ax is None:
        ax = plt.gca()
    hist, edges = np.histogram(df[var], bins=bins)
    errors = np.sqrt(hist)
    if normalize:
        total_area = np.sum(hist)*np.diff(edges)
        hist = hist/(total_area)
        errors = errors/(total_area)
    bin_centers = 0.5*(edges[1:] + edges[:-1])
    plot = ax.errorbar(bin_centers, hist, yerr=errors, fmt='.',color='black',zorder=1e3,label='data')
    return hist, errors, plot

def plot_mc_data(mc_dfs, data_df, var, bins, 
                 scale=None, pdg=False,
                 xlabel="", ylabel="", title="", 
                 normalize=False,
                 figsize = (7, 6),
                 cut_val = [],
                 ratio_min=0.0, ratio_max=2.0,
                 incolors=None,
                 inhatch=None,
                 savefig=""):
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 1, height_ratios=[6, 1], hspace=0.4)
    ax_main = fig.add_subplot(gs[0])
    ax_sub = fig.add_subplot(gs[1])
    data_hist, data_err, data_plot= data_plot_overlay(data_df, var, bins, ax=ax_main, normalize=normalize)
    if pdg==False:
        mc_bins, mc_steps, mc_err = plot_var(df=mc_dfs, var=var, bins=bins, scale=scale, ax=ax_main, normalize=normalize,
                                             title=" ",label=" ", ylabel=" ")
        # need to add legend manually
        legend_labels = [signal_labels[i] for i in range(len(signal_labels))]
        legend_handles = [Patch(edgecolor=mpl.colors.to_rgba(colors[i],1.0),
                        facecolor=mpl.colors.to_rgba(colors[i],0.25),
                        lw=2) for i in range(len(colors))]
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, fc="none", edgecolor=mpl.colors.to_rgba("gray", alpha=0.4), hatch="xxxxx", linewidth=0.0))
        legend_labels.append("MC stat. err.")
        ax_main.legend(handles=legend_handles+[data_plot], labels =legend_labels+["data"],loc='upper right',ncol=2, fontsize=9)
    else:
        mc_bins, mc_steps, mc_err = plot_var_pdg(df=mc_dfs, var=var, bins=bins, scale=scale,ax=ax_main, normalize=normalize,pdg_colors=incolors,pdg_hatch=inhatch,
                                                 title=" ",label=" ")
        ax_main.legend( fontsize=9)
    ax_main.set_xlabel(xlabel) if xlabel else ax_main.set_xlabel(var)
    ax_main.set_ylabel(ylabel) if ylabel else ax_main.set_ylabel("Counts")
    ax_main.set_title(title) if title else ax_main.set_title(f"{var}")
    
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