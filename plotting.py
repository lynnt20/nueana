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
from __future__ import annotations

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

__all__ = [
    'annotate_sbnd',
    'annotate_chisq',
    'plot_var',
    'plot_var_pdg',
    'data_plot_overlay',
    'plot_mc_data',
    'plot_mc_data_ccbc',
    'plot_detvar',
    'plot_syst_category_breakdown',
    'plot_syst_breakdown',
]

from .analysis import (signal_dict, signal_categories, signal_categories_external,
                       generic_dict, generic_categories,
                       pdg_categories,
                       mode_dict, mode_categories,
                       detvar_subcat_dict)
from .utils import ensure_lexsorted
from .syst import get_syst
from .utils import get_hist1d
from .classes import PlottingConfig, VariableConfig, SystematicsInput, SystematicsOutput

# Kept above any per-plot zorder used for stairs, error bands, cut lines, and data
# markers so legends and annotations always sit on top of cut_val dashed lines.
_TEXT_ZORDER = 10_000

def _get_weight_column(df: pd.DataFrame):
    """Return the weights_mc column key in df (flat or MultiIndex), or None."""
    for col in df.columns:
        if isinstance(col, tuple):
            if col[0] == 'weights_mc':
                return col
        elif col == 'weights_mc':
            return col
    return None


def _clipped_minor_locator(xmin, xmax):
    """AutoMinorLocator whose ticks are clipped to [xmin, xmax].

    Keeps the visual axis margin intact while preventing minor ticks
    from appearing in the margin area outside the data range.
    """
    class _L(mpl.ticker.AutoMinorLocator):
        def __call__(self):
            locs = super().__call__()
            return locs[(locs >= xmin) & (locs <= xmax)]
    return _L()


def _draw_step_band(
    ax: plt.Axes,
    bins: np.ndarray,
    frac_err: np.ndarray,
    *,
    center: float = 1.0,
    label: str | None = None,
    **kwargs,
) -> None:
    """Fill a symmetric ±frac_err band around center with step="pre" rendering.

    Prepends frac_err[0] so the band aligns with the leftmost bin edge. center
    may be a scalar (e.g. 1.0 for ratio panels) or an array of length len(bins)
    (e.g. steps[-1] for absolute error bands in main panels). All kwargs are
    forwarded to ax.fill_between.
    """
    err = np.append(frac_err[0], frac_err)
    kwargs.setdefault("step", "pre")
    ax.fill_between(bins, center - err, center + err, label=label, **kwargs)


def annotate_sbnd(ax, internal=True):
    """Stamp a status label in the upper-left and the tune label in the upper-right of *ax*.

    Parameters
    ----------
    internal : bool, default True
        If True, stamp "SBND Internal". If False, stamp "SBND Analysis In Progress".
    """
    label = "SBND Internal" if internal else "SBND Analysis In Progress"
    ax.annotate(label, xy=(0.0, 1.02), xycoords='axes fraction', ha='left',
                color='gray', fontweight='bold', zorder=_TEXT_ZORDER)
    # ax.annotate("GENIE v3.40 AR23_00i_00_000", xy=(1.0, 1.02), xycoords='axes fraction', ha='right', color='gray')


def annotate_chisq(
    ax,
    chisq: float,
    ndof: int,
    xy: tuple = (0.98, 0.02),
    xycoords='axes fraction',
    ha: str = 'right',
    va: str = 'top',
    **kwargs,
) -> None:
    """Annotate an axes with a chi^2/ndof and p-value label.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    chisq : float
        Chi-squared value. Non-finite values are silently skipped.
    ndof : int
        Number of degrees of freedom.
    xy : tuple, optional
        Annotation anchor in the coordinate system given by xycoords.
    xycoords : optional
        Coordinate system for xy (default 'axes fraction').
    ha, va : str, optional
        Horizontal and vertical alignment.
    **kwargs
        Passed to ax.annotate (e.g. fontsize, xytext, textcoords).
    """
    if not np.isfinite(chisq):
        return
    p_str = f"{chi2_dist.sf(chisq, ndof):.2g}" if chi2_dist is not None else "N/A"
    kwargs.setdefault('zorder', _TEXT_ZORDER)
    kwargs.setdefault('bbox', dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor='none', alpha=0.5))
    ax.annotate(
        rf"$\chi^2$/ndf = {chisq:.1f}/{ndof}, $p$ = {p_str}",
        xy=xy,
        xycoords=xycoords,
        ha=ha,
        va=va,
        **kwargs,
    )


def plot_var(indf: pd.DataFrame,
             var: tuple | str,
             bins: np.ndarray,
             ax = None,
             config: PlottingConfig | None = None,
             **kwargs,
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plot a variable as stacked histograms, selectable by category type.

    Category mode is controlled by ``generic``, ``pdg``, and ``mode`` (checked in
    that priority order):

    - default: stack by interaction type (``signal_categories``).
    - ``categories=<dict>``: use any custom category dict, including ``generic_categories``,
      ``signal_categories_external``, or a user-defined scheme. Entries may carry either
      ``"value"`` (int) or ``"values"`` (list of ints) for multi-signal-type merging.
    - ``pdg=True``: stack by leading-particle PDG code (``pdg_categories``).
    - ``mode=True``: stack by GENIE interaction mode (``mode_categories``).

    Parameters
    ----------
    indf : pandas.DataFrame
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
        Scale factor applied to all histogram bins (and error arrays). When
        ``systs`` is a ``SystematicsInput``/``SystematicsOutput``, ``indf``'s
        raw POT is assumed to equal ``systs.mcbnb_pot``; pass
        ``target_pot / mcbnb_pot`` to move the plot onto ``target_pot``.
    normalize : bool, default False
        If True, normalize histograms so the integral equals 1 (uses bin widths).
    mult_factor : float, default 1.0
        Extra multiplicative factor applied to the first category only. Intended for
        quick visual scaling; error propagation is not adjusted.
    cut_val : list of float, optional
        x-values at which to draw vertical dashed cut lines.
    plot_err : bool, default True
        If True, draw MC error bands (stat and/or syst).
    systs : True | SystematicsInput | SystematicsOutput | None, default None
        Controls how uncertainties are computed and displayed:

        - ``True``: read universe columns from ``df`` via :func:`~nueana.syst.get_syst`.
          If an MCstat universe is present the combined stat+syst band is drawn; otherwise
          stat and syst bands are drawn separately.
        - :class:`~nueana.classes.SystematicsInput`: call :func:`~nueana.funcs.get_total_cov`
          on-the-fly with the bundled parameters and use the resulting ``rate_cov``.
        - :class:`~nueana.classes.SystematicsOutput`: use a pre-computed result from
          :func:`~nueana.funcs.get_total_cov`. ``rate_cov`` is assumed to be in
          events² at ``systs.mcbnb_pot``; the caller's ``scale`` moves it onto
          the target POT together with the plotted histograms.
        - ``None`` (default): MC stat error only (diagonal, sum-of-weights-squared).
    pdg : bool, default False
        Stack by PDG code rather than signal type.
    pdg_col : tuple | str, default 'pfp_shw_truth_p_pdg'
        Column containing the PDG code per particle (used when ``pdg=True``).
    mode : bool, default False
        Stack by GENIE interaction mode.
    mode_col : tuple | str, default ('slc', 'truth', 'genie_mode')
        Column containing the GENIE interaction mode values (used when
        ``mode=True``).
    hatch : list of str, optional
        Hatch pattern per category (must match number of categories).
    bin_labels : list of str, optional
        Custom tick labels placed at each bin edge.
    categories : dict, optional
        Custom category dict passed directly; takes priority over all flags.
        Use ``generic_categories`` here for the broad CC/NC/non-FV/dirt/cosmic view.
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
    xlabel          = _p.get('xlabel', '')
    ylabel          = _p.get('ylabel', '')
    title           = _p.get('title', '')
    counts          = _p.get('counts', False)
    percents        = _p.get('percents', False)
    scale           = _p.get('scale', 1.0)
    normalize       = _p.get('normalize', False)
    mult_factor     = _p.get('mult_factor', 1.0)
    cut_val         = _p.get('cut_val', None)
    plot_err        = _p.get('plot_err', True)
    systs           = _p.get('systs', None)
    pdg             = _p.get('pdg', False)
    pdg_col         = _p.get('pdg_col', 'pfp_shw_truth_p_pdg')
    mode            = _p.get('mode', False)
    mode_col        = _p.get('mode_col', ('slc', 'truth', 'genie_mode'))
    hatch           = _p.get('hatch', None)
    bin_labels      = _p.get('bin_labels', None)
    overflow        = _p.get('overflow', True)
    legend_kwargs   = _p.get('legend_kwargs', None)
    internal        = _p.get('internal', True)
    categories_kwarg = _p.get('categories', None)
    if isinstance(indf, pd.DataFrame):
        indf = ensure_lexsorted(indf, axis=0)
        indf = ensure_lexsorted(indf, axis=1)

    _weight_col = _get_weight_column(indf)

    if ax is None: ax = plt.gca()
    if pdg:        categories = pdg_categories
    elif mode:     categories = mode_categories
    elif categories_kwarg is not None: categories = categories_kwarg
    else:          categories = signal_categories
    ncategories = len(categories)
    if hatch is None: hatch = [""]*ncategories
    alpha = 0.25 if pdg else 0.4
    
    hists       = np.zeros((ncategories,len(bins)-1)) # this is for storing the histograms
    steps       = np.zeros((ncategories,len(bins))) # this is for plotting
    bin_widths  = np.diff(bins)
    
    stats       = np.zeros(len(bins)-1)
    stats_err   = np.zeros(len(bins)-1)
    systs_err   = np.zeros(len(bins)-1)
    total_cov   = np.zeros((len(bins)-1, len(bins)-1))

    if (pdg==False) & (mode==False):
        for i, (key, entry) in enumerate(categories.items()):
            vals = entry["values"] if "values" in entry else [entry["value"]]
            mask = indf.signal.isin(vals)
            hists[i] = get_hist1d(data=indf[mask][var],
                                  weights=indf[mask][_weight_col] if _weight_col is not None else None,
                                  bins=bins, overflow=overflow)

    elif mode:
        this_nu    = indf[indf[mode_col] == indf[mode_col]]
        this_other = indf[indf[mode_col] != indf[mode_col]]
        for i, (key, entry) in enumerate(categories.items()):
            if entry["value"] is not None:
                this_cat = entry["value"]
                hists[i] = get_hist1d(data=indf[indf[mode_col]==this_cat][var],
                                      weights=indf[indf[mode_col]==this_cat][_weight_col] if _weight_col is not None else None,
                                      bins=bins, overflow=overflow)
                this_nu = this_nu[this_nu[mode_col] != this_cat]
            elif entry["filter"] == "other_nu":
                hists[i] = get_hist1d(data=this_nu[var],
                                      weights=this_nu[_weight_col] if _weight_col is not None else None,
                                      bins=bins, overflow=overflow)
            elif entry["filter"] == "non_nu":
                hists[i] = get_hist1d(data=this_other[var],
                                      weights=this_other[_weight_col] if _weight_col is not None else None,
                                      bins=bins, overflow=overflow)
    else:
        process_col = tuple(list(pdg_col)[:-1] + ['start_process'])
        # other_df stores any particles that we don't specify the pdg of
        this_nu_df      = indf[indf.signal <  signal_dict['cosmic']]#.sort_index()
        this_cosmic_df  = indf[indf.signal == signal_dict['cosmic']]#.sort_index()
        this_offbeam_df = indf[indf.signal == signal_dict['offbeam']]#.sort_index()
        # really only want to see electrons that are
        # primaries from a FV neutrino interaction
        if process_col in indf.columns:
            where_notprim = ((abs(this_nu_df[pdg_col])==11) &
                              (this_nu_df[process_col] != 0))
        else:
            where_notprim = pd.Series(False, index=this_nu_df.index)
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
                                      weights=pdg_df[_weight_col] if _weight_col is not None else None,
                                      bins=bins, overflow=overflow)
                this_other = this_other[abs(this_other[pdg_col])!=pdg_value]
            else:
                filt = entry["filter"]
                pop = _pdg_populations.get(filt, this_other if filt == "other_nu" else None)
                if pop is not None and len(pop) != 0:
                    hists[i] = get_hist1d(data=pop[var],
                                          weights=pop[_weight_col] if _weight_col is not None else None,
                                          bins=bins, overflow=overflow)
    
    # Verify every row in df contributed to exactly one category bin.
    # Mismatched filter keys, unhandled signal values, or accidental row drops
    # will show up here before they silently skew the ratio or chi-sq.
    _expected_total = get_hist1d(data=indf[var],
                                 weights=indf[_weight_col] if _weight_col is not None else None,
                                 bins=bins, overflow=overflow)
    _actual_total = np.sum(hists, axis=0)
    if np.sum(_expected_total) > 0 and not np.isclose(
        np.sum(_actual_total), np.sum(_expected_total), rtol=1e-5
    ):
        _missing_frac = 1.0 - np.sum(_actual_total) / np.sum(_expected_total)
        warnings.warn(
            f"plot_var: {abs(_missing_frac):.1%} of weighted events are unaccounted for "
            f"({'over' if _missing_frac < 0 else 'under'}-counted). "
            "Check that all category filter keys and signal values cover the full DataFrame.",
            stacklevel=2,
        )

    # ! THIS ASSUMES that the PDG of interest and the signal type of interest are both index 0
    # ! e.g. for nueCC (signal==0), e- is the first entry in the pdg_dict
    hists    *= scale
    hists[0] = mult_factor*hists[0]

    # storing the sum of each category in case we want to display it
    hist_counts = np.sum(hists,axis=1)

    # --- Systematics ---
    # Four cases, resolved before the plot loop:
    #   SystematicsInput  → call get_total_cov on-the-fly; MCstat folded in.
    #   SystematicsOutput → use pre-computed get_total_cov result; MCstat folded in.
    #   True              → read universe columns from df; MCstat separate if no MCstat universe.
    #   None/else         → MC stat error only.
    _mcstat_err_annot = None
    _syst_source = 'none'  # 'none' | 'reweight_only' | 'full' — tags scope of the syst band

    def _apply_syst_output(output):
        """Shared logic for SystematicsInput and SystematicsOutput paths.

        Assumes ``output.rate_cov`` is in events² at ``output.mcbnb_pot`` (the
        invariant enforced by ``get_total_cov``). The caller's ``scale`` then
        moves both hists and cov onto the target POT together.
        """
        _total_cov = np.array(output.rate_cov, dtype=float, copy=True)
        _systs_arr = np.sqrt(np.clip(np.diag(_total_cov), a_min=0.0, a_max=None))
        _syst_dict = dict(output.rate_syst_dict)
        _mcstat_key = next((k for k in _syst_dict if str(k).lower() == 'mcstat'), None)
        _calc_sep   = _mcstat_key is None
        _mcstat_ann = (
            np.sqrt(np.diag(_syst_dict[_mcstat_key]['cov'])) * scale
            if _mcstat_key is not None else None
        )
        return _total_cov, _systs_arr, _syst_dict, _calc_sep, _mcstat_ann

    if isinstance(systs, SystematicsInput) or type(systs).__name__ == 'SystematicsInput':
        # Case 1: call get_total_cov on-the-fly with the bundled parameters.
        from .funcs import get_total_cov
        _output = get_total_cov(reco_df=indf, reco_var=var, bins=bins, **systs.to_kwargs())
        total_cov, systs_arr, syst_dict, calc_separate_mcstat, _mcstat_err_annot = _apply_syst_output(_output)
        _syst_source = 'full'

    elif isinstance(systs, SystematicsOutput) or type(systs).__name__ == 'SystematicsOutput':
        # Case 2: caller already ran get_total_cov and passes the result directly.
        if systs.mcbnb_pot is None:
            raise ValueError("SystematicsOutput.mcbnb_pot is not set; use get_total_cov to produce it")
        total_cov, systs_arr, syst_dict, calc_separate_mcstat, _mcstat_err_annot = _apply_syst_output(systs)
        _syst_source = 'full'

    elif systs is True:
        # Case 3: inherit systematics from universe columns in the dataframe.
        found_systs = any(
            isinstance(col, tuple) and any("univ_" in str(c) for c in col)
            for col in indf.columns
        )
        if not found_systs:
            print("systs=True but no universe columns found; computing stat error only")
            syst_dict = {}
            systs_arr = np.zeros(len(bins)-1)
            calc_separate_mcstat = True
        else:
            syst_dict = get_syst(reco_df=indf, reco_var=var, bins=bins)
            has_mcstat = any(str(k).lower() == 'mcstat' for k in syst_dict)
            for key in syst_dict:
                total_cov += syst_dict[key]['cov']
            systs_arr = np.sqrt(np.clip(np.diag(total_cov), a_min=0.0, a_max=None))
            calc_separate_mcstat = not has_mcstat
            _syst_source = 'reweight_only'

    else:
        # Case 4: systs=None — no systematics; only MC stat error is shown.
        syst_dict = {}
        systs_arr = np.zeros(len(bins)-1)
        calc_separate_mcstat = True

    # MC stat variance — added when not already folded into the syst covariance.
    # For weighted MC the per-bin variance is sum(w^2); unweighted reduces to Poisson N.
    if calc_separate_mcstat:
        stats_var = get_hist1d(data=indf[var],
                               weights=np.square(indf[_weight_col]) if _weight_col is not None else None,
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
        if counts: plot_label += f" ({int(hist_counts[i]):,})" if hist_counts[i] < 1e6 else f"({hist_counts[i]:.2e})"
        if percents: plot_label += f" ({hist_counts[i]/np.sum(hist_counts)*100:.1f}%)"
        bottom=steps[i-1] if i>0 else 0
        # steps needs the first entry to be repeated!
        steps[i] = np.insert(hists[i], obj=0, values=hists[i][0]) + bottom
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

        if has_systs:
            # Always combine stat and syst in quadrature into a single band.
            # When MCstat is folded into the covariance (SystematicsInput/Output),
            # stats_err is zero so combined reduces to systs_err unchanged.
            combined_err = np.sqrt(systs_err**2 + stats_err**2)
            _band_label = ("MC stat.+syst.\n(GENIE+Flux+G4)"
                           if _syst_source == 'reweight_only' else "MC stat.+syst.")
            _draw_step_band(ax, bins, combined_err, center=steps[-1],
                            label=_band_label, **systs_options)
        else:
            # systs=None — stat error only.
            _draw_step_band(ax, bins, stats_err, center=steps[-1],
                            label="MC stat.", **stats_options)

    cut_line_zorder = ncategories + 2
    if cut_val is not None:
        for i in range(len(cut_val)):
            ax.axvline(cut_val[i],lw=2,color="gray",linestyle="--",zorder=cut_line_zorder)
    
    total_err = np.sqrt(np.clip(np.diag(total_cov), a_min=0.0, a_max=None))
    syst_dict['__total_cov__']        = total_cov
    syst_dict['__stats_err__']        = stats_err
    syst_dict['__systs_err__']        = systs_err
    syst_dict['__separate_errors__']  = calc_separate_mcstat
    syst_dict['__syst_source__']      = _syst_source
    # MCstat for annotation: from the SystematicsInput path if available, else stats_err.
    syst_dict['__mcstat_err__']       = _mcstat_err_annot if _mcstat_err_annot is not None else stats_err

    _var_str = var if isinstance(var, str) else '_'.join(var)
    ax.set_xlabel(_var_str) if xlabel == "" else ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Counts") if ylabel == "" else ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title (_var_str) if title  == "" else ax.set_title (title)
    annotate_sbnd(ax, internal=internal)
    
    if bin_labels is not None:
        ax.set_xticks(bins)
        ax.set_xticklabels(bin_labels)
    else:
        ax.xaxis.set_minor_locator(_clipped_minor_locator(bins[0], bins[-1]))

    # Apply legend with custom kwargs
    default_legend_kwargs = {'ncol': 2, 'loc': 'upper right'}
    if legend_kwargs:
        default_legend_kwargs.update(legend_kwargs)
    legend = ax.legend(**default_legend_kwargs)
    legend.set_zorder(_TEXT_ZORDER)

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
                 data_first: bool = True,
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
    ylim_scale : float, default 1.5
        Multiply the auto upper y-limit of the main panel by this factor.
        Set to 1.0 to leave the y-axis unchanged.
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
    ratio_min  = _p.get('ratio_min', ratio_min)
    ratio_max  = _p.get('ratio_max', ratio_max)
    ylim_scale = _p.get('ylim_scale', 1.5)
    data_first = _p.get('data_first', data_first)
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 1, height_ratios=[6, 1], hspace=0.05)
    ax_main = fig.add_subplot(gs[0])
    ax_sub = fig.add_subplot(gs[1], sharex=ax_main)

    normalize = _p.get('normalize', False)
    overflow  = _p.get('overflow', True)

    # Data is always plotted as raw counts; when normalize=True, MC is scaled so
    # its area matches the data area rather than dividing data by its own integral.
    data_args = dict(df=data_df, var=var, bins=bins, ax=ax_main, normalize=False, overflow=overflow)
    data_hist, data_err, data_plot = data_plot_overlay(**data_args)

    mc_kwargs = dict(kwargs)
    if normalize:
        _mc_wt  = _get_weight_column(mc_df)
        _mc_raw = get_hist1d(
            data=mc_df[var],
            weights=mc_df[_mc_wt] if _mc_wt is not None else None,
            bins=np.asarray(bins), overflow=overflow,
        )
        _bw       = np.diff(np.asarray(bins))
        _mc_area  = float(np.sum(_mc_raw * _bw))
        _dat_area = float(np.sum(data_hist * _bw))
        if _mc_area > 0:
            mc_kwargs['scale'] = _dat_area / _mc_area
        mc_kwargs['normalize'] = False
    mc_args = dict(indf=mc_df, var=var, bins=bins, ax=ax_main, config=config, **mc_kwargs)

    mc_bins, mc_steps, mc_err, mc_dict = plot_var(**mc_args)
    
    xmin, xmax = ax_main.get_xlim()
    
    # plot the ratio
    mc_tot = mc_steps[-1][1:]  # last step contains the total MC counts

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",message="invalid value encountered in divide")
        # ratio is (data bin content) / (mc bin content)
        ratio = data_hist / mc_tot
        # error in ratio is just (data error) / (mc bin content)
        ratio_err = data_err / mc_tot
        # error in shading should just be (mc error) / (mc bin content)
        # Use 0 for zero-MC bins: NaN here causes fill_between(step="pre") to
        # offset the entire band one bin to the right for all subsequent bins.
        mc_contribution = np.where(mc_tot > 0, mc_err / mc_tot, 0.0)

    nbins = len(bins)-1
    mc_total_cov = mc_dict.get('__total_cov__') if isinstance(mc_dict, dict) else None

    bin_centers = 0.5 * (mc_bins[1:] + mc_bins[:-1])

    ax_sub.errorbar(bin_centers, ratio, yerr=ratio_err, fmt='s', markersize=3,color='black', zorder=1e3, label='Data/Pred ratio')
    _draw_step_band(ax_sub, mc_bins, mc_contribution,
                     color=mpl.colors.to_rgba("gray", alpha=0.4), lw=0.0, label='Pred err.')
    
    ax_sub.axhline(1, color='red', linestyle='--', linewidth=1, zorder=0,label="y=1.0")
    ax_sub.set_xlim(xmin, xmax)
    ax_sub.set_ylim(ratio_min, ratio_max)
    ax_sub.set_ylabel("Data/Pred")
    # Move xlabel to ratio panel and suppress top-panel x-axis tick labels.
    ax_sub.set_xlabel(ax_main.get_xlabel(), fontsize=12)
    ax_main.set_xlabel("")
    plt.setp(ax_main.get_xticklabels(), visible=False)
    ax_main.tick_params(axis='x', which='both', bottom=True, top=False)
    # ax_sub.legend(loc='upper center', bbox_to_anchor=(0.5, 1.4),
                #   ncol=3, fontsize='small', frameon=False)

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
    total_ratio_data_err = np.sqrt(total_data) / total_mc

    # Separate stat (data + MC stat) from syst for the annotation.
    # __mcstat_err__ always holds the per-bin MC stat regardless of how systematics were computed.
    _mc_st = mc_dict.get('__mcstat_err__', mc_err) if isinstance(mc_dict, dict) else mc_err
    mcstat_ratio_err     = np.sqrt(np.sum(_mc_st ** 2)) * (total_ratio / total_mc)
    total_ratio_stat_err = np.sqrt(total_ratio_data_err**2 + mcstat_ratio_err**2)
    syst_cov_sum         = max(0.0, np.sum(mc_cov) - np.sum(_mc_st ** 2))
    total_ratio_syst_err = np.sqrt(syst_cov_sum) * (total_ratio / total_mc)

    total_ratio_mc_err = np.sqrt(np.sum(mc_cov)) * (total_ratio / total_mc)
    total_ratio_err    = np.sqrt(total_ratio_data_err**2 + total_ratio_mc_err**2)

    valid = np.isfinite(data_hist) & np.isfinite(mc_tot)
    ndf  = nbins
    chi2 = np.nan
    if np.count_nonzero(valid) > 0:
        delta   = data_hist[valid] - mc_tot[valid]
        cov_sel = counts_cov[np.ix_(valid, valid)]
        try:
            chi2 = float(delta.T @ np.linalg.pinv(cov_sel) @ delta)
        except np.linalg.LinAlgError:
            chi2 = np.nan

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
        ann_lines = [rf"$\Sigma$ Data/Pred = {total_ratio:.2f} $\pm$ {total_ratio_stat_err:.2f} (stat.) $\pm$ {total_ratio_syst_err:.2f} (syst.)"]
        if np.isfinite(chi2):
            p_str = f"{chi2_dist.sf(chi2, ndf):.2g}" if chi2_dist is not None else "N/A"
            ann_lines.append(rf"$\chi^2$/ndf = {chi2:.1f}/{ndf}, $p$ = {p_str}")
        ax_main.annotate(
            "\n".join(ann_lines),
            xy=(ann_x, ann_y),
            xycoords=ax_main.transAxes,
            xytext=(0, -6),
            textcoords='offset points',
            ha=ann_ha, va='top', fontsize=ann_fontsize,
            zorder=_TEXT_ZORDER,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.5),
        )

    if bin_labels is not None:
        ax_main.set_xticks(bins)
        plt.setp(ax_main.get_xticklabels(), visible=False)
        ax_main.xaxis.set_minor_locator(mpl.ticker.NullLocator())
        ax_sub.set_xticks(bins)
        ax_sub.set_xticklabels(bin_labels)
    else:
        ax_sub.xaxis.set_minor_locator(_clipped_minor_locator(mc_bins[0], mc_bins[-1]))
    ax_sub.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())


    if ylim_scale != 1.0:
        ax_main.set_ylim(top=ax_main.get_ylim()[1] * ylim_scale)

    if data_first:
        handles, labels = ax_main.get_legend_handles_labels()
        idx = next((i for i, l in enumerate(labels) if l.startswith('data')), None)
        if idx is not None and idx != 0:
            order = [idx] + [i for i in range(len(labels)) if i != idx]
            _leg_kw = {'ncol': 2, 'loc': 'upper right'}
            _leg_kw.update(_p.get('legend_kwargs') or {})
            ax_main.legend([handles[i] for i in order], [labels[i] for i in order], **_leg_kw).set_zorder(_TEXT_ZORDER)

    annotate_sbnd(ax_main, internal=_p.get('internal', True))

    if savefig!="":
        plt.savefig(savefig,bbox_inches='tight')

    return fig, ax_main, ax_sub, mc_dict


def plot_mc_data_ccbc(
    mc_df: pd.DataFrame,
    data_df: pd.DataFrame,
    side_mc_df: pd.DataFrame,
    side_data_df: pd.DataFrame,
    var: str | tuple,
    bins: np.ndarray,
    side_var: str | tuple,
    ccbc_cov: dict | None = None,
    allowed_keys: tuple = ("GENIE", "Flux", "Geant4", "MCstat"),
    scale: float = 1.0,
    overflow: bool = True,
    figsize: tuple[int, int] = (7, 6),
    ratio_min: float = 0.0,
    ratio_max: float = 2.0,
    ylim_scale: float = 1.5,
    xlabel: str = "",
    ylabel: str = "Events",
    title: str = "",
    signal_label: str = "Signal",
    bkg_label: str = "Background (w/ CCBC)",
    signal_color: str = "C0",
    bkg_color: str = "C1",
    counts: bool = False,
    percents: bool = False,
    data_first: bool = True,
    cut_val: list | None = None,
    bin_labels: list | None = None,
    legend_kwargs: dict | None = None,
    annot: bool = True,
    internal: bool = True,
    savefig: str = "",
) -> tuple[plt.Figure, plt.Axes, plt.Axes, dict]:
    """MC+data plot with CCBC-constrained background (systs=True path).

    Stacks signal (``signal == 0``) and CCBC-constrained background
    (``signal != 0``) using universe columns already present in the DataFrames,
    draws a ``cov_ms_ms`` syst band, data overlay, ratio panel, and chi-sq
    annotation — mirroring :func:`plot_mc_data`.

    Parameters
    ----------
    mc_df : pd.DataFrame
        Signal-region MC DataFrame (must have a ``signal`` column and universe
        weight columns so ``systs=True`` works).
    data_df : pd.DataFrame
        Signal-region data DataFrame.
    side_mc_df : pd.DataFrame
        Sideband MC DataFrame (universe columns required).
    side_data_df : pd.DataFrame
        Sideband data DataFrame.
    var : str or tuple
        Variable to plot in the signal region (and used as the sideband
        variable after ``side_var`` is applied).
    bins : np.ndarray
        Shared bin edges for both the signal region and sideband.
    side_var : str or tuple
        Variable to histogram in the sideband DataFrames.
    ccbc_cov : dict, optional
        Pre-computed output of :func:`~nueana.ccbc.ccbc_cov_from_universes`.
        When supplied, the sideband universe histograms are not recomputed —
        pass this when looping over many signal-region variables so the sideband
        work is done once.
    allowed_keys : tuple of str
        Systematic categories forwarded to :func:`~nueana.ccbc.ccbc_cov_from_universes`.
    scale : float, default 1.0
        Multiplicative scale applied to all histograms and covariances
        (use ``projected_pot / mcbnb_pot`` to move onto a target POT).
    overflow : bool, default True
        Fold out-of-range values into edge bins.
    signal_label, bkg_label : str
        Legend labels for the two stack categories.
    signal_color, bkg_color : str
        Fill colours for the two stack categories.
    counts : bool, default False
        If True, append scaled event counts to each category's legend label.
    percents : bool, default False
        If True, append each category's percentage of the total MC to its
        legend label.
    cut_val : list of float, optional
        x-values for vertical dashed cut lines.
    annot : bool, default True
        Annotate with integrated Data/Pred ratio and chi-sq / p-value.
    savefig : str, optional
        Path to save the figure; skipped when empty.

    Returns
    -------
    fig, ax_main, ax_sub, out_dict
        ``out_dict`` contains ``ccbc_cov``, ``constrained_bkg``,
        ``cov_ms_ms``, and ``total_cov`` for downstream reuse.
    """
    from .syst import get_syst
    from .ccbc import ccbc_cov_from_universes, get_constrained_background
    from .utils import get_hist1d, ensure_lexsorted

    mc_df      = ensure_lexsorted(mc_df,      axis=1)
    side_mc_df = ensure_lexsorted(side_mc_df, axis=1)

    sig_mask = mc_df.signal == 0
    bkg_mask = ~sig_mask

    _weight_col = _get_weight_column(mc_df)
    _side_weight_col = _get_weight_column(side_mc_df)

    cv_ps = get_hist1d(
        data=mc_df[sig_mask][var],
        weights=mc_df[sig_mask][_weight_col] if _weight_col else None,
        bins=bins, overflow=overflow,
    )
    cv_bs = get_hist1d(
        data=mc_df[bkg_mask][var],
        weights=mc_df[bkg_mask][_weight_col] if _weight_col else None,
        bins=bins, overflow=overflow,
    )

    # Build CCBC covariance — reuse cached sideband if supplied.
    if ccbc_cov is None:
        from .syst import get_syst_hists
        syst_ps_h, _ = get_syst_hists(mc_df[sig_mask],  var,      bins)
        syst_bs_h, _ = get_syst_hists(mc_df[bkg_mask],  var,      bins)
        syst_nc_h, _ = get_syst_hists(side_mc_df,       side_var, bins)
        cv_nc = get_hist1d(
            data=side_mc_df[side_var],
            weights=side_mc_df[_side_weight_col] if _side_weight_col else None,
            bins=bins, overflow=overflow,
        )
        # data_stat_nc must be in mcbnb_pot units (same as cov_nc_nc).
        # Raw data counts are at data_pot; dividing by scale converts to mcbnb_pot.
        # Poisson variance of (data_nc / scale) is also data_nc / scale.
        _data_nc_raw = get_hist1d(data=side_data_df[side_var], bins=bins, overflow=overflow)
        data_stat_nc = np.diag(_data_nc_raw / scale)
        ccbc_cov = ccbc_cov_from_universes(
            syst_ps_h, cv_ps,
            syst_bs_h, cv_bs,
            syst_nc_h, cv_nc,
            allowed_keys=allowed_keys,
            data_stat_nc=data_stat_nc,
        )

    # Pass data in mcbnb_pot units so (data_nc - cv_nc) is a meaningful residual.
    # constrained_bkg is returned in mcbnb_pot units; bkg_hist = constrained_bkg * scale
    # below brings it onto the target POT, consistent with sig_hist = cv_ps * scale.
    data_nc = get_hist1d(data=side_data_df[side_var], bins=bins, overflow=overflow)
    constrained_bkg = get_constrained_background(ccbc_cov, data_nc / scale)

    # MC stat variance (sum of w^2 per bin) for the combined stack.
    mc_stat_var = get_hist1d(
        data=mc_df[var],
        weights=np.square(mc_df[_weight_col]) if _weight_col else None,
        bins=bins, overflow=overflow,
    )

    # Full covariance: cov_ms_ms (syst, already contains signal+bkg blocks)
    # plus diagonal MC stat.
    cov_ms_ms = np.asarray(ccbc_cov["cov_ms_ms"]) * scale ** 2
    total_cov = cov_ms_ms + np.diag(mc_stat_var) * scale ** 2
    total_err = np.sqrt(np.clip(np.diag(total_cov), 0.0, None))

    # --- Draw ---
    fig = plt.figure(figsize=figsize)
    gs     = GridSpec(2, 1, height_ratios=[6, 1], hspace=0.05)
    ax_main = fig.add_subplot(gs[0])
    ax_sub  = fig.add_subplot(gs[1], sharex=ax_main)

    alpha  = 0.4
    nbins  = len(bins) - 1

    sig_hist = cv_ps * scale
    bkg_hist = constrained_bkg * scale

    def _fmt_count(n):
        return f" ({int(n):,})" if n < 1e6 else f" ({n:.2e})"

    if counts:
        signal_label = signal_label + _fmt_count(np.sum(sig_hist))
        bkg_label    = bkg_label    + _fmt_count(np.sum(bkg_hist))
    if percents:
        _total = np.sum(sig_hist) + np.sum(bkg_hist)
        if _total > 0:
            signal_label = signal_label + f" ({np.sum(sig_hist) / _total * 100:.1f}%)"
            bkg_label    = bkg_label    + f" ({np.sum(bkg_hist) / _total * 100:.1f}%)"

    # Stacked fill: background first (bottom), signal on top.
    bkg_step = np.insert(bkg_hist, 0, bkg_hist[0])
    tot_step = np.insert(sig_hist + bkg_hist, 0, (sig_hist + bkg_hist)[0])

    ax_main.fill_between(bins, 0,        bkg_step, step="pre",
                         facecolor=mpl.colors.to_rgba(bkg_color, alpha),
                         edgecolor=mpl.colors.to_rgba(bkg_color, 1.0),
                         lw=1.5, label=bkg_label)
    ax_main.fill_between(bins, bkg_step, tot_step, step="pre",
                         facecolor=mpl.colors.to_rgba(signal_color, alpha),
                         edgecolor=mpl.colors.to_rgba(signal_color, 1.0),
                         lw=1.5, label=signal_label)

    _draw_step_band(ax_main, bins, total_err, center=tot_step,
                    color=mpl.colors.to_rgba("gray", 0.75),
                    lw=0.0, facecolor="none", hatch="xxx",
                    label="MC stat.+syst.\n(GENIE+Flux+G4, w/ CCBC)")

    # Data overlay.
    data_hist   = get_hist1d(data=data_df[var], bins=bins, overflow=overflow)
    data_err    = np.sqrt(data_hist)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    data_count_label = (f" ({int(np.sum(data_hist)):,})" if np.sum(data_hist) < 1e6
                        else f" ({np.sum(data_hist):.2e})")
    ax_main.errorbar(bin_centers, data_hist, yerr=data_err,
                     fmt='.', color='black', zorder=1e3,
                     label="data" + data_count_label)

    # Ratio panel.
    mc_tot = (sig_hist + bkg_hist)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in divide")
        ratio      = np.where(mc_tot > 0, data_hist / mc_tot, np.nan)
        ratio_err  = np.where(mc_tot > 0, data_err  / mc_tot, np.nan)
        mc_contrib = np.where(mc_tot > 0, total_err / mc_tot, 0.0)

    ax_sub.errorbar(bin_centers, ratio, yerr=ratio_err,
                    fmt='s', markersize=3, color='black', zorder=1e3)
    _draw_step_band(ax_sub, bins, mc_contrib,
                    color=mpl.colors.to_rgba("gray", 0.4), lw=0.0)
    ax_sub.axhline(1, color='red', linestyle='--', linewidth=1, zorder=0)
    ax_sub.set_ylim(ratio_min, ratio_max)
    ax_sub.set_ylabel("Data/Pred")

    xmin, xmax = ax_main.get_xlim()
    ax_sub.set_xlim(xmin, xmax)

    if cut_val is not None:
        for cut in cut_val:
            ax_main.axvline(cut, lw=2, color="gray", linestyle="--", zorder=nbins + 2)
            ax_sub.axvline (cut, lw=2, color="black", linestyle="--", alpha=0.5, zorder=1e2)

    # Compute annotation values before drawing legend (no canvas ops yet).
    if annot:
        total_data  = np.sum(data_hist)
        total_mc    = np.sum(mc_tot)
        total_ratio = total_data / total_mc if total_mc > 0 else np.nan
        data_cov    = np.diag(np.square(data_err))
        counts_cov  = data_cov + total_cov
        valid = np.isfinite(data_hist) & np.isfinite(mc_tot)
        chi2  = np.nan
        if np.count_nonzero(valid) > 0:
            delta   = data_hist[valid] - mc_tot[valid]
            cov_sel = counts_cov[np.ix_(valid, valid)]
            try:
                chi2 = float(delta @ np.linalg.pinv(cov_sel) @ delta)
            except np.linalg.LinAlgError:
                pass
        ann_lines = []
        if np.isfinite(total_ratio) and total_mc > 0:
            _prop = total_ratio / total_mc   # d(ratio)/d(MC) propagation factor
            _data_stat_err  = np.sqrt(total_data) / total_mc
            _mc_stat_err    = np.sqrt(np.sum(mc_stat_var)) * scale * _prop
            _ratio_stat_err = np.sqrt(_data_stat_err ** 2 + _mc_stat_err ** 2)
            _ratio_syst_err = np.sqrt(max(0.0, float(np.sum(cov_ms_ms)))) * _prop
            ann_lines.append(
                rf"$\Sigma$ Data/Pred = {total_ratio:.4f}"
                rf" $\pm$ {_ratio_stat_err:.2f} (stat.)"
                rf" $\pm$ {_ratio_syst_err:.2f} (syst.)"
            )
        if np.isfinite(chi2):
            p_str = (f"{chi2_dist.sf(chi2, nbins):.2g}" if chi2_dist is not None else "N/A")
            ann_lines.append(rf"$\chi^2$/ndf = {chi2:.1f}/{nbins}, $p$ = {p_str}")

    _var_str = var if isinstance(var, str) else '_'.join(str(v) for v in var)
    ax_sub.set_xlabel(_var_str if xlabel == "" else xlabel, fontsize=12)
    ax_main.set_xlabel("")
    ax_main.set_ylabel("Events" if ylabel == "" else ylabel, fontsize=12)
    ax_main.set_title(_var_str if title == "" else title)
    plt.setp(ax_main.get_xticklabels(), visible=False)
    ax_main.tick_params(axis='x', which='both', bottom=True, top=False)

    if bin_labels is not None:
        ax_main.set_xticks(bins)
        plt.setp(ax_main.get_xticklabels(), visible=False)
        ax_main.xaxis.set_minor_locator(mpl.ticker.NullLocator())
        ax_sub.set_xticks(bins)
        ax_sub.set_xticklabels(bin_labels)
    else:
        ax_sub.xaxis.set_minor_locator(_clipped_minor_locator(bins[0], bins[-1]))
    ax_sub.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())

    if ylim_scale != 1.0:
        ax_main.set_ylim(bottom=0, top=ax_main.get_ylim()[1] * ylim_scale)

    _leg_kw = {'ncol': 2, 'loc': 'upper right'}
    if legend_kwargs:
        _leg_kw.update(legend_kwargs)
    ax_main.legend(**_leg_kw).set_zorder(_TEXT_ZORDER)
    if data_first:
        handles, labels = ax_main.get_legend_handles_labels()
        idx = next((i for i, l in enumerate(labels) if l.startswith('data')), None)
        if idx is not None and idx != 0:
            order = [idx] + [i for i in range(len(labels)) if i != idx]
            ax_main.legend(
                [handles[i] for i in order], [labels[i] for i in order], **_leg_kw
            ).set_zorder(_TEXT_ZORDER)
    annotate_sbnd(ax_main, internal=internal)

    # Position annotation below the legend (mirrors plot_mc_data).
    if annot and ann_lines:
        fig.canvas.draw()
        legend_loc  = str((_leg_kw).get('loc', '')).lower()
        main_legend = ax_main.get_legend()
        if main_legend is not None:
            renderer     = fig.canvas.get_renderer()
            legend_box   = main_legend.get_window_extent(renderer).transformed(ax_main.transAxes.inverted())
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
        ax_main.annotate(
            "\n".join(ann_lines),
            xy=(ann_x, ann_y),
            xycoords=ax_main.transAxes,
            xytext=(0, -6), textcoords='offset points',
            ha=ann_ha, va='top', fontsize=ann_fontsize,
            zorder=_TEXT_ZORDER,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor='none', alpha=0.5),
        )

    if savefig:
        plt.savefig(savefig, bbox_inches='tight')

    out_dict = {
        "ccbc_cov":        ccbc_cov,
        "constrained_bkg": constrained_bkg,
        "cov_ms_ms":       cov_ms_ms,
        "total_cov":       total_cov,
    }
    return fig, ax_main, ax_sub, out_dict


def plot_detvar(
    detvar_dict: dict,
    key: str,
    var: str | tuple,
    bins: np.ndarray,
    figsize: tuple[int, int] = (5, 5),
    xlabel: str = "",
    ylabel: str = "Events",
    ratio_min: float = 0.5,
    ratio_max: float = 1.5,
    internal: bool = True,
    bin_labels: list[str] | None = None,
) -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    """Compare DV and CV histograms for one detector variation entry.

    Parameters
    ----------
    detvar_dict : dict
        Detector variation dictionary as returned by
        :func:`~nueana.detvar.store.load_detvar_dict`.
    key : str
        Group name to plot (a key in ``detvar_dict``).
    var : str or tuple
        Column to histogram.
    bins : np.ndarray
        Bin edges.
    figsize : tuple, default (5, 5)
    xlabel : str, optional
        x-axis label placed on the ratio panel.
    ylabel : str, default "Events"
    ratio_min, ratio_max : float, default (0.5, 1.5)
        y-axis limits for the DV/CV ratio subplot.
    bin_labels : list of str, optional
        Custom tick labels placed at each bin edge on the ratio panel.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax_main : matplotlib.axes.Axes
        Upper panel with CV and DV histograms.
    ax_ratio : matplotlib.axes.Axes
        Lower panel with DV/CV ratio.
    """
    entry  = detvar_dict[key]
    cv_df  = ensure_lexsorted(entry['cv_df'], axis=1)

    cv_hist = get_hist1d(data=cv_df[var], bins=bins)

    dv_entry = entry['dv_df']
    dv_dfs   = dv_entry if isinstance(dv_entry, list) else [dv_entry]
    dv_hists = [
        get_hist1d(data=ensure_lexsorted(dv, axis=1)[var], bins=bins)
        for dv in dv_dfs
    ]

    fig = plt.figure(figsize=figsize)
    gs       = GridSpec(2, 1, height_ratios=[4, 1], hspace=0.15)
    ax_main  = fig.add_subplot(gs[0])
    ax_ratio = fig.add_subplot(gs[1])

    ax_main.stairs(cv_hist, bins, color='black', lw=1.5, label='CV')
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, dv_hist in enumerate(dv_hists):
        label = f'DV {i}' if len(dv_hists) > 1 else 'DV'
        color = colors[i % len(colors)]
        ax_main.stairs(dv_hist, bins, color=color, lw=1.5, linestyle='--', label=label)
        with np.errstate(invalid='ignore', divide='ignore'):
            ratio = np.where(cv_hist > 0, dv_hist / cv_hist, np.nan)
        ax_ratio.stairs(ratio, bins, color=color, lw=1.5, linestyle='--')

    ax_ratio.axhline(1.0, color='black', lw=1)
    ax_ratio.set_ylim(ratio_min, ratio_max)
    ax_ratio.set_ylabel("DV / CV")
    if xlabel:
        ax_ratio.set_xlabel(xlabel,fontsize=12)

    xmin, xmax = ax_main.get_xlim()
    ax_ratio.set_xlim(xmin, xmax)

    ax_main.set_ylabel(ylabel)
    ax_main.set_title(key)
    ax_main.legend()
    annotate_sbnd(ax_main, internal=internal)

    if bin_labels is not None:
        ax_main.set_xticks(bins)
        ax_main.set_xticklabels(bin_labels)
        ax_main.xaxis.set_minor_locator(mpl.ticker.NullLocator())
        ax_ratio.set_xticks(bins)
        ax_ratio.set_xticklabels(bin_labels)
    else:
        ax_main.xaxis.set_minor_locator(_clipped_minor_locator(bins[0], bins[-1]))
        ax_ratio.xaxis.set_minor_locator(_clipped_minor_locator(bins[0], bins[-1]))

    return fig, ax_main, ax_ratio


def _combine_syst_uncertainties(syst_df: pd.DataFrame) -> np.ndarray:
    """Combine per-row uncertainty arrays into a single per-bin band."""
    if hasattr(syst_df, 'empty') and syst_df.empty:
        return np.array([])

    if isinstance(syst_df, pd.Series):
        unc_values = np.stack(syst_df.to_numpy())
    else:
        unc_values = np.stack(syst_df['unc_diag'].to_numpy())
    return np.sqrt(np.sum(np.square(unc_values), axis=0))


def plot_syst_category_breakdown(
    syst_vars: list[tuple],
    category_dict: dict,
    region_label: str = "Signal Region",
    figsize: tuple[int, int] | None = None,
    xsec: bool = False,
    show_cv: bool = False,
    projected_pot: float = 1e20,
    group_by: str = 'category',
) -> tuple[plt.Figure, np.ndarray, list, list]:
    """Plot the category-level systematics summary for any number of variables.

    Parameters
    ----------
    syst_vars : list of tuple
        One entry per variable, each a 3- or 4-tuple:
        ``(SystematicsOutput, bins, xlabel)`` or
        ``(SystematicsOutput, bins, xlabel, bin_labels)``.
    category_dict : dict
        Mapping of category name → style dict (``color``, ``label``, ``line``).
        When ``group_by='subcategory'``, keys are subcategory names (e.g.
        ``'PMT'``, ``'WireMod'``, ``'SCE'``, ``'calorimetry'``).
    region_label : str, default "Signal Region"
        Text stamped in the corner of each subplot.
    figsize : tuple, optional
        Figure size. Defaults to ``(5 * n_vars, 4)``.
    xsec : bool, default False
        If True, plot uncertainties on the cross section (``xsec_syst_df``)
        instead of the event rate (``rate_syst_df``).
    show_cv : bool, default True
        If True, overlay the predicted event-rate histogram on a twin y-axis
        (right) as a semi-transparent filled band.
    projected_pot : float, default 1e20
        POT used to scale the CV histogram to predicted event counts.
    group_by : str, default 'category'
        Column to group by: ``'category'`` (GENIE, Flux, MCstat, DetVar, …) or
        ``'subcategory'`` (PMT, WireMod, SCE, calorimetry, other for DetVar rows).

    Returns
    -------
    fig, axes, cats_per_var, cat_sums_per_var
        ``cats_per_var`` and ``cat_sums_per_var`` are lists (one per variable)
        of grouped uncertainty arrays and normalisation sums.
    """
    n = len(syst_vars)
    if figsize is None:
        figsize = (5 * n, 4)

    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = np.array([axes])
    plt.subplots_adjust(wspace=0.3)

    cats_per_var = []
    cat_sums_per_var = []

    for ax, item in zip(axes, syst_vars):
        syst_output, bins, xlabel = item[0], item[1], item[2]
        bin_labels = item[3] if len(item) > 3 else None
        if xsec:
            if not syst_output.has_xsec:
                raise ValueError("SystematicsOutput does not contain xsec results; recompute with xsec_inputs set.")
            syst_df = syst_output.xsec_syst_df
            cv_hist = np.asarray(syst_output.xsec_hist_cv)
        else:
            syst_df = syst_output.rate_syst_df
            cv_hist = np.asarray(syst_output.rate_hist_cv)

        _display_pot = getattr(syst_output, 'data_pot', None) or projected_pot
        pot_scale = _display_pot / syst_output.mcbnb_pot

        if show_cv:
            plt.subplots_adjust(wspace=0.5)
            cv_counts = cv_hist * pot_scale
            ax_cv = ax.twinx()
            ax_cv.stairs(cv_counts, bins, fill=True, alpha=0.25, color='steelblue', lw=0)
            ax_cv.set_ylim(bottom=0, top=np.max(cv_counts) * 1.25)
            pot_label = f"{_display_pot:.3g}"
            ax_cv.set_ylabel(f"Predicted Events ({pot_label} POT)", color='steelblue', alpha=0.7, fontsize=10)
            ax_cv.tick_params(axis='y', labelcolor='steelblue')
            ax_cv.set_zorder(ax.get_zorder() - 1)
            ax.set_facecolor('none')

        cat    = syst_df.sort_values('unc_norm').groupby(group_by)['unc_diag'].apply(_combine_syst_uncertainties)
        sums   = syst_df.groupby(group_by)['unc_norm'].apply(lambda s: float(np.sqrt(np.sum(s**2))))
        cats_per_var.append(cat)
        cat_sums_per_var.append(sums)

        for category in category_dict.keys():
            if category not in cat.index:
                continue
            style = category_dict[category]
            _label = (f"Data statistics\n[{_display_pot:.3g} POT]"
                      if category == 'Datastat' else style['label'])
            ax.stairs(
                cat[category] * 100,
                bins,
                lw=1.8,
                linestyle=style['line'],
                label=f"{_label} ({sums.get(category, 0.):.1%})",
                color=style['color'],
                alpha=0.8,
            )

        tot = _combine_syst_uncertainties(syst_df)
        total_sum = float(np.sqrt(np.sum(syst_df['unc_norm'] ** 2)))
        if tot.size:
            ax.stairs(tot * 100, bins, lw=2, color='black', label=f'Total ({total_sum:.1%})')

        ax.set_xlabel(xlabel,fontsize=12)
        _ylabel = "Uncertainty on the Cross Section [%]" if xsec else "Uncertainty on the Event Rate [%]"
        ax.set_ylabel(_ylabel)
        ax.set_ylim(0, 35)
        ax.set_xticks(bins)
        if bin_labels is not None:
            ax.set_xticklabels(bin_labels)
        ax.annotate(text=region_label, xy=(0.02, 0.925), xycoords='axes fraction',
                    fontsize=11, fontweight='bold', alpha=0.5)

    axes[-1].legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                    title="Uncertainty Sources (Normalization %)")

    return fig, axes, cats_per_var, cat_sums_per_var


def plot_syst_breakdown(
    syst_vars: list[tuple],
    category: str,
    category_dict: dict,
    region_label: str | None = None,
    figsize: tuple[int, int] | None = None,
    xsec: bool = False,
    subcategory: str | None = None,
    show_subcategories: bool = False,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot the per-source systematics breakdown for one category.

    Parameters
    ----------
    syst_vars : list of tuple
        One entry per variable, each a 3- or 4-tuple:
        ``(SystematicsOutput, bins, xlabel)`` or
        ``(SystematicsOutput, bins, xlabel, bin_labels)``.
    category : str
        Category key from ``category_dict`` to plot (e.g. ``'GENIE'``,
        ``'DetVar'``). Ignored for filtering when ``subcategory`` is set, but
        still used to look up style in ``category_dict`` unless ``subcategory``
        is also a key there.
    category_dict : dict
        Mapping of category (or subcategory) name → style dict
        (``color``, ``label``, ``line``).
    region_label : str, optional
        Text stamped in the corner of each subplot.
    figsize : tuple, optional
        Figure size. Defaults to ``(5 * n_vars, 4)``.
    xsec : bool, default False
        If True, plot uncertainties on the cross section (``xsec_syst_df``)
        instead of the event rate (``rate_syst_df``).
    subcategory : str, optional
        When set, filter rows by ``syst_df.subcategory == subcategory`` instead
        of ``syst_df.category == category``. Use this to drill into DetVar
        subcategories (``'PMT'``, ``'WireMod'``, ``'SCE'``, ``'calorimetry'``).
        The style is looked up as ``category_dict[subcategory]`` if that key
        exists, otherwise falls back to ``category_dict[category]``.
    show_subcategories : bool, default False
        If True, overlay one combined line per subcategory on top of the
        individual contributions. Each subcategory line is the quadrature sum
        of all rows in that subcategory, styled via ``category_dict``. Useful
        for DetVar to see PMT, WireMod, SCE, etc. at a glance alongside the
        individual sources.

    Returns
    -------
    fig, axes
    """
    n = len(syst_vars)
    if figsize is None:
        figsize = (5 * n, 4)

    _style_key = subcategory if (subcategory is not None and subcategory in category_dict) else category
    this_color = category_dict[_style_key]['color']
    this_label = category_dict[_style_key]['label']

    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = np.array([axes])
    plt.subplots_adjust(wspace=0.3)

    for ax, item in zip(axes, syst_vars):
        syst_output, bins, xlabel = item[0], item[1], item[2]
        bin_labels = item[3] if len(item) > 3 else None
        if xsec:
            if not syst_output.has_xsec:
                raise ValueError("SystematicsOutput does not contain xsec results; recompute with xsec_inputs set.")
            syst_df = syst_output.xsec_syst_df
        else:
            syst_df = syst_output.rate_syst_df

        if subcategory is not None:
            this_df = syst_df[syst_df.subcategory == subcategory].sort_values('unc_norm', ascending=False)
        else:
            this_df = syst_df[syst_df.category == category].sort_values('unc_norm', ascending=False)

        if show_subcategories:
            subcat_order = (
                this_df.groupby('subcategory')['unc_norm']
                .apply(lambda s: float(np.sqrt(np.sum(s**2))))
                .sort_values(ascending=False)
                .index
            )
            for subcat in subcat_order:
                group = this_df[this_df['subcategory'] == subcat]
                style    = category_dict.get(subcat, category_dict.get(category, {}))
                unc_sum  = float(np.sqrt(np.sum(group['unc_norm'] ** 2)))
                combined = _combine_syst_uncertainties(group)
                if combined.size:
                    ax.stairs(
                        combined * 100,
                        bins,
                        lw=2.0,
                        linestyle=style.get('line', '--'),
                        color=style.get('color', None),
                        label=f"{style.get('label', subcat)} ({unc_sum:.1%})",
                    )
        else:
            for _, row in this_df.iterrows():
                ax.stairs(
                    row.unc_diag * 100,
                    bins,
                    lw=1.5,
                    label=row.key + f" ({row['unc_norm']:.1%})" if row.top5 else "",
                    alpha=0.5,
                )

        tot = _combine_syst_uncertainties(this_df)
        tot_sum = float(np.sqrt(np.sum(this_df['unc_norm'] ** 2)))
        if tot.size:
            ax.stairs(tot * 100, bins, lw=2, color=this_color,
                      label=f'Total {this_label} ({tot_sum:.1%})')

        ax.set_xlabel(xlabel)
        _ylabel = "Uncertainty on the Cross Section [%]" if xsec else "Uncertainty on the Event Rate [%]"
        ax.set_ylabel(_ylabel)
        ax.set_ylim(0, 35)
        ax.set_xticks(bins)
        if bin_labels is not None:
            ax.set_xticklabels(bin_labels)
        ax.legend(title='top 5 sources', fontsize=9)
        if region_label is not None:
            ax.annotate(text=region_label, xy=(0.02, 0.925), xycoords='axes fraction',
                        fontsize=11, fontweight='bold', alpha=0.5)

    return fig, axes