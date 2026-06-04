from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional

import numpy as np
import pandas as pd


@dataclass
class CutSpec:
    """Declarative description of a single selection cut.

    Exactly one of ``variable``, ``accessor``, or ``fn`` must be set.

    Parameters
    ----------
    name : str
        Unique identifier used for stage stopping and savedict keys.
    variable : tuple, optional
        MultiIndex key resolved via getattr chaining, e.g.
        ``("primshw", "shw", "len")``. Cut passes when
        ``min < df.<variable> < max``.
    min, max : float
        Lower and upper bounds for variable/accessor cuts.
        Default to ``-inf`` / ``+inf`` (i.e. open-ended).
    accessor : callable, optional
        ``lambda df: <Series>`` — use when column access is more
        complex than a simple tuple key. Cut passes when
        ``min < accessor(df) < max``.
    fn : callable, optional
        ``fn(df) -> bool mask`` — full override for cuts that are not
        a simple min/max comparison. Takes precedence over
        ``variable`` and ``accessor``.
    label : str, optional
        Human-readable description, e.g. for cut-flow tables and plots.
        Defaults to ``name`` if not set.
    """
    name: str
    variable: tuple = None
    min: float = -np.inf
    max: float = np.inf
    accessor: Callable = None
    fn: Callable = None
    label: str = None

    def __post_init__(self):
        if self.fn is None and self.accessor is None and self.variable is None:
            raise ValueError(
                f"CutSpec '{self.name}': at least one of variable, accessor, or fn must be set."
            )
        if self.label is None:
            self.label = self.name


class VariableConfig:
    """Configurable container for an unfolding variable.

    Instantiate directly with all parameters, or use the pre-built factory
    functions in :mod:`nueana.variables` (e.g. ``variables.electron_energy()``).
    """
    def __init__(self, var_save_name, var_plot_name, var_unit, bins, bin_labels, var_evt_reco_col, var_evt_truth_col, var_nu_col):
        self.var_save_name = var_save_name
        self.var_plot_name = var_plot_name
        self.var_unit = var_unit
        # var_plot_name is the math content; we accept it with or without outer
        # $...$ (stripping if present so nested-dollar parsing doesn't break).
        # var_unit is bare (e.g. "GeV", not "(GeV)"); brackets are added here.
        # Variable, superscript, AND unit all live inside one \mathrm{...} so
        # everything renders upright (units in italic math is wrong typography).
        plot_math   = var_plot_name.strip("$")
        unit_suffix = f"~[{var_unit}]" if len(var_unit) > 0 else ""
        self.var_labels = [r"$\mathrm{" + plot_math + unit_suffix + "}$",
                           r"$\mathrm{" + plot_math + "^{reco.}" + unit_suffix + "}$",
                           r"$\mathrm{" + plot_math + "^{true}"  + unit_suffix + "}$"]
        self.bins = bins
        self.bin_centers = (bins[:-1] + bins[1:]) / 2.
        self.bin_labels = bin_labels
        self.bin_diff_labels = [f"{bin_labels[i]}-{bin_labels[i+1]}" for i in range(len(bin_labels)-1)]
        self.var_evt_reco_col = var_evt_reco_col
        self.var_evt_truth_col = var_evt_truth_col
        self.var_nu_col = var_nu_col

@dataclass(frozen=True)
class XSecInputs:
    """
    Run-level inputs for cross-section unfolding.
    Column references live on VariableConfig; only truth-signal
    information that is independent of the choice of variable belongs here.
    """

    true_signal_df: pd.DataFrame
    true_signal_scale: float
    reco_var_true: str | tuple
    true_var_true: str | tuple

    def __post_init__(self):
        if self.true_signal_df is not None:
            from .utils import ensure_lexsorted
            df = ensure_lexsorted(self.true_signal_df, axis=1)
            object.__setattr__(self, 'true_signal_df', df[df.signal == 0])


@dataclass(frozen=True)
class SystematicsOutput:
    """
    Results of a systematics evaluation for a single variable.
    xsec_* fields are optional; check .has_xsec before accessing them.
    """

    rate_hist_cv: np.ndarray
    rate_cov: np.ndarray
    rate_syst_df: pd.DataFrame
    rate_syst_dict: dict
    mcbnb_pot: float | None = None
    xsec_hist_cv: np.ndarray | None = None
    xsec_cov: np.ndarray | None = None
    xsec_syst_df: pd.DataFrame | None = None
    xsec_syst_dict: dict | None = None

    @property
    def has_xsec(self) -> bool:
        """True if cross-section covariance was computed."""
        return self.xsec_cov is not None


@dataclass(frozen=True)
class SystematicsInput:
    """Arguments forwarded to :func:`~nueana.funcs.get_total_cov` at plot time.

    Pass an instance as ``systs`` to :func:`~nueana.plotting.plot_var` or
    :func:`~nueana.plotting.plot_mc_data` to compute the full covariance
    matrix on-the-fly inside the plotting call.

    All fields map 1-to-1 to the corresponding :func:`~nueana.funcs.get_total_cov`
    parameters; ``reco_df``, ``reco_var``, and ``bins`` are supplied automatically
    from the plotting function's own arguments.

    Parameters
    ----------
    mcbnb_pot : float
        MC BNB POT for normalisation.
    cuts : list of CutSpec, optional
        Custom cut sequence for detector-variation selection.
    projected_pot : float, default 1e20
        POT used for the data-statistics estimate.
    mcbnb_ngen : float, optional
        Number of generated events; required for in-time cosmic uncertainty.
    intime_threshold : float, default 0.05
        Floor threshold for the in-time cosmic uncertainty.
    event_type : str or None, default 'all'
        Event mask forwarded to :func:`~nueana.utils.apply_event_mask`.
    select_region : str, default 'signal'
        Detector variation region: ``'signal'``, ``'control'``, or ``'all'``.
    uncertainty_keys : list of str, optional
        Subset of uncertainty blocks to compute.  Defaults to all available.
    xsec_inputs : XSecInputs, optional
        Cross-section inputs; required when ``'xsec'`` is in ``uncertainty_keys``.
    detvar_dict : dict, optional
        Pre-loaded detector variation dictionary (avoids repeated disk reads).
    """
    mcbnb_pot: float
    cuts: object = None
    projected_pot: float = 1e20
    mcbnb_ngen: float | None = None
    intime_threshold: float = 0.05
    event_type: str | None = "all"
    select_region: str = "signal"
    uncertainty_keys: object = None
    xsec_inputs: object = None
    detvar_dict: object = None

    def to_kwargs(self) -> dict:
        """Return fields as a dict suitable for unpacking into get_total_cov.

        Fields whose value is None are omitted so that explicit keyword
        arguments at the call site take precedence over the dataclass defaults.
        """
        from dataclasses import fields
        return {f.name: getattr(self, f.name) for f in fields(self)
                if getattr(self, f.name) is not None}


@dataclass
class PlottingConfig:
    """Style and display options for plot_var and plot_mc_data.

    Pass an instance as the ``config`` argument to avoid spelling out all
    parameters inline.  Any keyword argument passed directly to the plotting
    function overrides the corresponding field here.
    """
    xlabel: str = ""
    ylabel: str = ""
    title: str = ""
    counts: bool = False
    percents: bool = False
    scale: float = 1.0
    normalize: bool = False
    mult_factor: float = 1.0
    cut_val: list[float] | None = None
    plot_err: bool = True
    systs: bool | SystematicsInput | SystematicsOutput | None = None
    pdg: bool = False
    pdg_col: tuple | str = 'pfp_shw_truth_p_pdg'
    mode: bool = False
    hatch: list[str] | None = None
    bin_labels: list[str] | None = None
    overflow: bool = True
    legend_kwargs: dict | None = None
    ratio_min: float = 0.0
    ratio_max: float = 2.0
    data_first: bool = True
    internal: bool = True
    categories: dict | None = None


__all__ = [
    'CutSpec',
    'VariableConfig',
    'XSecInputs',
    'SystematicsOutput',
    'SystematicsInput',
    'PlottingConfig',
]