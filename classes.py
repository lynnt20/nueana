from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

class VariableConfig:
    """Configurable container for an unfolding variable.

    Instantiate directly with all parameters, or use the pre-built factory
    functions in :mod:`nueana.variables` (e.g. ``variables.electron_energy()``).
    """
    def __init__(self, var_save_name, var_plot_name, var_unit, bins, bin_labels, var_evt_reco_col, var_evt_truth_col, var_nu_col):
        self.var_save_name = var_save_name
        self.var_plot_name = var_plot_name
        self.var_unit = var_unit
        unit_suffix = f"~[{var_unit}]" if len(var_unit) > 0 else ""
        self.var_labels = [r"$\mathrm{" + var_plot_name + unit_suffix + "}$",
                           r"$\mathrm{" + var_plot_name + "^{reco.}" + unit_suffix + "}$",
                           r"$\mathrm{" + var_plot_name + "^{true}" + unit_suffix + "}$"]
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


@dataclass(frozen=True)
class SystematicsOutput:
    """
    Results of a systematics evaluation for a single variable.
    xsec_* fields are optional; check .has_xsec before accessing them.
    """

    hist_cv: np.ndarray
    rate_cov: np.ndarray
    rate_syst_df: pd.DataFrame
    rate_syst_dict: dict
    xsec_cov: np.ndarray | None = None
    xsec_syst_df: pd.DataFrame | None = None
    xsec_syst_dict: dict | None = None

    @property
    def has_xsec(self) -> bool:
        """True if cross-section covariance was computed."""
        return self.xsec_cov is not None


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
    systs: bool | np.ndarray | None = None
    pdg: bool = False
    pdg_col: tuple | str = 'pfp_shw_truth_p_pdg'
    mode: bool = False
    hatch: list[str] | None = None
    bin_labels: list[str] | None = None
    generic: bool = False
    overflow: bool = True
    legend_kwargs: dict | None = None


__all__ = [
    'VariableConfig',
    'XSecInputs',
    'SystematicsOutput',
    'PlottingConfig',
]