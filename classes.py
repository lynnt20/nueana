from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

class VariableConfig:
    """
    A configurable class for setting up unfolding variable configurations.
    Choose a configuration using one of the provided class methods,
    or instantiate directly with custom parameters.
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


    @classmethod
    def electron_energy(cls):
        return cls(
            var_save_name="energy",
            var_plot_name="$E_{e-}$",
            var_unit="GeV",
            bins=np.array([0.5,0.7,0.95,1.25,1.7,2.5]),
            bin_labels =  np.array([0.5, 0.7, 0.95, 1.25, 1.7, 5]),
            var_evt_reco_col=('primshw', 'shw', 'reco_energy'),
            var_evt_truth_col=('slc','truth','e','genE'),
            var_nu_col=('e','genE'),
        )

    @classmethod
    def electron_direction(cls):
        return cls(
            var_save_name="direction",
            var_plot_name="$\\cos\\theta_{e-}$",
            var_unit="",
            bins= np.array([0.5,0.6,0.75,0.85,0.925,1.0]),
            bin_labels =  np.array([0.0  , 0.6  , 0.75 , 0.85 , 0.925, 1.   ]),
            var_evt_reco_col=('primshw', 'shw', 'dir','z'),
            var_evt_truth_col=('e','dir','z'),
            var_nu_col=('e','dir','z')
        )

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


__all__ = [
    'VariableConfig',
    'XSecInputs',
    'SystematicsOutput',
]