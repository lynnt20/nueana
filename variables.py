"""Pre-built VariableConfig instances for the nue CC cross-section analysis.

Add new analysis variables here as module-level factory functions rather than
baking them into the VariableConfig class.
"""
import numpy as np
from .classes import VariableConfig

__all__ = ['electron_energy', 'electron_direction']


def electron_energy() -> VariableConfig:
    """VariableConfig for primary electron energy (GeV)."""
    return VariableConfig(
        var_save_name="energy",
        var_plot_name="$E_{e-}$",
        var_unit="GeV",
        bins=np.array([0.5, 0.7, 0.95, 1.25, 1.7, 2.5]),
        bin_labels=np.array([0.5, 0.7, 0.95, 1.25, 1.7, 5]),
        var_evt_reco_col=('primshw', 'shw', 'reco_energy'),
        var_evt_truth_col=('slc', 'truth', 'e', 'genE'),
        var_nu_col=('e', 'genE'),
    )


def electron_direction() -> VariableConfig:
    """VariableConfig for primary electron direction (cos theta)."""
    return VariableConfig(
        var_save_name="direction",
        var_plot_name="$\\cos\\theta_{e-}$",
        var_unit="",
        bins=np.array([0.5, 0.6, 0.75, 0.85, 0.925, 1.0]),
        bin_labels=np.array([0.0, 0.6, 0.75, 0.85, 0.925, 1.0]),
        var_evt_reco_col=('primshw', 'shw', 'dir', 'z'),
        var_evt_truth_col=('slc','truth','e', 'dir', 'z'),
        var_nu_col=('e', 'dir', 'z'),
    )
