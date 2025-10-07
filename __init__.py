# Import all modules
from . import constants
from . import utils
from . import plotting
from . import corrections
from . import selections
from . import dataframes
from . import masks

# Import commonly used functions for convenience
from .constants import signal_dict, signal_labels, pdg_dict, colors
from .utils import flatten_df, get_slc, get_slices, get_evt, get_signal_evt, get_backgr_evt
from .selections import cutPreselection, cutShowerEnergy, cutContainment, cutCRUMBS
from .corrections import shw_energy_fix
from .plotting import plot_var, plot_var_pdg, plot_mc_data, data_plot_overlay
from .masks import maskTrueVtxFv
# from .dataframes import defineBackground, getPDGCounts, getPFP

# This allows both:
# import nue; nue.cutPreselection(df)
# from nue import cutPreselection; cutPreselection(df)