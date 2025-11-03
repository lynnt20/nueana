# Import all modules
from . import constants
from . import utils
from . import plotting
from . import corrections
from . import selections
from . import dataframes
from . import masks

# Import commonly used functions for convenience
from .constants import signal_dict, signal_labels, pdg_dict
from .utils import get_n_split, print_keys, load_dfs, get_mcexposure_info, define_signal
from .selections import cutPreselection, cutShowerEnergy, cutContainment, cutCRUMBS
from .corrections import shw_energy_fix
from .plotting import plot_var, plot_var_pdg, data_plot_overlay, plot_mc_data
from .masks import maskTrueVtxFv
# from .dataframes import defineBackground, getPDGCounts, getPFP

# This allows both:
# import nue; nue.cutPreselection(df)
# from nue import cutPreselection; cutPreselection(df)