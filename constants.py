"""Configurations and constants."""
import seaborn as sns
import uproot
from . import config

# dictionary mapping signal to ints. Signal == 0 is assumed to be the desired topology. 
signal_dict = {"nueCC":0,
               "numuCCpi0":1,
               "NCpi0":2,
               "othernumuCC":3,
               "othernueCC": 4,
               "otherNC":5, 
               "nonFV":6 ,
               "dirt":7,
               "cosmic":8,
               "offbeam":9}

signal_labels = [r"CC $\nu_e$",
                 r"CC $\nu_\mu\pi^0$",
                 r"NC $\nu$$\pi^0$",
                 r"other CC $\nu_\mu$",
                 r"other CC $\nu_e$",
                 r"other NC $\nu$",
                 r"Non-FV $\nu$",
                 r"Dirt $\nu$",
                 "cosmic",
                 "offbeam"]

# default colors used for plotting 
signal_colors = ["C0", "C1", "C2", "C3", "darkslateblue", "C4","C6","C5","darkgray","lightgray"]

generic_dict = {"CCnu":0,"NCnu":1,"nonFV":2,"dirt":3,"cosmic":4}
generic_labels = [r"CC $\nu$",r"NC $\nu$",r"Non-FV $\nu$",r"Dirt $\nu$","cosmic"]
generic_colors = ["C3", "darkslateblue", "C5", "C6","C7"]

# dictionary mapping particle to pdg code, used for plotting
pdg_dict = {
    r"$e$":        {"pdg":11,   },
    r"$\mu$":      {"pdg":13,   },
    r"$\gamma$":   {"pdg":22,   },
    r"$p$":        {"pdg":2212, },
    r"$\pi^{+/-}$":{"pdg":211,  },
}

mode_dict = {
    "QE": 0,
    "RES": 1,
    "DIS": 2,
    "COH": 3,
    "MEC": 10,
}

mode_colors = sns.color_palette("Dark2", n_colors=len(mode_dict)+1)

# flux file, units: /m^2/10^6 POT, 50 MeV bins
with uproot.open(config.FLUX_FILE) as f:
    nue_flux = f["flux_sbnd_nue"].to_numpy()
    flux_vals = nue_flux[0]
integrated_flux = flux_vals.sum()/1e4 # to cm2
integrated_flux *= (180*180)/(200*200) # rescale the front face to AV front face

RHO = 1.3836  #g/cm3, liquid Ar density
N_A = 6.02214076e23 # Avogadro’s number
M_AR = 40 # g, molar mass of argon
# V_SBND = 380 * 380 * 440 # cm3, the active volume of the detector 
# x cm (drift) * z cm (width) * y cm (height), excluding 90 cm of y-dimension at high z
V_SBND = (190)*2 * ((250 - 10)*(190*2) + (450-250)*(100 + 190))
NTARGETS = RHO * V_SBND * N_A / M_AR