"""Configurations and constants."""
import seaborn as sns
import uproot
from . import config

__all__ = [
    'signal_categories', 'signal_dict',
    'generic_categories', 'generic_dict',
    'pdg_categories', 'pdg_dict',
    'mode_categories', 'mode_dict',
    'nue_flux', 'flux_vals', 'integrated_flux',
    'RHO', 'N_A', 'M_AR', 'V_SBND', 'NTARGETS',
    'POT_NORM_UNC', 'NTARGETS_UNC',
]

# Signal == 0 is assumed to be the desired topology.
# Note: nonFV uses "C6" and dirt uses "C5" — intentionally non-sequential.
signal_categories = {
    "nueCC":       {"value": 0, "label": r"CC $\nu_e$",         "color": "C0"},
    "numuCCpi0":   {"value": 1, "label": r"CC $\nu_\mu\pi^0$",  "color": "C1"},
    "NCpi0":       {"value": 2, "label": r"NC $\nu$$\pi^0$",    "color": "C2"},
    "othernumuCC": {"value": 3, "label": r"other CC $\nu_\mu$", "color": "C3"},
    "othernueCC":  {"value": 4, "label": r"other CC $\nu_e$",   "color": "darkslateblue"},
    "otherNC":     {"value": 5, "label": r"other NC $\nu$",     "color": "C4"},
    "nonFV":       {"value": 6, "label": r"Non-FV $\nu$",       "color": "C6"},
    "dirt":        {"value": 7, "label": r"Dirt $\nu$",         "color": "C5"},
    "cosmic":      {"value": 8, "label": "cosmic",              "color": "darkgray"},
    "offbeam":     {"value": 9, "label": "offbeam",             "color": "lightgray"},
}
signal_dict = {k: v["value"] for k, v in signal_categories.items()}

generic_categories = {
    "CCnu":   {"value": 0, "label": r"CC $\nu$",     "color": "C3"},
    "NCnu":   {"value": 1, "label": r"NC $\nu$",     "color": "darkslateblue"},
    "nonFV":  {"value": 2, "label": r"Non-FV $\nu$", "color": "C5"},
    "dirt":   {"value": 3, "label": r"Dirt $\nu$",   "color": "C6"},
    "cosmic": {"value": 4, "label": "cosmic",        "color": "C7"},
}
generic_dict = {k: v["value"] for k, v in generic_categories.items()}

# PDG categories for plotting. The 5 named entries use pdg-code filtering; the 4
# extras (pdg=None) use filter-based population selection. Insertion order matters:
# named entries must precede extras so that "other_nu" is built from the remainder.
pdg_categories = {
    r"$e$":            {"pdg": 11,   "color": "C0"},
    r"$\mu$":          {"pdg": 13,   "color": "C1"},
    r"$\gamma$":       {"pdg": 22,   "color": "C2"},
    r"$p$":            {"pdg": 2212, "color": "C3"},
    r"$\pi^{+/-}$":    {"pdg": 211,  "color": "darkslateblue"},
    r"non-$\nu$ $e$":  {"pdg": None, "color": "C4",       "filter": "notprime"},
    "cosmic":          {"pdg": None, "color": "darkgray",  "filter": "cosmic"},
    "offbeam":         {"pdg": None, "color": "lightgray", "filter": "offbeam"},
    "other":           {"pdg": None, "color": "sienna",    "filter": "other_nu"},
}
pdg_dict = pdg_categories

_mode_palette = sns.color_palette("Dark2", n_colors=7)
mode_categories = {
    "QE":            {"value": 0,    "color": _mode_palette[0]},
    "RES":           {"value": 1,    "color": _mode_palette[1]},
    "DIS":           {"value": 2,    "color": _mode_palette[2]},
    "COH":           {"value": 3,    "color": _mode_palette[3]},
    "MEC":           {"value": 10,   "color": _mode_palette[4]},
    r"other $\nu$":  {"value": None, "color": _mode_palette[5], "filter": "other_nu"},
    r"non $\nu$":    {"value": None, "color": "darkgray",       "filter": "non_nu"},
}
mode_dict = {k: v["value"] for k, v in mode_categories.items() if v["value"] is not None}

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

POT_NORM_UNC  = 0.02  # fractional uncertainty on beam exposure (POT counting)
NTARGETS_UNC  = 0.01  # fractional uncertainty on number of Ar targets