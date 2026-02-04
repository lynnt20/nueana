"""Configurations and constants."""
import seaborn as sns
import uproot
# dictionary mapping signal to ints. Signal == 0 is assumed to be the desired topology. 
signal_dict = {"nueCC":0,"numuCCpi0":1,"NCpi0":2,"othernumuCC":3,"othernueCC": 4,"otherNC":5, "nonFV":6 ,"dirt":7,"cosmic":8}
signal_labels = [r"CC $\nu_e$",
                 r"CC $\nu_\mu\pi^0$",
                 r"NC$\nu$$\pi^0$",
                 r"other CC $\nu_\mu$",
                 r"other CC $\nu_e$",
                 r"other NC $\nu$",
                 r"Non-FV $\nu$",
                 r"Dirt $\nu$",
                 "cosmic"]

# default colors used for plotting 
signal_colors = ["C0", "C1", "C2", "C3", "darkslateblue", "C4", "C5", "C6","C7"]

# dictionary mapping particle to pdg code, used for plotting
pdg_dict = {
    r"$e$":    {"pdg":11,   "mass":0.000511},
    r"$\mu$":   {"pdg":13,   "mass":0.105658},
    r"$\gamma$": {"pdg":22,   "mass":0},
    r"$p$":     {"pdg":2212, "mass":0.938272},
    # "pi0": {"pdg":111, "mass":0.134976},
    r"$\pi^{+/-}$":   {"pdg":211,   "mass":0.139570},
    # "n": {"pdg":2112, "mass":0.939565},
    # "other": {"pdg":0, "mass":0}
}

# flux file, units: /m^2/10^6 POT, 50 MeV bins
fluxfile = "/exp/sbnd/data/users/lynnt/xsection/flux/sbnd_original_flux.root"
with uproot.open(fluxfile) as f:
    nue_flux = f["flux_sbnd_nue"].to_numpy()
    flux_vals = nue_flux[0]
integrated_flux = flux_vals.sum()/1e4 # to cm2