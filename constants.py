"""Configurations and constants."""

# dictionary mapping signal to ints. Signal == 0 is assumed to be the desired topology. 
signal_dict = {"nueCC":0,"numuCCpi0":1,"NCpi0":2,"othernumuCC":3,"othernueCC": 4,"otherNC":5, "nonFV":6 ,"dirt":7,"cosmic":8}
signal_labels = [r"CC $\nu_e$ ",
                     r"CC $\nu_\mu\pi^0$",
                     r"NC$\nu$$\pi^0$",
                     r"other CC $\nu_\mu$",
                     r"other CC $\nu_e$",
                     r"other NC $\nu$",
                     r"Non-FV $\nu$",
                     r"Dirt $\nu$",
                     "cosmic"]

# default colors used for plotting 
colors = ["C0", "C1", "C2", "C3", "darkslateblue", "C4", "C5", "C6","C7"]

# dictionary mapping particle to pdg code, used for plotting
pdg_dict = {
    "e-": {"pdg":11, "mass":0.000511},
    "mu-": {"pdg":13, "mass":0.105658},
    "gamma": {"pdg":22, "mass":0},
    "p": {"pdg":2212, "mass":0.938272},
    # "pi0": {"pdg":111, "mass":0.134976},
    "pi+": {"pdg":211, "mass":0.139570},
    "n": {"pdg":2112, "mass":0.939565},
    "other": {"pdg":0, "mass":0}
}

