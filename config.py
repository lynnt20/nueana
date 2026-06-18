"""Global configuration and paths for nueana module."""

import sys
from pathlib import Path

# ========================
# Directory Paths
# ========================

# Root directories
NUEANA_DIR    = "/exp/sbnd/data/users/lynnt/xsection/nueana"
CAFPYANA_PATH = "/exp/sbnd/app/users/lynnt/cafpyana"

# Setup cafpyana - append to sys.path so we can import cafpyana modules
if CAFPYANA_PATH not in sys.path:
    sys.path.append(CAFPYANA_PATH)
    sys.path.append(CAFPYANA_PATH+"/analysis_village")

# ========================
# Data and File Paths
# ========================

# Flux file path (legacy, nue only, flat front-face units)
FLUX_FILE = "/exp/sbnd/data/users/lynnt/xsection/flux/sbnd_original_flux.root"

# Volume-averaged BNB gsimple flux (nue+nuebar, FV_split_truncY_eastonly)
# Units: cm^-2 POT^-1 per 50 MeV bin
FLUX_FILE_NEW = "/exp/sbnd/app/users/lynnt/cafpyana/analysis_village/flux/sbnd_flux_new.root"

# In-time cosmic sample file path
INTIME_FILE = "/exp/sbnd/data/users/lynnt/xsection/samples/MCP2025B_v10_06_00_09/dfs_nu26/mc_intime.df"

# Detector variation (detvar) dictionaries path
# List of pickle files to load and combine for detector variations
DETVAR_DICT_DIR = "/exp/sbnd/data/users/lynnt/xsection/samples/MCP2025B_v10_06_00_09/dfs_nu26/detvars"
DETVAR_DICT_FILES = [DETVAR_DICT_DIR + "/detvars.h5",]
DETVAR_DICT_SIGNAL = DETVAR_DICT_DIR + "/detvars_signal.h5"
DETVAR_DICT_CONTROL = DETVAR_DICT_DIR + "/detvars_sideband.h5"

# ========================
# Path Verification (Optional)
# ========================

def _verify_path(path, name):
    """Verify that a path exists and is accessible."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{name} not found at: {path}")
    return path

# Set to True for debugging to verify all critical paths exist
VERIFY_PATHS = False

if VERIFY_PATHS:
    _verify_path(CAFPYANA_PATH, "CAFPYANA_PATH")
    _verify_path(FLUX_FILE, "FLUX_FILE")
    _verify_path(INTIME_FILE, "INTIME_FILE")
    for i, detvar_file in enumerate(DETVAR_DICT_FILES):
        _verify_path(detvar_file, f"DETVAR_DICT_FILES[{i}]")
