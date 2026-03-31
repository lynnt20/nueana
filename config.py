"""Global configuration and paths for nueana module."""

import sys
from pathlib import Path

# ========================
# Directory Paths
# ========================

# Root directories
XSECTION_DIR  = "/exp/sbnd/data/users/lynnt/xsection"
NUEANA_DIR    = "/exp/sbnd/data/users/lynnt/xsection/nueana"
CAFPYANA_PATH = "/exp/sbnd/app/users/lynnt/cafpyana"

# Setup cafpyana - append to sys.path so we can import cafpyana modules
if CAFPYANA_PATH not in sys.path:
    sys.path.append(CAFPYANA_PATH)

# ========================
# Data and File Paths
# ========================

# Flux file path
FLUX_FILE = "/exp/sbnd/data/users/lynnt/xsection/flux/sbnd_original_flux.root"

# In-time cosmic sample file path
INTIME_FILE = "/scratch/7DayLifetime/lynnt/MCP2025B_v10_06_00_09/intime.df"

# Detector variation (detvar) dictionaries path
# List of pickle files to load and combine for detector variations
DETVAR_DICT_DIR = "/exp/sbnd/data/users/lynnt/xsection/samples/MCP2025B_v10_06_00_09/mc/dfs/detvars"
DETVAR_DICT_FILES = [
    DETVAR_DICT_DIR + "/detvar_dict_combined.pkl",
]
DETVAR_DICT_SIGNAL = DETVAR_DICT_DIR + "/detvar_dict_signal.pkl"
DETVAR_DICT_CONTROL = DETVAR_DICT_DIR + "/detvar_dict_control.pkl"

# Data directories
DATA_DIR = "/exp/sbnd/data/users/lynnt/xsection/data"
SAMPLES_DIR = "/exp/sbnd/data/users/lynnt/xsection/samples"
FIGURES_DIR = "/exp/sbnd/data/users/lynnt/xsection/figures"

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
