# nueana

Utilities for SBND CCnue cross-section analysis.

This package is designed for notebook and script workflows where CAF-derived dataframes
have already been produced (typically via `cafpyana`) and you want to run selection,
plotting, and uncertainty studies.

## What this package provides

- Selection helpers for signal and sideband studies.
- Signal-category definitions for MC truth labeling.
- Histogram utilities with overflow handling.
- Plotting helpers for stacked MC, data overlays, and data/MC ratios.
- Systematic uncertainty tools (covariance, correlation, universe handling, detector-variation helpers).
- I/O helpers for split HDF5 dataframe files (output of cafpyana).
- Common constants and geometry utilities.

## Package layout

- `config.py`: Global paths and environment setup (cafpyana path, flux file, detvar files).
- `constants.py`: Signal/background category dicts, physics constants, flux values.
- `classes.py`: Core dataclasses — `VariableConfig`, `XSecInputs`, `SystematicsOutput`, `PlottingConfig`.
- `variables.py`: Pre-built `VariableConfig` factory functions (`electron_energy`, `electron_direction`).
- `selection.py`: Event selection and signal labeling — `select`, `select_sideband`, `define_signal`, `define_generic`.
- `plotting.py`: MC/data plotting — `plot_var`, `plot_mc_data`, `data_plot_overlay`, `plot_syst_category_breakdown`, `plot_syst_breakdown`.
- `funcs.py`: High-level systematics driver — `get_total_cov`, `load_detvar_dicts`, `add_uncertainty`, `add_flat_norm_uncertainty`, `add_fractional_uncertainty`.
- `syst.py`: Low-level systematics — `get_syst`, `get_syst_hists`, `get_detvar_systs`, `calc_matrices`, `get_syst_df`, `mcstat`.
- `histogram.py`: Histogram wrappers with overflow — `get_hist1d`, `get_hist2d`.
- `utils.py`: DataFrame helpers — `ensure_lexsorted`, `merge_hdr`, `apply_event_mask`.
- `io.py`: Split-HDF5 loading — `load_dfs`, `get_n_split`, `print_keys`.
- `geometry.py`: Detector geometry — `whereTPC`.

## Quickstart for a new analysis

There are four things to update when adapting this package to a different signal or
selection: the paths in `config.py`, the signal categories in `constants.py`, the
selection cuts in `selection.py`, and the analysis variables in `variables.py`.

### 1. Update paths — `config.py`

Set the paths for your environment before importing anything else:

```python
import nueana.config as config

config.CAFPYANA_PATH  = "/path/to/your/cafpyana"
config.FLUX_FILE      = "/path/to/your/flux.root"
config.INTIME_FILE    = "/path/to/your/intime.df"
config.DETVAR_DICT_SIGNAL  = "/path/to/your/detvar_signal.pkl"
config.DETVAR_DICT_CONTROL = "/path/to/your/detvar_control.pkl"
```

> **Note:** `INTIME_FILE`, `DETVAR_DICT_SIGNAL`, and `DETVAR_DICT_CONTROL` are only
> used by the systematic uncertainty functions (`get_total_cov`, `get_intime_cov`,
> `get_detvar_systs`). If you are not yet running systematics, these paths can be
> left as-is. `FLUX_FILE` is the exception — it is read at import time by
> `constants.py`, so it must point to a valid file before `import nueana` is called.

### 2. Define your signal categories — `constants.py`

`signal_categories` drives the integer labels written by `define_signal()` and the
colors/labels used by `plot_var()`. Each entry needs a `"value"` (integer ID),
`"label"` (legend text), and `"color"`. By convention, the signal topology has ID `0`.

```python
# In constants.py — replace or extend signal_categories with your own channels
signal_categories = {
    "mySignal":   {"value": 0,  "label": r"My Signal",  "color": "steelblue"},
    "background1":{"value": 1,  "label": "Background 1","color": "tomato"},
    "nonFV":      {"value": 10, "label": "Non-FV",      "color": "gray"},
    "dirt":       {"value": 11, "label": "Dirt",        "color": "peru"},
    "cosmic":     {"value": 12, "label": "Cosmic",      "color": "orchid"},
    "offbeam":    {"value": 13, "label": "Off-beam",    "color": "silver"},
}
signal_dict = {k: v["value"] for k, v in signal_categories.items()}
```

Then update `define_signal()` in `selection.py` to assign your integer IDs.

### 3. Adjust selection cuts — `selection.py`

Pass cut overrides directly to `select()` without touching the source, or skip cuts
entirely:

```python
import nueana as nue

# Override individual cut thresholds
df_sel = nue.select(df, min_shower_energy=0.3, max_track_length=150)

# Skip a cut that doesn't apply to your topology
df_sel = nue.select(df, skip_cuts=["cut_muon_rejection"])

# Add a custom cut on top of the standard pipeline
my_cut = df.primshw.shw.open_angle < 0.1
df_sel = nue.select(df, extra_cuts=[my_cut])
```

`select()` can return all intermediate stages for cut-flow studies:

```python
stages = nue.select(df, savedict=True)
# Keys: 'preselection', 'flash matching', 'shower energy',
#       'muon rejection', 'conversion gap', 'dEdx',
#       'opening angle', 'shower length', 'shower density'

# Or stop at a specific stage
df_presel = nue.select(df, stage="preselection")
```

### 4. Define your analysis variables — `variables.py`

Add a factory function returning a `VariableConfig` for each variable you want to unfold:

```python
from nueana.classes import VariableConfig
import numpy as np

def my_variable() -> VariableConfig:
    return VariableConfig(
        var_save_name  = "my_var",
        var_plot_name  = r"$p_T$",
        var_unit       = "GeV",
        bins           = np.array([0.0, 0.2, 0.5, 1.0, 2.0]),
        bin_labels     = np.array([0.0, 0.2, 0.5, 1.0, 2.0]),
        var_evt_reco_col   = ("primshw", "shw", "my_reco_col", "", "", ""),
        var_evt_truth_col  = ("slc", "truth", "e", "my_truth_col"),
        var_nu_col         = ("e", "my_truth_col"),
    )
```

### Minimal working example

```python
import numpy as np
import nueana as nue
from nueana.classes import PlottingConfig

# Load CAF-derived dataframes
mc_dfs   = nue.load_dfs("/path/to/mc.df",   ["mcnu", "hdr", "nuecc"])
data_dfs = nue.load_dfs("/path/to/data.df", ["hdr",  "nuecc"])

# Run selection and label signal categories
mc_df   = nue.define_signal(nue.select(mc_dfs["nuecc"],   savedict=False))
data_df = nue.select(data_dfs["nuecc"], savedict=False)

# Make a stacked MC + data plot
cfg = PlottingConfig(xlabel="Reco shower energy [GeV]", plot_err=True)
fig, ax_main, ax_sub, mc_dict = nue.plot_mc_data(
    mc_df=mc_df, data_df=data_df,
    var=nue.electron_energy().var_evt_reco_col,
    bins=nue.electron_energy().bins,
    config=cfg,
)
```

## Notes and caveats

- `constants.py` reads the flux ROOT file at import time. If the file is unavailable,
  importing `nueana` will fail.
- Several utilities assume pandas MultiIndex columns with CAF-style naming.
- Many routines expect a `signal` column to already be present — call `define_signal()`
  or `define_generic()` before plotting or applying event masks.
- Overflow is enabled by default (`overflow=True`) and folds out-of-range values into
  the first/last bin.
