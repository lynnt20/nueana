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

## Package layout

- `analysis.py`: **The single file to edit for a new analysis.** Signal/background category dicts, physics constants, flux values, cut sequences, truth-labeling functions, and analysis variable definitions.
- `classes.py`: Core dataclasses — `CutSpec`, `VariableConfig`, `SystematicsInput`, `SystematicsOutput`, `PlottingConfig`.
- `config.py`: Global paths and environment setup (cafpyana path, flux file, detvar files).
- `preprocess.py`: DataFrame preprocessing — column fixes, pi0 kinematics, secondary shower energy corrections. **Contains hard-coded CAF column names specific to Gen1 nuecc; update for a different analyzer.**
- `selection.py`: Event selection pipeline (`select`, `select_sideband`) and cut helpers (`drop_cuts`, `modify_cut`).
- `plotting.py`: Stacked MC, data overlay, data/MC ratio, and systematics breakdown plots.
- `funcs.py`: High-level systematics driver — total covariance calculation and custom uncertainty helpers.
- `syst.py`: Low-level systematics — universe histograms, covariance matrices, detector variations.
- `utils.py`: DataFrame helpers for MultiIndex sorting, header merging, event masking, and histograms.
- `io.py`: HDF5 dataframe loading — split-file primitives (`load_dfs`) and high-level loaders (`load_mc`, `load_data`).
- `ccbc.py`: Cross-region Background Constraint (CCBC) — covariance propagation, constrained background estimation, and comparison plots.
- `fdt.py`: Fake-data tests — `UnfoldInput` container, WienerSVD unfolding driver, and random-background ensemble (`run_random_background_fdt`).
- `detvar/`: Detector variation (DetVar) subpackage. See [`detvar/README.md`](detvar/README.md).

## Quickstart for a new analysis

There are four things to update when adapting this package to a different signal or
selection: the file paths in `config.py`, the signal categories, the cut sequences,
and the analysis variables — the latter three all in `analysis.py`.

### 1. Update paths — `config.py`

Set the paths for your environment:

```python
# In config.py
CAFPYANA_PATH = "/path/to/your/cafpyana"
FLUX_FILE     = "/path/to/your/flux.root"
INTIME_FILE   = "/path/to/your/mc_intime.df"

DETVAR_DICT_DIR     = "/path/to/your/detvars/"
DETVAR_DICT_SIGNAL  = DETVAR_DICT_DIR + "detvars_signal.h5"
DETVAR_DICT_CONTROL = DETVAR_DICT_DIR + "detvars_sideband.h5"
```

> **Note:** `FLUX_FILE` is read at import time by `analysis.py`, so it must point to a
> valid file before `import nueana` is called. The detvar and intime paths are only used
> when running systematics and can be left as placeholders until then.

### 2. Define your signal categories — `analysis.py`

`signal_categories` drives the integer labels written by `define_signal()` and the
colors/labels used by `plot_mc_data()`. Each entry needs a `"value"` (integer ID),
`"label"` (legend text), and `"color"`. By convention, the signal topology has ID `0`.

```python
# In analysis.py — replace or extend signal_categories with your own channels
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

Then update `define_signal()` in `analysis.py` to assign your integer IDs.

### 3. Adjust selection cuts — `analysis.py`

`DEFAULT_CUTS` and `SIDEBAND_CUTS` are defined in `analysis.py`. Pass cut overrides
directly to `select()` without touching the source, or modify the sequences in place:

```python
import nueana as nue
from nueana.selection import DEFAULT_CUTS, modify_cut, drop_cuts, CutSpec

# Override individual cut thresholds
cuts = nue.modify_cut(nue.DEFAULT_CUTS, "dedx", min=1.5, max=3.0)
df_sel = nue.select(df, cuts=cuts)

# Drop a cut that doesn't apply to your topology
cuts = nue.drop_cuts(nue.DEFAULT_CUTS, "muon_rejection")

# Add a custom cut on top of the standard pipeline
my_cut = nue.CutSpec("my_cut", fn=lambda df: df.x > 10)
cuts = nue.DEFAULT_CUTS + [my_cut]
```

`select()` can return all intermediate stages for cut-flow studies:

```python
stages = nue.select(df, savedict=True)

# Or stop at a specific stage
df_preflash = nue.select(df, stage="flash_pe")
```

### 5. Update preprocessing — `preprocess.py`

`preprocess_mc` and `preprocess_data` apply fixes that may be analysis specific, or 
MC/data version specific. For a different analyzer, update the individual fix functions
(`fix_flash_pe_scale`, `fix_flash_time`, `fix_prim_shw_energy`, `fix_sec_shw_energy`,
`add_phi`) with your variables, or replace the functions entirely. 

### 4. Define your analysis variables — `analysis.py`

Add a factory function returning a `VariableConfig` for each variable you want to plot or unfold:

```python
from nueana.classes import VariableConfig
import numpy as np

def my_variable() -> VariableConfig:
    return VariableConfig(
        var_save_name      = "my_var",
        var_plot_name      = r"$p_T$",
        var_unit           = "GeV",
        bins               = np.array([0.0, 0.2, 0.5, 1.0, 2.0]),
        bin_labels         = np.array([0.0, 0.2, 0.5, 1.0, 2.0]),
        var_evt_reco_col   = ("primshw", "shw", "my_reco_col", "", "", ""),
        var_evt_truth_col  = ("slc", "truth", "e", "my_truth_col"),
        var_nu_col         = ("e", "my_truth_col"),
    )
```

### Minimal working example

```python
import numpy as np
import nueana as nue

# load_mc:  preprocess → select (cuts, optional) → merge_hdr → define_signal
# load_data: preprocess → merge_hdr → select (cuts, optional) → stamp offbeam signal label
mc_df, mc_pot, mc_ngen     = nue.load_mc  ("/path/to/mc.df",   keys=["nuecc","hdr"], cuts=nue.DEFAULT_CUTS)
data_df, data_pot, ngates  = nue.load_data("/path/to/data.df", keys=["nuecc","hdr"], onbeam=True, cuts=nue.DEFAULT_CUTS)

# Make a stacked MC + data plot
cfg = nue.PlottingConfig(scale=data_pot/mc_pot, ylabel=f"Events [{data_pot:.2e} POT]")
output = nue.plot_mc_data(
    mc_df=mc_df, data_df=data_df,
    var=nue.electron_energy().var_evt_reco_col,
    bins=nue.electron_energy().bins,
    xlabel="Reco shower energy [GeV]",
    config=cfg,
)
```

See [`examples/signal_plots.ipynb`](examples/signal_plots.ipynb) for a complete worked example of signal-region plots, and [`examples/sideband_plots.ipynb`](examples/sideband_plots.ipynb) for the sideband equivalent.

## Notes and caveats

- `analysis.py` reads the flux ROOT file at import time. If the file is unavailable,
  importing `nueana` will fail.
- Several utilities assume pandas MultiIndex columns with CAF-style naming.
- Many routines expect a `signal` column to already be present — call `define_signal()`
  or `define_generic()` before plotting or applying event masks.
- Overflow is enabled by default (`overflow=True`) and folds out-of-range values into
  the first/last bin.
