# Changelog

A running record of breaking changes and new features. Add a new dated section at the
top each time changes are merged that other users should know about.

---

## 2026-05-01 — Cross-section systematics framework, selection improvements, plotting overhaul

### Bug fixes

**`plot_var()` — MC stat uncertainty now detected from the covariance dictionary**
Previously the plotting function always computed MC stat as a separate uncertainty band.
It now scans the passed `systs` dict for an `"MCstat"` key first: if found, MC stat is
already folded into the covariance and a single combined stat+syst band is drawn; if not
found, a separate MC stat band is drawn alongside the systematic band. This prevents
double-counting when `get_total_cov` has already included MC stat.

**`syst.py` — GENIE/xsec knob identification no longer relies on string matching**
Previously, whether a systematic knob followed the cross-section (response-matrix) path
was decided by checking whether the column name contained the string `"GENIE"`. This
misclassified knobs whose names don't include that string (e.g. `SBNNuSyst`, `SuSAv2`).
The check now uses the authoritative lists `regen_systematics` and
`ar23p_genie_systematics` imported directly from `cafpyana`'s `geniesyst.py`, so
classification is exact regardless of naming convention.

---

### Breaking changes

**`select()` — `min_shower_length` default changed from `0.1` → `10` cm**
If your analysis relied on the old default, pass it explicitly:

```python
df_sel = nue.select(df, min_shower_length=0.1)
```

**Category dicts are now nested**
`signal_categories`, `pdg_categories`, `mode_categories`, and `generic_categories`
are now nested dicts of the form `{name: {"value": int, "label": str, "color": str}}`.
The flat `{name: value}` dicts (`signal_dict`, `pdg_dict`, `mode_dict`, `generic_dict`)
are still exported unchanged — use those if you only need integer IDs.

```python
# Before (no longer works for label/color)
color = signal_colors[i]
label = signal_labels[i]

# After
color = nue.signal_categories["nueCC"]["color"]
label = nue.signal_categories["nueCC"]["label"]
value = nue.signal_dict["nueCC"]   # still works as before
```

**Wildcard imports no longer export everything**
All modules now define `__all__`. Replace `import *` with explicit imports:

```python
# Before
from nueana.funcs import *

# After
from nueana.funcs import get_total_cov, load_detvar_dicts, add_flat_norm_uncertainty
```

**`plot_var()` and `plot_mc_data()` return an extra value**
Both functions now return a 4-tuple. Update any unpacking:

```python
# plot_var — was 3-tuple, now 4-tuple
bins, steps, total_err, syst_dict = nue.plot_var(df, var, bins, ax=ax)

# plot_mc_data — was 3-tuple, now 4-tuple
fig, ax_main, ax_sub, mc_dict = nue.plot_mc_data(mc_df, data_df, var, bins)
```

**`SystematicsOutput` and `XSecInputs` are frozen**
These dataclasses are now `frozen=True` and cannot be modified after construction.
Use `dataclasses.replace()` or the `add_*` helpers (see below) to build modified copies.

---

### New features

**Fine-grained selection control with `extra_cuts` / `skip_cuts`**

Pass additional boolean masks or skip built-in cuts without rewriting the full pipeline:

```python
# Skip the shower-length cut entirely
df_no_len = nue.select(df, skip_cuts=["cut_shower_length"])

# Add a custom cut on top of the standard selection
my_cut = df.primshw.shw.reco_energy > 0.8
df_custom = nue.select(df, extra_cuts=[my_cut])

# Combine: skip one built-in, add one custom
df_hybrid = nue.select(df, skip_cuts=["cut_direction"], extra_cuts=[my_cut])
```

Available cut names: `cut_preselection`, `cut_flash_matching`, `cut_shower_energy`,
`cut_muon_rejection`, `cut_conversion_gap`, `cut_dedx`, `cut_opening_angle`,
`cut_shower_length`, `cut_direction`.

---

**Streamlined plot config with `PlottingConfig`**

Bundle display options into a reusable dataclass instead of spelling them out every call:

```python
from nueana.classes import PlottingConfig

signal_cfg = PlottingConfig(
    xlabel=r"$\cos\theta_e$",
    ylabel="Events / bin",
    plot_err=True,
    overflow=True,
)

# Pass as config=; individual kwargs still override
bins, steps, err, systs = nue.plot_var(df, var, bins, ax=ax, config=signal_cfg)

# Override one field for a specific plot without modifying the config
bins, steps, err, systs = nue.plot_var(df, var, bins, ax=ax, config=signal_cfg, normalize=True)
```

---

**Cross-section covariance with `XSecInputs`**

Pass signal truth information to get separate event-rate and cross-section covariance matrices:

```python
from nueana.classes import XSecInputs

xsec_inputs = XSecInputs(
    true_signal_df=mcsig_df,
    true_signal_scale=1 / (nue.integrated_flux * (mc_pot / 1e6)),
    reco_var_true=nue.electron_energy().var_evt_truth_col,
    true_var_true=nue.electron_energy().var_nu_col,
)

output = nue.get_total_cov(
    reco_df=mc_df,
    reco_var=nue.electron_energy().var_evt_reco_col,
    bins=nue.electron_energy().bins,
    mcbnb_pot=mc_pot,
    select_region="signal",
    uncertainty_keys=["xsec", "detv", "norm"],
    xsec_inputs=xsec_inputs,
)

if output.has_xsec:
    print("xsec CV:  ", output.xsec_hist_cv)
    print("xsec cov:\n", output.xsec_cov)

print("rate CV:  ", output.rate_hist_cv)
print("rate cov:\n", output.rate_cov)
```

---

**Selective uncertainty inclusion with `uncertainty_keys`**

Compute only the systematic blocks you need. Allowed keys: `"rate"`, `"xsec"`, `"detv"`,
`"norm"`, `"cosmic"`. Default (when `None`): `{"rate", "detv", "norm", "cosmic"}`, plus
`"xsec"` automatically when `xsec_inputs` is provided.

```python
# Rate systematics + detector variations only
output = nue.get_total_cov(reco_df=mc_df, reco_var=var, bins=bins,
                           mcbnb_pot=mc_pot, uncertainty_keys=["rate", "detv"])

# Norm uncertainties only (fast cross-check)
output = nue.get_total_cov(reco_df=mc_df, reco_var=var, bins=bins,
                           mcbnb_pot=mc_pot, uncertainty_keys=["norm"])
```

---

**Pre-load detector variations to avoid repeated disk reads**

`load_detvar_dicts()` is slow. Load once per session and pass the result to every
`get_total_cov` call:

```python
detvar_dict = nue.load_detvar_dicts()

output_angle = nue.get_total_cov(..., detvar_dict=detvar_dict)
output_energy = nue.get_total_cov(..., detvar_dict=detvar_dict)  # no extra disk read
```

---

**Adding custom uncertainties to `SystematicsOutput`**

```python
from nueana.funcs import add_flat_norm_uncertainty, add_fractional_uncertainty, add_uncertainty

# 2% fully-correlated normalization uncertainty
output = add_flat_norm_uncertainty(output, frac_unc=0.02, key="MyNorm", category="BeamExposure")

# Per-bin fractional uncertainties (uncorrelated bin-to-bin)
frac_unc = np.array([0.05, 0.10, 0.10, 0.08])
output = add_fractional_uncertainty(output, frac_unc=frac_unc, key="MyBinUnc", correlation="diagonal")

# Fully custom covariance matrix
my_cov = np.diag([0.1, 0.2, 0.2, 0.1]) ** 2
output = add_uncertainty(output, cov=my_cov, key="MyCustom", category="MyCategory", target="rate")
```

`target` controls which covariance matrix the source is added to: `"rate"`, `"xsec"`, or `"both"`.

---

**Region-aware detector variations**

```python
output_sig  = nue.get_total_cov(..., select_region="signal")   # default
output_ctrl = nue.get_total_cov(..., select_region="control")
```

---

**Event masking**

Filter a selected dataframe to signal-only or background-only events
(requires `define_signal` to have been called first):

```python
from nueana.utils import apply_event_mask

df_signal_only     = apply_event_mask(df, "signal")      # signal == 0
df_background_only = apply_event_mask(df, "background")  # signal != 0
df_all             = apply_event_mask(df, "all")          # no filter
```

---

**Data/MC ratio panel and chi-squared annotation in `plot_mc_data()`**

`plot_mc_data()` now automatically draws a data/MC ratio subplot below the main stack
and annotates the main axis with the integrated Data/MC ratio and a chi-squared
goodness-of-fit test (using `scipy.stats.chi2` when available). Both can be suppressed:

```python
fig, ax_main, ax_sub, mc_dict = nue.plot_mc_data(
    mc_df, data_df, var, bins,
    annot=False,          # suppress Data/MC and chi-sq text
    ratio_min=0.5,        # customize ratio panel y-limits
    ratio_max=1.5,
)

# The returned mc_dict includes the chi-sq value for downstream use
chi2  = mc_dict["chi2"]
p_val = mc_dict["p_value"]
ratio = mc_dict["ratio"]      # integrated Data/MC
```

---

**Systematics breakdown plots**

```python
fig, axes, angle_summary, energy_summary = nue.plot_syst_category_breakdown(
    angle_syst_df=output_angle.rate_syst_df,
    energy_syst_df=output_energy.rate_syst_df,
    category_dict=nue.category_dict_signal,
    angle_var=r"$\cos\theta_e$",
    energy_var=r"$E_e$ [GeV]",
    region_label="Signal Region",
)

# Drill into a single category
fig, axes = nue.plot_syst_breakdown(
    angle_syst_df=output_angle.rate_syst_df,
    energy_syst_df=output_energy.rate_syst_df,
    category="GENIE",
    category_dict=nue.category_dict_signal,
)
```
