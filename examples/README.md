# examples

Worked example notebooks for nueana.

- [`signal_plots.ipynb`](signal_plots.ipynb) — signal-region stacked MC + data plots, cut-flow, and systematic uncertainties.
- [`sideband_plots.ipynb`](sideband_plots.ipynb) — same workflow applied to the sideband region.
- [`ccbc.ipynb`](ccbc.ipynb) — CCBC sideband constraint: block covariance structure, per-systematic constraint comparison, fake data tests, and GiBUU alternate-generator test with data-statistical softening.
- [`fdt.ipynb`](fdt.ipynb) — unfolding fake data tests: response matrices, Asimov closure, MC-model variations, and GiBUU alternate-generator test.

These notebooks assume dataframes have already been produced by `cafpyana` and that paths in `config.py` are configured for your environment. See the [package README](../README.md) for setup instructions.
