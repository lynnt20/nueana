"""Generic cut infrastructure for nue CC selection.

:class:`~nueana.classes.CutSpec` (defined in :mod:`nueana.classes`) and the
functions to build, modify, and apply cut sequences live here.

The analysis-specific cut sequences (:data:`~nueana.analysis.DEFAULT_CUTS`,
:data:`~nueana.analysis.SIDEBAND_CUTS`) and signal categorisation live in
:mod:`nueana.analysis`.

**Tighten or loosen a cut**::

    from nueana import modify_cut, select, DEFAULT_CUTS

    cuts = modify_cut(DEFAULT_CUTS, "dedx", min=1.5, max=3.0)
    df_sel = select(df, cuts=cuts)

**Drop a cut**::

    from nueana import DEFAULT_CUTS, drop_cuts

    cuts = drop_cuts(DEFAULT_CUTS, "muon_rejection")
    cuts = drop_cuts(DEFAULT_CUTS, "direction", "shower_length")

**Add a custom cut**::

    from nueana.classes import CutSpec
    cuts = DEFAULT_CUTS + [CutSpec("my_cut", fn=lambda df: df.x > 10)]

**Adjust a parameter of an fn-based cut** (combine with ``functools.partial``)::

    from functools import partial
    cuts = modify_cut(DEFAULT_CUTS, "muon_rejection",
                      fn=partial(cut_muon_rejection, max_track_length=100))
"""

from __future__ import annotations

import warnings

import pandas as pd
from dataclasses import replace
from functools import reduce

from .classes import CutSpec


__all__ = ['drop_cuts', 'modify_cut', 'select', 'select_sideband']


# ---------------------------------------------------------------------------
# Cut application
# ---------------------------------------------------------------------------

def _mask(df, spec):
    """Return a boolean mask for spec applied to df."""
    if spec.fn is not None:
        return spec.fn(df)
    series = spec.accessor(df) if spec.accessor is not None else reduce(getattr, spec.variable, df)
    return (series > spec.min) & (series < spec.max)


def drop_cuts(cuts, *names):
    """Return a copy of cuts with the named cut(s) removed.

    Parameters
    ----------
    cuts : list of CutSpec
        The cut sequence to modify.
    *names : str
        Names of cuts to drop.

    Examples
    --------
    >>> cuts = drop_cuts(DEFAULT_CUTS, "muon_rejection")
    >>> cuts = drop_cuts(DEFAULT_CUTS, "direction", "shower_length")
    """
    unknown = set(names) - {c.name for c in cuts}
    if unknown:
        raise ValueError(f"No cuts named {sorted(unknown)}. Available: {[c.name for c in cuts]}")
    return [c for c in cuts if c.name not in names]


def modify_cut(cuts, name, **kwargs):
    """Return a copy of cuts with the named CutSpec updated.

    Parameters
    ----------
    cuts : list of CutSpec
        The cut sequence to modify.
    name : str
        Name of the cut to update.
    **kwargs
        Fields to update on the matching CutSpec (passed to
        ``dataclasses.replace``).

    Returns
    -------
    list of CutSpec
        New list with the named entry replaced.

    Raises
    ------
    ValueError
        If no cut with the given name exists.

    Examples
    --------
    >>> cuts = modify_cut(DEFAULT_CUTS, "dedx", min=1.5, max=3.0)
    >>> cuts = modify_cut(cuts, "muon_rejection",
    ...                   fn=partial(cut_muon_rejection, max_track_length=100))
    """
    names = [c.name for c in cuts]
    if name not in names:
        raise ValueError(f"No cut named '{name}'. Available: {names}")
    return [replace(c, **kwargs) if c.name == name else c for c in cuts]


# ---------------------------------------------------------------------------
# Selection pipeline
# ---------------------------------------------------------------------------

def select(
    indf: pd.DataFrame,
    cuts: list[CutSpec] | None = None,
    stage: str | None = None,
    savedict: bool = False,
    check_preprocessed: bool = True,
) -> pd.DataFrame | dict:
    """Apply a sequence of cuts to a DataFrame.

    Parameters
    ----------
    indf : pandas.DataFrame
        Input DataFrame. Must have ``primshw.shw.reco_energy`` set — call
        :func:`~nueana.preprocess.preprocess_mc` or
        :func:`~nueana.preprocess.preprocess_data` first.
    cuts : list of CutSpec, optional
        Ordered cut sequence. Defaults to ``DEFAULT_CUTS`` from
        :mod:`nueana.cuts`. Use :func:`modify_cut` to adjust individual cuts,
        or build a list from scratch using :class:`~nueana.classes.CutSpec`.
    stage : str, optional
        Stop and return after this cut (matched by ``CutSpec.name``).
    savedict : bool, default False
        If True, return a dict of DataFrames keyed by cut name instead
        of the final DataFrame.
    check_preprocessed : bool, default True
        If True, warn when no preprocessing fixes (``_fix_*`` columns)
        are detected on ``indf``. Suppress with ``check_preprocessed=False``
        for DataFrames where preprocessing is not applicable.

    Returns
    -------
    pandas.DataFrame or dict
        Final selected DataFrame, or per-stage dict when
        ``savedict=True`` or ``stage`` is set.
    """
    if cuts is None:
        from .analysis import DEFAULT_CUTS
        cuts = DEFAULT_CUTS

    if check_preprocessed:
        from .preprocess import applied_fixes
        if not applied_fixes(indf):
            warnings.warn(
                "No preprocessing fixes detected on this DataFrame. "
                "Call preprocess_mc() or preprocess_data() before select().",
                stacklevel=2,
            )

    if stage is not None and stage not in {c.name for c in cuts}:
        raise ValueError(
            f"Unknown stage '{stage}'. Valid options: {[c.name for c in cuts]}"
        )

    df = indf.copy()

    df_dict = {}
    for spec in cuts:
        df = df[_mask(df, spec)]
        if savedict:
            df_dict[spec.name] = df
        if stage == spec.name:
            return df_dict if savedict else df

    return df_dict if savedict else df


def select_sideband(indf, cuts=None, **kwargs):
    """Apply the sideband cut sequence. Accepts the same kwargs as ``select``."""
    if cuts is None:
        from .analysis import SIDEBAND_CUTS
        cuts = SIDEBAND_CUTS
    return select(indf, cuts=cuts, **kwargs)
