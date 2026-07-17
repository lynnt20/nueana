"""
Detector variation (DetVar) HDF5 store: write and read helpers.

Storage layout
--------------
/meta                        small DataFrame: group, cv_key, n_dv, pot
/cv/{cv_key}                 full CV nuecc DataFrame, stored once per unique CV
/dv/{group}/0                DV nuecc DataFrame pre-filtered to the CV∩DV intersection
/dv/{group}/1                (second DV for multisim groups, same intersection)
/cv_iloc/{group}             int64 array: iloc positions in /cv/{cv_key} for this group

Matching is a two-step process
-------------------------------
Step 1 — external match (nulite):
    Each event is reduced to its highest-energy neutrino interaction, then matched
    on physics-level columns ('run','subrun','evt','E') across CV and DV files.

Step 2 — internal match (nuecc):
    The matched nulite events carry internal file-level indices (__ntuple, entry, file_idx)
    that select the corresponding rows in each file's nuecc DataFrame.

Usage
-----
Step 1 — prepare each raw .df file:

    cv1  = prepare_detvar_df('/path/to/detvar_cv.df')
    pmt  = prepare_detvar_df('/path/to/detvar_pmtgain.df')
    xw1  = prepare_detvar_df('/path/to/detvar_wiremod_xw1.df')
    xw2  = prepare_detvar_df('/path/to/detvar_wiremod_xw2.df')
    alph = prepare_detvar_df('/path/to/detvar_recomb_alpha.df')

Step 2 — write the store once (matching done internally):

    write_detvar_store('detvars.h5',
        cv_dict = {'cv_1': cv1, 'cv_2': cv2},
        dv_dict = {'pmtgain': pmt, 'wiremod': [xw1, xw2], 'recomb_alpha': alph},
        cv_map  = {'pmtgain': 'cv_1', 'wiremod': 'cv_1', 'recomb_alpha': 'cv_2'},
    )

Step 3 — load at analysis time:

    detvar_dict = load_detvar_dict('detvars.h5')
    # or a subset:
    detvar_dict = load_detvar_dict('detvars.h5', groups=['pmtgain', 'wiremod'])
    # each entry: {'dv_df': df_or_list, 'cv_df': df, 'pot': float}
"""
from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd
from collections import namedtuple
from ..io import load_dfs

__all__ = ['DetVarFile', 'prepare_detvar_df', 'write_detvar_store', 'load_detvar_dict', 'select_detvar_dict', 'subsample_detvar_dict', 'detvar_store_info', 'apply_selection']

_DEFAULT_EXT_MATCH_COL = ['run', 'subrun', 'evt', 'E']
_DEFAULT_INT_MATCH_COL = ['__ntuple', 'entry', 'file_idx']
_HDF_KW            = dict(format='fixed')


def _ensure_file_idx(df: pd.DataFrame, default: int = 0) -> pd.DataFrame:
    """Append file_idx=default as an index level if not already present."""
    if 'file_idx' not in df.index.names:
        df = df.copy()
        df['file_idx'] = default
        df = df.set_index('file_idx', append=True)
    return df

DetVarFile = namedtuple('DetVarFile', ['lite_df', 'slc_df'])
"""Named tuple returned by prepare_detvar_df.

Attributes
----------
lite_df : pd.DataFrame
    nulite DataFrame with ext_match_col set as the index.
    Used only for external event matching; not stored in the HDF5 file.
slc_df : pd.DataFrame
    nuecc DataFrame with its original MultiIndex.
    This is the analysis-level DataFrame that gets stored after filtering.
"""


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def prepare_detvar_df(
    file: str,
    ext_match_col: list | None = None,
    e_col: str = 'E',
    nulite_key: str = 'nulite',
    slc_key: str = 'nuecc',
) -> DetVarFile:
    """Load a detvar .df file and prepare it for the two-step matching.

    Reduces the nulite table to one row per event (the highest-energy neutrino
    interaction), reindexes on ``ext_match_col``, and returns it alongside the
    raw nuecc table. The intersection and internal filtering happen later in
    :func:`write_detvar_store`.

    If ``file_idx`` is absent from either DataFrame's index it is added with
    a default value of 0.

    Parameters
    ----------
    file : str
        Path to the HDF5 ``.df`` file (CV or DV).
    ext_match_col : list of str, optional
        Physics-level columns used for cross-file event matching (external match).
        Defaults to ``['run', 'subrun', 'evt', 'E']``.
    e_col : str, optional
        Top-level column name for neutrino energy (default ``'E'``).
    nulite_key : str, optional
        Table key for the light neutrino table (default ``'nulite'``).
    slc_key : str, optional
        Table key for the slice / nuecc table (default ``'nuecc'``).

    Returns
    -------
    DetVarFile
        ``(lite_df, slc_df)`` named tuple where ``lite_df`` is the nulite
        DataFrame filtered to one row per event with ``ext_match_col`` as its
        index, and ``slc_df`` is the raw nuecc DataFrame.
    """
    if ext_match_col is None:
        ext_match_col = _DEFAULT_EXT_MATCH_COL

    dfs = load_dfs(file, [nulite_key, slc_key])

    lite_df = _ensure_file_idx(dfs[nulite_key])
    slc_df  = _ensure_file_idx(dfs[slc_key])

    emax_names = [n for n in lite_df.index.names if n in _DEFAULT_INT_MATCH_COL]
    e_max_per_event = lite_df.groupby(level=emax_names)[e_col].transform('max')
    lite_df = lite_df[lite_df[e_col] == e_max_per_event]
    lite_df = lite_df.reset_index().set_index(ext_match_col)

    return DetVarFile(lite_df=lite_df, slc_df=slc_df)


# ---------------------------------------------------------------------------
# Internal helpers (not exported)
# ---------------------------------------------------------------------------

def _add_rse_cols(slc_df: pd.DataFrame, lite_df: pd.DataFrame) -> pd.DataFrame:
    """Append run, subrun, evt to slc_df as MultiIndex columns.

    Looks up RSE values from ``lite_df`` via the internal match columns
    (``__ntuple``, ``entry``, ``file_idx``), broadcasting across all slices
    that share the same event.
    """
    n   = slc_df.columns.nlevels
    pad = ('',) * (n - 1)

    rse = (lite_df.reset_index()
                  [_DEFAULT_INT_MATCH_COL + ['run', 'subrun', 'evt']]
                  .drop_duplicates(_DEFAULT_INT_MATCH_COL)
                  .set_index(_DEFAULT_INT_MATCH_COL))

    # Reindex slc by int_match to align RSE across all slices per event
    slc_by_int  = slc_df.reset_index().set_index(_DEFAULT_INT_MATCH_COL)
    rse_aligned = rse.reindex(slc_by_int.index)

    rse_aligned.index   = slc_df.index
    rse_aligned.columns = pd.MultiIndex.from_tuples([(c,) + pad for c in rse_aligned.columns])

    return pd.concat([slc_df, rse_aligned], axis=1)


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def write_detvar_store(
    outfile: str,
    cv_dict: dict,
    dv_dict: dict,
    cv_map: dict,
    mode: str = 'w',
) -> None:
    """Write a DetVar HDF5 store.

    For each DV group, performs the two-step matching:
    1. External: intersect CV and DV nulite indices to find common events.
    2. Internal: use the matched nulite indices to filter the nuecc DataFrames.

    The CV nuecc DataFrame is stored once per unique CV key; a per-group iloc
    array is stored so that :func:`load_detvar_dict` can reconstruct the
    matched CV subset without re-doing the matching.

    POT is derived automatically from the ``pot`` column of the matched CV
    nulite DataFrame (``sum`` over matched events).

    Parameters
    ----------
    outfile : str
        Output HDF5 file path.
    cv_dict : dict[str, DetVarFile]
        CV files keyed by an arbitrary name.
        Each value must be a :class:`DetVarFile` from :func:`prepare_detvar_df`.
    dv_dict : dict[str, DetVarFile | list[DetVarFile]]
        DV files keyed by group name.  Pass a single :class:`DetVarFile` for
        unisim variations or a list for multisim groups.
    cv_map : dict[str, str]
        Maps each group name in ``dv_dict`` to a key in ``cv_dict``.
    mode : {'w', 'a'}, default 'w'
        ``'w'`` creates or fully overwrites the store.
        ``'a'`` patches only the groups in ``dv_dict`` into an existing store,
        leaving all other groups untouched.  Falls back to ``'w'`` if the file
        does not exist yet.  Raises ``ValueError`` if a CV being rewritten is
        also depended on by groups that are *not* in ``dv_dict`` (those groups'
        iloc indices would become invalid); include all affected groups in the
        call to avoid this.

    Notes
    -----
    pandas ``format='fixed'`` does not support in-file compression. To reduce
    disk usage after writing, run ``h5repack -f GZIP=5 detvars.h5 detvars_compressed.h5``
    (requires HDF5 tools) or ``ptrepack --complevel=5 detvars.h5 out.h5``
    (requires PyTables).
    """
    _validate_inputs(cv_dict, dv_dict, cv_map)

    if mode == 'a' and not os.path.exists(outfile):
        print(f"  (store not found; writing fresh)")
        mode = 'w'

    kw = _HDF_KW

    meta_rows    = []
    written_cvs: set[str] = set()
    preserved_meta: pd.DataFrame | None = None

    if mode == 'a':
        with pd.HDFStore(outfile, mode='r') as _store:
            existing_meta = _store['meta'] if 'meta' in _store else pd.DataFrame(
                columns=['cv_key', 'n_dv', 'pot'], dtype=object
            )
            existing_meta.index.name = 'group'

            # Safety: if a CV is being overwritten, all groups using it must be in dv_dict
            cvs_to_write = {cv_map[g] for g in dv_dict}
            for cv_key in cvs_to_write:
                if f'cv/{cv_key}' in _store:
                    dependent = existing_meta[existing_meta['cv_key'] == cv_key].index.tolist()
                    stale = set(dependent) - set(dv_dict)
                    if stale:
                        raise ValueError(
                            f"CV '{cv_key}' must be rewritten but groups {sorted(stale)} in "
                            f"the existing store also depend on it — their iloc indices would "
                            f"become invalid.  Add them to --groups to keep the store consistent."
                        )

        preserved_meta = existing_meta[~existing_meta.index.isin(dv_dict)]

        # Remove stale keys for groups about to be overwritten
        with pd.HDFStore(outfile, mode='a') as _store:
            for group in dv_dict:
                for key in [f'cv_iloc/{group}', f'dv/{group}/v0', f'dv/{group}/v1']:
                    if key in _store:
                        _store.remove(key)
            for cv_key in cvs_to_write:
                if f'cv/{cv_key}' in _store:
                    _store.remove(f'cv/{cv_key}')
            if 'meta' in _store:
                _store.remove('meta')

    with warnings.catch_warnings(), pd.HDFStore(outfile, mode=mode) as store:
        warnings.filterwarnings('ignore', '.*not a valid Python identifier.*')
        warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
        for group, dv_entry in dv_dict.items():
            dvs    = dv_entry if isinstance(dv_entry, list) else [dv_entry]
            cv_key = cv_map[group]
            cv     = cv_dict[cv_key]

            # --- Step 1: external intersection via nulite indices ---
            common_idx = cv.lite_df.index
            for dv in dvs:
                common_idx = common_idx.intersection(dv.lite_df.index)

            if len(common_idx) == 0:
                raise ValueError(
                    f"Group '{group}': empty intersection between CV '{cv_key}' "
                    f"and its DV(s). Check that match columns are consistent."
                )

            n_cv  = len(cv.lite_df)
            n_dv0 = len(dvs[0].lite_df)
            n_com = len(common_idx)
            print(
                f"  {group}: {n_com} common events "
                f"({n_com/n_dv0*100:.1f}% of DV, {n_com/n_cv*100:.1f}% of CV '{cv_key}')"
            )

            # --- Step 2a: filter CV nuecc via internal indices ---
            cv_lite_matched = cv.lite_df.loc[common_idx]
            pot             = float(cv_lite_matched['pot'].to_numpy().sum())
            cv_int_idx      = (cv_lite_matched.reset_index()
                                              .set_index(_DEFAULT_INT_MATCH_COL)
                                              .index.unique())
            cv_slc          = cv.slc_df
            cv_slc_reindexed = cv_slc.reset_index().set_index(_DEFAULT_INT_MATCH_COL)
            cv_iloc          = np.where(cv_slc_reindexed.index.isin(cv_int_idx))[0]

            store.put(f'cv_iloc/{group}', pd.Series(cv_iloc.astype(np.int64)), **kw)

            # Write the full CV nuecc once per cv_key (iloc handles per-group slicing)
            if cv_key not in written_cvs:
                store.put(f'cv/{cv_key}', _add_rse_cols(cv_slc, cv.lite_df), **kw)
                written_cvs.add(cv_key)

            # --- Step 2b: filter each DV nuecc via internal indices ---
            for i, dv in enumerate(dvs):
                dv_lite_matched = dv.lite_df.loc[common_idx]
                dv_int_idx      = (dv_lite_matched.reset_index()
                                                  .set_index(_DEFAULT_INT_MATCH_COL)
                                                  .index.unique())
                dv_slc       = dv.slc_df
                slc_names    = dv_slc.index.names
                dv_slc_matched = (dv_slc.reset_index()
                                        .set_index(_DEFAULT_INT_MATCH_COL)
                                        .loc[lambda df: df.index.isin(dv_int_idx)]
                                        .reset_index()
                                        .set_index(slc_names))
                store.put(f'dv/{group}/v{i}', _add_rse_cols(dv_slc_matched, dv_lite_matched), **kw)

            meta_rows.append({
                'group':  group,
                'cv_key': cv_key,
                'n_dv':   len(dvs),
                'pot':    pot,
            })

        meta = pd.DataFrame(meta_rows).set_index('group')
        if preserved_meta is not None:
            meta = pd.concat([preserved_meta, meta])
        store.put('meta', meta, **kw)

    print(f"Wrote {outfile}")


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def load_detvar_dict(
    h5file: str,
    groups: list | None = None,
    preprocess_fn=None,
) -> dict:
    """Load a DetVar HDF5 store into a dict compatible with get_detvar_systs.

    Each loaded DataFrame is preprocessed with *preprocess_fn* before being
    returned. This mirrors the preprocessing applied to the main MC DataFrames
    (flash PE scaling, phi angles, etc.) so that selection cuts behave
    identically for DV/CV and nominal MC.

    Parameters
    ----------
    h5file : str
        Path to an HDF5 file written by :func:`write_detvar_store`.
    groups : list of str, optional
        Subset of group names to load. If None, all groups are loaded.
    preprocess_fn : callable or None, optional
        Function applied to each loaded DataFrame before it is stored in the
        output dict.  Signature: ``fn(df) -> df``.  Defaults to ``None``
        (no preprocessing), since stores written by :func:`process_detvars`
        are already preprocessed at write time.  Pass an explicit callable
        to apply additional transforms on load.

    Returns
    -------
    dict
        Maps each group name to ``{'dv_df': df_or_list, 'cv_df': df, 'pot': float}``.
    """
    _preprocess_label = None
    if preprocess_fn is not None:
        _preprocess_label = getattr(preprocess_fn, '__name__', repr(preprocess_fn))

    meta = pd.read_hdf(h5file, 'meta')

    if groups is not None:
        missing = set(groups) - set(meta.index)
        if missing:
            raise KeyError(f"Groups not found in store: {missing}")
        meta = meta.loc[groups]

    cv_cache: dict[str, pd.DataFrame] = {}
    out: dict = {}

    with warnings.catch_warnings(), pd.HDFStore(h5file, mode='r') as store:
        warnings.filterwarnings('ignore', '.*not a valid Python identifier.*')
        warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
        for group, row in meta.iterrows():
            cv_key = row['cv_key']
            if cv_key not in cv_cache:
                cv_df = store[f'cv/{cv_key}']
                cv_cache[cv_key] = preprocess_fn(cv_df) if preprocess_fn is not None else cv_df
            cv_full = cv_cache[cv_key]

            cv_iloc    = store[f'cv_iloc/{group}'].values
            cv_matched = cv_full.iloc[cv_iloc]

            n_dv   = int(row['n_dv'])
            dv_dfs = [
                preprocess_fn(store[f'dv/{group}/v{i}']) if preprocess_fn is not None
                else store[f'dv/{group}/v{i}']
                for i in range(n_dv)
            ]

            out[group] = {
                'dv_df': dv_dfs if n_dv > 1 else dv_dfs[0],
                'cv_df': cv_matched,
                'pot':   float(row['pot']),
            }

    preprocess_str = _preprocess_label if _preprocess_label is not None else "none"
    print(f"Loaded {len(out)} detvar group(s) from {h5file}  [preprocess: {preprocess_str}]")
    print(f"  Keys: {list(out.keys())}")

    _col_warnings = []
    for group, entry in out.items():
        cv_cols  = set(entry['cv_df'].columns.tolist())
        dv_list  = entry['dv_df'] if isinstance(entry['dv_df'], list) else [entry['dv_df']]
        for i, dv in enumerate(dv_list):
            dv_cols  = set(dv.columns.tolist())
            only_cv  = cv_cols - dv_cols
            only_dv  = dv_cols - cv_cols
            if only_cv or only_dv:
                suffix = f"[v{i}]" if len(dv_list) > 1 else ""
                if only_cv:
                    _col_warnings.append(f"  {group}{suffix}: columns only in CV:  {sorted(only_cv)}")
                if only_dv:
                    _col_warnings.append(f"  {group}{suffix}: columns only in DV:  {sorted(only_dv)}")

    if _col_warnings:
        print("  Column inconsistencies found:")
        for w in _col_warnings:
            print(w)
    else:
        print("  Column check: OK (all DV/CV column sets match)")

    return out


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------

def select_detvar_dict(detvar_dict: dict, cuts, in_place: bool = True) -> dict:
    """Apply a cut sequence to every CV/DV frame in a loaded detvar dict.

    Use this to shrink an already-loaded ``detvar_dict`` (the output of
    :func:`load_detvar_dict`) so that subsequent ``get_total_cov(cuts=...)``
    calls do not re-run :func:`~nueana.selection.select` on the full-size
    frames. Useful when iterating on plots at a looser preselection.

    Parameters
    ----------
    detvar_dict : dict
        Output of :func:`load_detvar_dict` — maps group name to
        ``{'cv_df', 'dv_df', 'pot'}``. ``dv_df`` may be a DataFrame or a list
        of DataFrames (multisim).
    cuts : list of CutSpec
        Cut sequence forwarded to :func:`~nueana.selection.select`.
    in_place : bool, default True
        If True, mutate the input dict and return it; if False, return a new
        dict with shallow-copied entries and replaced frames.

    Returns
    -------
    dict
        ``detvar_dict`` with each ``cv_df`` / ``dv_df`` replaced by its
        selected version.
    """
    from ..selection import select

    out = detvar_dict if in_place else {}
    for key, entry in detvar_dict.items():
        cv = select(entry['cv_df'], cuts=cuts, check_preprocessed=False)
        dv = entry['dv_df']
        if isinstance(dv, list):
            dv = [select(d, cuts=cuts, check_preprocessed=False) for d in dv]
        else:
            dv = select(dv, cuts=cuts, check_preprocessed=False)
        if in_place:
            entry['cv_df'] = cv
            entry['dv_df'] = dv
        else:
            out[key] = {**entry, 'cv_df': cv, 'dv_df': dv}
    return out


def _event_key_series(df: pd.DataFrame) -> pd.Series:
    """Return a per-row ``(run, subrun, evt)`` event identifier as a Series of tuples.

    The CV/DV slc frames are stored with ``run``, ``subrun``, ``evt`` columns
    added by :func:`_add_rse_cols` regardless of MultiIndex depth, so they are
    addressable by top-level name.
    """
    def _top(df, name):
        col = df[name]
        return col.iloc[:, 0] if isinstance(col, pd.DataFrame) else col

    run    = _top(df, 'run').to_numpy()
    subrun = _top(df, 'subrun').to_numpy()
    evt    = _top(df, 'evt').to_numpy()
    return pd.Series(list(zip(run, subrun, evt)), index=df.index)


def subsample_detvar_dict(detvar_dict: dict, frac: float, seed: int | None = None,
                          in_place: bool = True) -> dict:
    """Randomly subsample CV/DV frames at the event level and rescale POT.

    Selects a random fraction of ``(run, subrun, evt)`` tuples shared by CV and
    every DV in the group, then keeps every slice belonging to those events on
    both sides. CV and DV slice rows are not 1-to-1 aligned (detector variations
    can produce a different number of reco slices per event), so subsampling
    must be done by event identifier rather than by row position — this is the
    same notion of "same events" used at write time (see
    :func:`write_detvar_store`, which intersects via the nulite event index).

    ``pot`` is multiplied by ``frac`` so the predicted rate is preserved in
    expectation; only the per-bin statistical noise on the detvar covariance
    grows (as ~1/sqrt(frac)).

    Use during memory-constrained iteration when detvar statistical noise is
    not the limiting uncertainty.

    Parameters
    ----------
    detvar_dict : dict
        Output of :func:`load_detvar_dict` — maps group name to
        ``{'cv_df', 'dv_df', 'pot'}``. Frames must still carry
        ``run``/``subrun``/``evt`` columns (call this *before* any column
        trimming or aggregation that would drop them).
    frac : float
        Fraction of events to keep, in (0, 1].
    seed : int, optional
        Seed for reproducibility.
    in_place : bool, default True
        If True, mutate the input dict and return it; otherwise return a new dict.

    Returns
    -------
    dict
        ``detvar_dict`` with each ``cv_df`` / ``dv_df`` restricted to slices
        from the sampled events and ``pot`` rescaled by ``frac``.
    """
    if not 0 < frac <= 1:
        raise ValueError(f"frac must be in (0, 1], got {frac}")

    rng = np.random.default_rng(seed)
    out = detvar_dict if in_place else {}

    for key, entry in detvar_dict.items():
        cv      = entry['cv_df']
        dv      = entry['dv_df']
        dv_list = dv if isinstance(dv, list) else [dv]

        # Build per-frame event-id series; intersect with DV(s) to get the
        # event pool actually shared on both sides (write_detvar_store
        # already enforces this, but we re-check defensively).
        cv_evts = _event_key_series(cv)
        common  = set(cv_evts.unique())
        for d in dv_list:
            common &= set(_event_key_series(d).unique())
        common_list = sorted(common)

        n_keep = int(round(frac * len(common_list)))
        if n_keep == 0 and len(common_list) > 0:
            n_keep = 1  # never empty out a group on purpose
        idx     = rng.choice(len(common_list), size=n_keep, replace=False)
        sampled = {common_list[i] for i in idx}

        cv_sub      = cv[cv_evts.isin(sampled)]
        dv_sub_list = [d[_event_key_series(d).isin(sampled)] for d in dv_list]
        new_dv      = dv_sub_list if isinstance(dv, list) else dv_sub_list[0]
        new_pot     = entry['pot'] * frac

        if in_place:
            entry['cv_df'] = cv_sub
            entry['dv_df'] = new_dv
            entry['pot']   = new_pot
        else:
            out[key] = {**entry, 'cv_df': cv_sub, 'dv_df': new_dv, 'pot': new_pot}

    return out


def detvar_store_info(h5file: str) -> pd.DataFrame:
    """Return the metadata table from a DetVar store.

    Parameters
    ----------
    h5file : str
        Path to an HDF5 file written by :func:`write_detvar_store`.

    Returns
    -------
    pd.DataFrame
        Index: group name. Columns: cv_key, n_dv, pot.
    """
    return pd.read_hdf(h5file, 'meta')


def apply_selection(d: dict, fn, **kwargs) -> dict:
    """Apply a selection function to every slc_df in a cv_dict or dv_dict.

    Handles both single-DetVarFile entries and list entries (multisim groups).
    Extra keyword arguments are forwarded to ``fn`` on every call.

    Parameters
    ----------
    d : dict
        A ``cv_dict`` or ``dv_dict`` as passed to :func:`write_detvar_store`.
    fn : callable
        Selection function ``fn(df, **kwargs) -> DataFrame``.
    **kwargs
        Forwarded to ``fn`` on every call, e.g. ``stage='shower_energy'``.

    Returns
    -------
    dict
        Same structure as ``d`` with ``slc_df`` replaced by ``fn(slc_df, **kwargs)``
        on every entry.

    Examples
    --------
        cv_signal = apply_selection(cv_dict, nue.select)
        dv_signal = apply_selection(dv_dict, nue.select)
        write_detvar_store('detvars_signal.h5', cv_signal, dv_signal, cv_map)

        # with extra keyword arguments:
        cv_pre = apply_selection(cv_dict, nue.select, stage='shower_energy')
        dv_pre = apply_selection(dv_dict, nue.select, stage='shower_energy')
    """
    out = {}
    for name, entry in d.items():
        if isinstance(entry, list):
            out[name] = [e._replace(slc_df=fn(e.slc_df, **kwargs)) for e in entry]
        else:
            out[name] = entry._replace(slc_df=fn(entry.slc_df, **kwargs))
    return out


def _validate_inputs(cv_dict, dv_dict, cv_map):
    for key, val in cv_dict.items():
        if not isinstance(val, DetVarFile):
            raise TypeError(f"cv_dict['{key}'] must be a DetVarFile (got {type(val).__name__}). "
                            "Use prepare_detvar_df() to create it.")
    for key, val in dv_dict.items():
        entries = val if isinstance(val, list) else [val]
        for e in entries:
            if not isinstance(e, DetVarFile):
                raise TypeError(f"dv_dict['{key}'] must be a DetVarFile or list of DetVarFile "
                                f"(got {type(e).__name__}). Use prepare_detvar_df() to create it.")

    missing_cv = {g: cv_map[g] for g in cv_map if cv_map[g] not in cv_dict}
    if missing_cv:
        raise KeyError(f"cv_map references CV keys not in cv_dict: {missing_cv}")

    missing_map = set(dv_dict) - set(cv_map)
    if missing_map:
        raise KeyError(f"Groups in dv_dict have no entry in cv_map: {missing_map}")
