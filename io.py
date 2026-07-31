"""File input/output utilities for loading HDF5 data files."""
from __future__ import annotations

import contextlib
import gc
import os
import pickle
import tempfile
import warnings

import numpy as np
import pandas as pd

from .classes import SystematicsOutput

__all__ = ['get_n_split', 'print_keys', 'load_dfs', 'load_mc', 'load_data',
           'merge_beam_spills',
           'subset_to_pot', 'filter_by_runs',
           'save_signal_checkpoint', 'load_signal_checkpoint',
           'save_sideband_checkpoint', 'load_sideband_checkpoint']

# ---------------------------------------------------------------------------
# pnfs / XRootD helpers
# ---------------------------------------------------------------------------

_XROOTD_PREFIX = "root://fndcadoor.fnal.gov:1094//pnfs/fnal.gov/usr"


def _pnfs_to_xrootd(path: str) -> str:
    """Convert /pnfs POSIX path to an XRootD URL for streaming."""
    return _XROOTD_PREFIX + path[len("/pnfs"):]


def _chunk_copy(src, dst, chunk_size: int = 8 * 1024 * 1024) -> None:
    while True:
        chunk = src.read(chunk_size)
        if not chunk:
            break
        dst.write(chunk)


@contextlib.contextmanager
def _local_hdf(file: str):
    """Yield a local file path, streaming from pnfs via XRootD if needed.

    For /pnfs paths: converts to an XRootD URL, streams the file to a
    temporary local copy, yields that path, then deletes it on exit.
    For local paths: yields the path unchanged with no I/O.
    """
    if not file.startswith("/pnfs"):
        yield file
        return

    try:
        import fsspec
    except ImportError as exc:
        raise ImportError(
            "fsspec is required to read files from /pnfs. "
            "Install it with: pip install fsspec fsspec-xrootd"
        ) from exc

    url = _pnfs_to_xrootd(file)
    fd, tmp_path = tempfile.mkstemp(suffix=".h5")
    os.close(fd)
    try:
        print(f"Streaming from pnfs: {file}")
        with fsspec.open(url, "rb") as remote, open(tmp_path, "wb") as local:
            _chunk_copy(remote, local)
        yield tmp_path
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _get_n_split_local(local_path: str) -> int:
    """Read split count from a local HDF5 file (no pnfs resolution)."""
    return int(pd.read_hdf(local_path, key="split").n_split.iloc[0])


# credit for first three functions to Mun!
def get_n_split(file):
    """Get the number of splits in an HDF5 file.

    Parameters
    ----------
    file : str
        Path to HDF5 file. Accepts /pnfs paths (streamed via XRootD).

    Returns
    -------
    int
        Number of splits in the file.
    """
    with _local_hdf(file) as local_path:
        return _get_n_split_local(local_path)

def print_keys(file):
    """Print all keys available in an HDF5 file.

    Parameters
    ----------
    file : str
        Path to HDF5 file. Accepts /pnfs paths (streamed via XRootD).
    """
    with _local_hdf(file) as local_path:
        with pd.HDFStore(local_path, mode='r') as store:
            keys = store.keys()       # list of all keys in the file
            print("Keys:", keys)
        
def load_dfs(file, keys2load, n_max_concat=10, start_split=0):
    """Load DataFrames from split HDF5 file.

    Parameters
    ----------
    file : str
        Path to HDF5 file. Accepts /pnfs paths (streamed via XRootD).
    keys2load : list
        List of key names to load from the file.
    n_max_concat : int, optional
        Maximum number of splits to concatenate (default: 10).
    start_split : int, optional
        Starting split index to load from (default: 0).

    Returns
    -------
    dict
        Dictionary mapping key names to concatenated DataFrames.
    """
    with _local_hdf(file) as local_path:
        out_df_dict = {}
        this_n_keys = _get_n_split_local(local_path) - start_split
        n_concat = min(n_max_concat, this_n_keys)
        for key in keys2load:
            dfs = []  # collect all splits for this key
            for i in range(start_split, start_split + n_concat):
                this_df = pd.read_hdf(local_path, key=f"{key}_{i}")
                dfs.append(this_df)
            out_df_dict[key] = pd.concat(dfs, ignore_index=False)
        return out_df_dict


# ---------------------------------------------------------------------------
# High-level loaders
# ---------------------------------------------------------------------------

_DEFAULT_MC_KEYS   = ['hdr', 'nuecc', 'histpotdf', 'histgenevtdf']
_DEFAULT_DATA_KEYS = ['hdr', 'nuecc', 'histpotdf']

def load_mc(
    file: str,
    keys: list | None = None,
    cuts=None,
    max_splits: int | None = None,
    chunk_splits: int = 1,
    add_pi0: bool = False,
    excl_mc_df=None,
) -> tuple:
    """Load, preprocess, and optionally select an MC HDF5 file in chunks.

    Splits are loaded in batches of chunk_splits to balance memory and I/O
    overhead.  POT and generated-event counts are accumulated across all
    splits.  Header columns (run/subrun/event) are merged into the output
    DataFrame.

    Parameters
    ----------
    file : str
        Path to the HDF5 file.
    keys : list of str, optional
        Table keys to load.  Defaults to
        ``['hdr', 'nuecc', 'histpotdf', 'histgenevtdf']``.
    cuts : list of CutSpec, optional
        If supplied, passed to :func:`~nueana.selection.select`.
        When None the full preprocessed DataFrame is returned.
    max_splits : int, optional
        Cap on the number of splits to load.  Defaults to all splits.
    chunk_splits : int, default 1
        Number of splits to load per iteration.  Increase to reduce I/O
        overhead at the cost of higher peak memory per chunk.
    add_pi0 : bool, default False
        If True, compute pi0 kinematics via :func:`~nueana.preprocess.add_pi0`
        for each chunk after preprocessing.
    excl_mc_df : pd.DataFrame, optional
        Exclusive mcnuecc DataFrame with ``define_signal`` already applied
        (i.e. has a top-level ``signal`` column).  When provided,
        :func:`~nueana.exclusive.remove_signal_overlap` is called on the
        final concatenated result to strip events that are already covered
        by the exclusive sample and would otherwise be double-counted.

    Returns
    -------
    df : pd.DataFrame
        Concatenated, preprocessed (and optionally selected) MC DataFrame
        with header columns merged in and signal categories defined.
    pot : float
        Accumulated POT.
    ngen : float
        Accumulated generated-event count.
    """
    try:
        from tqdm import tqdm as _tqdm
    except ImportError:
        _tqdm = None

    from .preprocess import preprocess_mc, add_pi0 as _add_pi0
    from .selection import select
    from .analysis import define_signal
    from .utils import merge_hdr

    if keys is None:
        keys = _DEFAULT_MC_KEYS

    pot    = 0.0
    ngen   = 0.0
    chunks = []

    with _local_hdf(file) as local_file:
        n_total  = _get_n_split_local(local_file)
        n_splits = min(max_splits, n_total) if max_splits is not None else n_total
        starts   = range(0, n_splits, chunk_splits)
        iterator = _tqdm(starts) if _tqdm is not None else starts

        for i in iterator:
            n_load = min(chunk_splits, n_splits - i)
            dfs = load_dfs(local_file, keys2load=keys, n_max_concat=n_load, start_split=i)

            if 'histpotdf' in dfs:    pot  += dfs['histpotdf'].TotalPOT.sum()
            elif 'hdr' in dfs:        pot  += dfs['hdr'].pot.sum()
            if 'histgenevtdf' in dfs: ngen += dfs['histgenevtdf'].TotalGenEvents.sum()
            elif 'hdr' in dfs:        ngen += dfs['hdr'][dfs['hdr'].first_in_subrun == 1].ngenevt.sum()

            df    = preprocess_mc(dfs['nuecc'])
            sel   = select(df, cuts=cuts) if cuts is not None else df
            chunk = merge_hdr(dfs['hdr'], sel)
            del dfs
            chunk = define_signal(chunk, prefix=('slc', 'truth'))
            if add_pi0:
                chunk = _add_pi0(chunk)
            chunks.append(chunk)
            del chunk
            gc.collect()

    result = pd.concat(chunks, ignore_index=False).copy()
    if excl_mc_df is not None:
        from .exclusive import remove_signal_overlap
        result = remove_signal_overlap(result, excl_mc_df)
    return result, pot, ngen


def load_data(
    file: str,
    keys: list | None = None,
    onbeam: bool = True,
    cuts=None,
) -> tuple:
    """Load, preprocess, and optionally select a data HDF5 file.

    Parameters
    ----------
    file : str
        Path to the HDF5 file.
    keys : list of str, optional
        Table keys to load.  Defaults to ``['hdr', 'nuecc', 'histpotdf']``.
    onbeam : bool, default True
        True for on-beam (BNB) data; False for off-beam.  Controls which
        gate counter is returned and whether the offbeam signal category is
        stamped on the output DataFrame.
    cuts : list of CutSpec, optional
        If supplied, passed to :func:`~nueana.selection.select`.
        When None the full preprocessed DataFrame is returned.

    Returns
    -------
    df : pd.DataFrame
        Preprocessed (and optionally selected) data DataFrame with header
        columns merged in and pi0 kinematics added.  Off-beam DataFrames
        also have ``signal`` set to ``signal_dict['offbeam']``.
    pot : float
        Accumulated on-beam POT (0.0 for off-beam files).
    ngates : float
        BNB gate count (on-beam) or off-beam gate count.
    """
    from pyanalib.pandas_helpers import multicol_add
    from .preprocess import preprocess_data, add_pi0
    from .selection import select
    from .utils import merge_hdr
    from .analysis import signal_dict

    if keys is None:
        keys = _DEFAULT_DATA_KEYS

    with _local_hdf(file) as local_file:
        dfs = load_dfs(local_file, keys2load=keys)
    df  = merge_hdr(dfs['hdr'], dfs['nuecc'])
    df  = preprocess_data(df)

    pot    = 0.0
    ngates = 0.0
    if onbeam:
        pot    = dfs['histpotdf'].TotalPOT.sum() if 'histpotdf' in dfs else dfs['hdr'].pot.sum()
        ngates = dfs['hdr'].nbnbinfo.sum()
    else:
        ngates = dfs['hdr'].noffbeambnb.sum()
        signal = pd.Series(
            np.ones(len(df), dtype=np.int16) * signal_dict['offbeam'],
            name="signal", index=df.index,
        )
        df = multicol_add(df, signal)

    sel = select(df, cuts=cuts) if cuts is not None else df
    return sel, pot, ngates


def merge_beam_spills(
    dt_dfs: dict,
    fom_range: tuple = (0.98, 1.0),
    verbose: bool = True,
) -> tuple:
    """Annotate the trigger header with beam-spill monitoring columns.

    Merges ``dt_dfs['hdr']`` with ``dt_dfs['trigger']`` to obtain
    ``global_trigger_time``, then matches each trigger event to the most
    recent BNB spill in ``dt_dfs['pot']`` via :func:`numpy.searchsorted`.
    Matching is done per-ntuple first; events whose ntuple has no POT entry
    fall back to the nearest global spill.

    Parameters
    ----------
    dt_dfs : dict
        Dictionary of DataFrames as returned by :func:`load_dfs`, expected
        to contain keys ``'hdr'``, ``'trigger'``, and ``'pot'``.
    fom_range : tuple of (float, float), default (0.98, 1.0)
        Inclusive ``[lo, hi]`` range on the ``FOM`` column used when summing
        good-quality BNB POT from the spill table.
    verbose : bool, default True
        If True, print matching diagnostics and total POT to stdout.

    Returns
    -------
    hdr_df : pd.DataFrame
        Copy of ``dt_dfs['hdr']`` with beam columns
        ``['TOR875', 'TOR860', 'FOM', 'THCURR', 'spill_time']`` added.
    bnb_pot : float
        Sum of ``TOR875`` over spills whose ``FOM`` falls within *fom_range*.
    """
    _BEAM_COLS = ['TOR875', 'TOR860', 'FOM', 'THCURR', 'spill_time']

    hdr_df = pd.merge(
        dt_dfs['hdr'],
        dt_dfs['trigger'][['global_trigger_time']],
        left_index=True,
        right_index=True,
    )

    pot_df = dt_dfs['pot'].copy()
    pot_df['spill_time'] = pot_df['spill_time_sec'] * 1e9 + pot_df['spill_time_nsec']

    pot_reset = pot_df.reset_index()
    hdr_reset = hdr_df.reset_index()[['__ntuple', 'entry', 'file_idx', 'evt', 'global_trigger_time']]
    hdr_reset['_orig_idx'] = np.arange(len(hdr_reset))

    pot_by_ntuple = {
        ntuple: grp.sort_values('spill_time').reset_index(drop=True)
        for ntuple, grp in pot_reset.groupby('__ntuple')
    }
    all_spills = pot_reset.sort_values('spill_time').reset_index(drop=True)

    results = []
    n_global_fallback = 0
    fallback_rows = []

    for ntuple, hdr_grp in hdr_reset.groupby('__ntuple'):
        spills = pot_by_ntuple.get(ntuple)
        if spills is None:
            spills = all_spills
            n_global_fallback += len(hdr_grp)
            fallback_rows.append(hdr_grp[['__ntuple', 'entry', 'evt', 'global_trigger_time']])

        times = spills['spill_time'].values
        trig  = hdr_grp['global_trigger_time'].values
        idx   = np.clip(np.searchsorted(times, trig, side='right') - 1, 0, len(spills) - 1)

        matched = spills.iloc[idx][_BEAM_COLS].reset_index(drop=True)
        part    = pd.concat([hdr_grp[['_orig_idx']].reset_index(drop=True), matched], axis=1)
        results.append(part)

    matched_ordered = pd.concat(results, ignore_index=True).sort_values('_orig_idx')
    for col in _BEAM_COLS:
        hdr_df[col] = matched_ordered[col].values

    fom_lo, fom_hi = fom_range
    bnb_pot = pot_df[(pot_df.FOM >= fom_lo) & (pot_df.FOM <= fom_hi)].TOR875.sum()

    if verbose:
        n_matched = hdr_df['TOR875'].notna().sum()
        n_total   = len(hdr_df)
        if n_global_fallback:
            print(f"  {n_global_fallback} event(s) had no pot entry for their ntuple "
                  f"— matched to nearest global spill instead.")
            print(pd.concat(fallback_rows, ignore_index=True).to_string(index=False))
        print(f"Matched {n_matched} / {n_total} events to a beam spill.")
        failures = hdr_df[hdr_df['TOR875'].isna()]
        if len(failures) > 0:
            print(f"\n=== {len(failures)} UNMATCHED EVENT(S) ===")
            print(failures.reset_index()[['__ntuple', 'entry', 'evt', 'global_trigger_time']].to_string())
        else:
            print("All events matched successfully.")
        print(f"Total POT in pot table: {bnb_pot:.3e} (from {len(pot_df)} spills)")

    return hdr_df, bnb_pot


# ---------------------------------------------------------------------------
# POT subsetting
# ---------------------------------------------------------------------------

def subset_to_pot(hdr_df: pd.DataFrame, target_pot: float) -> tuple:
    """Select runs in ascending order until target_pot is reached.

    Runs are identified by ``first_in_subrun == 1`` rows in *hdr_df* and
    accumulated in ascending run-number order.  The loop stops as soon as the
    running total first meets or exceeds *target_pot*, so the returned POT may
    exceed *target_pot* by at most one run's worth.  If the file contains less
    than *target_pot* total, all runs are selected.

    Use :func:`filter_by_runs` to apply the returned *hdr_filtered* to any
    other DataFrame sharing the same index levels (raw *nuecc* from
    :func:`load_dfs`, or the post-processed event DataFrame from
    :func:`load_mc` / :func:`load_data`).

    Parameters
    ----------
    hdr_df : pd.DataFrame
        Header DataFrame with ``run``, ``pot``, and ``first_in_subrun``
        columns, indexed by ``(__ntuple, entry, ...)``.
    target_pot : float
        Target exposure in protons-on-target.

    Returns
    -------
    hdr_filtered : pd.DataFrame
        *hdr_df* rows restricted to the selected runs.
    actual_pot : float
        Cumulative POT of the selected runs.
    """
    run_pot = (
        hdr_df.loc[hdr_df['first_in_subrun'] == 1, ['run', 'pot']]
        .sort_values('run')
    )

    cumsum = 0.0
    selected: set[int] = set()
    for _, row in run_pot.iterrows():
        selected.add(int(row['run']))
        cumsum += row['pot']
        if cumsum >= target_pot:
            break

    if cumsum < target_pot:
        warnings.warn(
            f"subset_to_pot: file only contains {cumsum:.3e} POT but "
            f"target_pot={target_pot:.3e} — all runs were selected.",
            stacklevel=2,
        )
    return hdr_df[hdr_df['run'].astype(int).isin(selected)], cumsum


def filter_by_runs(df: pd.DataFrame, hdr_filtered: pd.DataFrame) -> pd.DataFrame:
    """Filter *df* to events whose run number appears in *hdr_filtered*.

    Designed to pair with :func:`subset_to_pot`.  Filters by the ``run``
    column, which is unambiguous across the hdr and nuecc table index
    structures.  Works for post-processed event DataFrames from
    :func:`load_mc` / :func:`load_data` (where ``run`` is a flat or
    top-level MultiIndex column after :func:`~nueana.utils.merge_hdr`).

    Parameters
    ----------
    df : pd.DataFrame
        Event DataFrame with a ``run`` column (flat or top-level MultiIndex).
    hdr_filtered : pd.DataFrame
        Filtered header returned by :func:`subset_to_pot`.

    Returns
    -------
    pd.DataFrame
        *df* restricted to runs present in *hdr_filtered*.
    """
    selected_runs = set(hdr_filtered['run'].astype(int))
    run_col = df['run']
    if isinstance(run_col, pd.DataFrame):
        run_col = run_col.iloc[:, 0]
    return df[run_col.astype(int).isin(selected_runs)]


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

def save_signal_checkpoint(
    path: str,
    reco_df: pd.DataFrame,
    true_df: pd.DataFrame,
    syst_total: dict[str, SystematicsOutput],
    syst_signal: dict[str, SystematicsOutput],
    syst_bkg: dict[str, SystematicsOutput],
) -> None:
    """Pickle signal region outputs needed for fake-data tests and CCBC.

    Parameters
    ----------
    path : str
        Destination file path (e.g. ``"signal_checkpoint.pkl"``).
    reco_df : pd.DataFrame
        Selected and preprocessed reco-level MC DataFrame.
    true_df : pd.DataFrame
        True-level signal DataFrame (used for response matrix / unfolding).
    syst_total : dict[str, SystematicsOutput]
        Total (signal + background) systematics keyed by ``var_save_name``
        (e.g. ``{"energy": ..., "direction": ...}``).
    syst_signal : dict[str, SystematicsOutput]
        Signal-only systematics, same keys as ``syst_total``.
    syst_bkg : dict[str, SystematicsOutput]
        Background-only systematics, same keys as ``syst_total``.
    """
    payload = {
        "reco_df":     reco_df,
        "true_df":     true_df,
        "syst_total":  syst_total,
        "syst_signal": syst_signal,
        "syst_bkg":    syst_bkg,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_signal_checkpoint(
    path: str,
) -> tuple[pd.DataFrame, pd.DataFrame,
           dict[str, SystematicsOutput],
           dict[str, SystematicsOutput],
           dict[str, SystematicsOutput]]:
    """Load a checkpoint written by :func:`save_signal_checkpoint`.

    Parameters
    ----------
    path : str
        Path to the pickle file.

    Returns
    -------
    reco_df : pd.DataFrame
    true_df : pd.DataFrame
    syst_total : dict[str, SystematicsOutput]
    syst_signal : dict[str, SystematicsOutput]
    syst_bkg : dict[str, SystematicsOutput]
    """
    with open(path, "rb") as f:
        ck = pickle.load(f)
    return ck["reco_df"], ck["true_df"], ck["syst_total"], ck["syst_signal"], ck["syst_bkg"]


def save_sideband_checkpoint(
    path: str,
    reco_df: pd.DataFrame,
    syst_total: dict[str, SystematicsOutput],
) -> None:
    """Pickle sideband region outputs.

    Parameters
    ----------
    path : str
        Destination file path (e.g. ``"sideband_checkpoint.pkl"``).
    reco_df : pd.DataFrame
        Selected and preprocessed reco-level DataFrame.
    syst_total : dict[str, SystematicsOutput]
        Total systematics keyed by ``var_save_name``
        (e.g. ``{"energy": ..., "direction": ...}``).
    """
    with open(path, "wb") as f:
        pickle.dump({"reco_df": reco_df, "syst_total": syst_total}, f)


def load_sideband_checkpoint(
    path: str,
) -> tuple[pd.DataFrame, dict[str, SystematicsOutput]]:
    """Load a checkpoint written by :func:`save_sideband_checkpoint`.

    Parameters
    ----------
    path : str
        Path to the pickle file.

    Returns
    -------
    reco_df : pd.DataFrame
    syst_total : dict[str, SystematicsOutput]
    """
    with open(path, "rb") as f:
        ck = pickle.load(f)
    return ck["reco_df"], ck["syst_total"]