"""File input/output utilities for loading HDF5 data files."""
from __future__ import annotations

import gc
import pickle

import numpy as np
import pandas as pd

from .classes import SystematicsOutput

__all__ = ['get_n_split', 'print_keys', 'load_dfs', 'load_mc', 'load_data',
           'save_signal_checkpoint', 'load_signal_checkpoint',
           'save_sideband_checkpoint', 'load_sideband_checkpoint']

# credit for first three functions to Mun! 
def get_n_split(file):
    """Get the number of splits in an HDF5 file.
    
    Parameters
    ----------
    file : str
        Path to HDF5 file.
    
    Returns
    -------
    int
        Number of splits in the file.
    """
    this_split_df = pd.read_hdf(file, key="split")
    this_n_split = this_split_df.n_split.iloc[0]
    return this_n_split

def print_keys(file):
    """Print all keys available in an HDF5 file.
    
    Parameters
    ----------
    file : str
        Path to HDF5 file.
    """
    with pd.HDFStore(file, mode='r') as store:
        keys = store.keys()       # list of all keys in the file
        print("Keys:", keys)
        
def load_dfs(file, keys2load, n_max_concat=10, start_split=0):
    """Load DataFrames from split HDF5 file.
    
    Parameters
    ----------
    file : str
        Path to HDF5 file.
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
    out_df_dict = {}
    this_n_keys = get_n_split(file) - start_split
    n_concat = min(n_max_concat, this_n_keys)
    for key in keys2load:
        dfs = []  # collect all splits for this key
        for i in range(start_split, start_split + n_concat):
            this_df = pd.read_hdf(file, key=f"{key}_{i}")
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

    n_total  = get_n_split(file)
    n_splits = min(max_splits, n_total) if max_splits is not None else n_total
    starts   = range(0, n_splits, chunk_splits)
    iterator = _tqdm(starts) if _tqdm is not None else starts

    pot    = 0.0
    ngen   = 0.0
    chunks = []

    for i in iterator:
        n_load = min(chunk_splits, n_splits - i)
        dfs = load_dfs(file, keys2load=keys, n_max_concat=n_load, start_split=i)

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

    dfs = load_dfs(file, keys2load=keys)
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