"""Generic DataFrame utilities."""
import pandas as pd
from pyanalib.pandas_helpers import *

from . import config

def ensure_lexsorted(frame, axis):
    """Ensure DataFrame axes are fully lexsorted when using MultiIndex.
    
    This avoids pandas PerformanceWarning about indexing past lexsort depth.
    
    Parameters
    ----------
    frame : pandas.DataFrame
        DataFrame to check and sort if needed.
    axis : int
        Axis to check (0 for index, 1 for columns).
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with sorted index/columns if MultiIndex, otherwise unchanged.
    """
    # axis: 0 -> index, 1 -> columns
    idx = frame.index if axis == 0 else frame.columns
    if isinstance(idx, pd.MultiIndex) and getattr(idx, "lexsort_depth", 0) < idx.nlevels:
        # sort by all levels (returns a new frame)
        return frame.sort_index(axis=axis)
    return frame

def merge_hdr(hdr_df,df):
    """Merge header DataFrame with main DataFrame on entry and __ntuple.
    
    Parameters
    ----------
    hdr_df : pandas.DataFrame
        DataFrame containing header information with columns including '__ntuple' and 'entry'.
    df : pandas.DataFrame
        Main DataFrame containing event data with columns including '__ntuple' and 'entry'.
    Returns
    -------
    pandas.DataFrame
        Merged DataFrame containing all columns from both hdr_df and df, merged on '__ntuple' and 'entry'.
    Notes
    -----
    - The merge is performed on the columns '__ntuple' and 'entry', which are expected to be present in both DataFrames.
    - The function ensures that both DataFrames are lexsorted on the relevant columns before merging to avoid performance issues with MultiIndex.
    """
    hdr_cols = ['__ntuple','entry','run','subrun','evt']

    hdr_merge_df = hdr_df.reset_index()[hdr_cols]
    evt_merge_df = df.reset_index()

    # Ensure both row and column MultiIndex objects are fully lexsorted before merge.
    hdr_merge_df = ensure_lexsorted(ensure_lexsorted(hdr_merge_df, axis=0), axis=1)
    evt_merge_df = ensure_lexsorted(ensure_lexsorted(evt_merge_df, axis=0), axis=1)

    # Build fully padded tuple keys for exact MultiIndex column matches.
    key_depth = max(hdr_merge_df.columns.nlevels, evt_merge_df.columns.nlevels)
    ntuple_key = tuple(['__ntuple'] + [''] * (key_depth - 1))
    entry_key = tuple(['entry'] + [''] * (key_depth - 1))

    merged_df = multicol_merge(hdr_merge_df,
                               evt_merge_df,
                         on = [ntuple_key, entry_key]
                               )
    return ensure_lexsorted(ensure_lexsorted(merged_df, axis=0), axis=1)

def apply_event_mask(df: pd.DataFrame, event_mask: str | None = None) -> pd.DataFrame:
    """ Apply event mask filter to DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a 'signal' column.
    event_mask : str or None
        Event classification filter: 'all', 'signal', or 'background'.
        If None (default), returns all events.
        
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame based on the event mask.
        - 'signal': events where signal == 0
        - 'background': events where signal != 0
        - 'all' or None: all events
        
    Raises
    ------
    ValueError
        If event_mask is not one of the allowed values.
    """
    # Normalize: convert None to "all" and validate
    if event_mask is None:
        event_mask = "all"
    if event_mask not in {"all", "signal", "background"}:
        raise ValueError("event_mask must be one of: 'all', 'signal', 'background', or None")
    
    # Apply: filter based on signal column (0 = signal, nonzero = background)
    if event_mask == "signal":
        return df[df.signal == 0]
    if event_mask == "background":
        return df[df.signal != 0]
    return df