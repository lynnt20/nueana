"""Generic DataFrame utilities."""
import pandas as pd
import sys; sys.path.append("/exp/sbnd/app/users/lynnt/cafpyana")
from pyanalib.pandas_helpers import *

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
    nlevels = df.index.nlevels 
    hdr_cols = ['__ntuple','entry','run','subrun','evt']
    return multicol_merge(ensure_lexsorted(hdr_df.reset_index(),axis=1)[hdr_cols],
                          ensure_lexsorted(df.reset_index(),axis=1),
                          on = [tuple(['__ntuple'] + (nlevels-1)*['']),
                                tuple(['entry']    + (nlevels-1)*['']),]
                          )