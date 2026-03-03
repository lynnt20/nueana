"""Generic DataFrame utilities."""
import pandas as pd

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
