"""Generic DataFrame and histogram utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pyanalib.pandas_helpers import *

from . import config

__all__ = ['ensure_lexsorted', 'merge_hdr', 'apply_event_mask', 'digitize_with_overflow', 'get_hist1d', 'get_hist2d', 'bin_geometry']

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

def merge_hdr(hdr_df, df):
    """Add header columns (run/subrun/evt) to main DataFrame by index join.

    Performs a left join on the shared index, so hdr_df values are broadcast
    to all matching rows in df (handles the many-slices-per-event case).
    More memory-efficient than the original multicol_merge: no reset_index
    copies, no merge-key hashing on flat columns.

    Parameters
    ----------
    hdr_df : pandas.DataFrame
        Header DataFrame with run, subrun, evt (and optionally file_idx) columns,
        indexed by (__ntuple, entry).
    df : pandas.DataFrame
        Main event DataFrame indexed by (__ntuple, entry), possibly with
        multiple rows per event (slices).

    Returns
    -------
    pandas.DataFrame
        df with run/subrun/evt (and file_idx if present) columns appended.
    """
    add_cols = ['run', 'subrun', 'evt']
    if 'file_idx' in hdr_df.columns:
        add_cols.append('file_idx')

    hdr_subset = hdr_df[add_cols]
    col_depth = df.columns.nlevels
    if col_depth > 1:
        hdr_subset = hdr_subset.copy()
        hdr_subset.columns = pd.MultiIndex.from_tuples(
            [tuple([c] + [''] * (col_depth - 1)) for c in add_cols]
        )

    result = df.join(hdr_subset)
    return ensure_lexsorted(ensure_lexsorted(result, axis=0), axis=1)

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
        raise ValueError(
            f"event_mask={event_mask!r} is not valid. Choose from: 'all', 'signal', 'background', or None"
        )

    # Apply: filter based on signal column (0 = signal, nonzero = background)
    if event_mask == "signal":
        return df[df.signal == 0]
    if event_mask == "background":
        return df[df.signal != 0]
    return df


# ---------------------------------------------------------------------------
# Histogram utilities
# ---------------------------------------------------------------------------

def digitize_with_overflow(data, bins):
    """Return 0-based bin indices with nan/inf/overflow clipped into edge bins.

    Values below ``bins[0]`` map to index 0; values at or above ``bins[-1]``
    map to index ``len(bins)-2`` (the last bin).  NaN and ±inf are treated as
    edge-bin values rather than raising or producing out-of-range indices.

    Parameters
    ----------
    data : array-like
        Values to digitize.
    bins : np.ndarray
        Monotonically increasing bin edges.

    Returns
    -------
    np.ndarray of int
        0-based bin indices, shape ``(len(data),)``.
    """
    a = np.asarray(data, dtype=float)
    a = np.nan_to_num(a, nan=bins[-1]-1e-10, posinf=bins[-1]-1e-10, neginf=bins[0])
    n_bins = len(bins) - 1
    return np.clip(np.searchsorted(bins, np.clip(a, bins[0], bins[-1]-1e-10), side='right') - 1,
                   0, n_bins - 1)



def get_hist1d(weights=None, data=None, bins=None, overflow=True, **kwargs):
    """1D histogram with optional overflow handling.

    Parameters
    ----------
    weights : np.ndarray, optional
        Per-event weights. If None, uses uniform weights of 1.0 for all events.
    data : np.ndarray
        Data values to histogram.
    bins : np.ndarray
        Bin edges.
    overflow : bool, optional
        If True (default), values above bins[-1] are clipped into the last bin.
        Non-finite values are assigned to edge bins. If False, standard numpy
        behavior with no clipping.
    **kwargs
        Passed to np.histogram().

    Returns
    -------
    np.ndarray
        Histogram counts of shape (len(bins)-1,).
    """
    if weights is None:
        weights = np.ones(len(data))
    if overflow:
        idx = digitize_with_overflow(data, bins)
        return np.bincount(idx, weights=weights, minlength=len(bins)-1).astype(float)
    else:
        return np.histogram(data, bins=bins, weights=weights, **kwargs)[0]


def get_hist2d(weights=None, x=None, y=None, bins=None, overflow=True, **kwargs):
    """2D histogram with optional overflow handling on both axes.

    Parameters
    ----------
    weights : np.ndarray, optional
        Per-event weights. If None, uses uniform weights of 1.0 for all events.
    x : np.ndarray
        X-axis data values.
    y : np.ndarray
        Y-axis data values.
    bins : np.ndarray or list of two np.ndarray
        Bin edges. Pass a 2-element list ``[x_bins, y_bins]`` for different axes.
    overflow : bool, optional
        If True (default), values outside the bin range are clipped into edge
        bins on both axes. If False, standard numpy behavior.
    **kwargs
        Passed to np.histogram2d().

    Returns
    -------
    np.ndarray
        2D histogram counts of shape (len(x_bins)-1, len(y_bins)-1).
    """
    if isinstance(bins, (list, tuple)) and len(bins) == 2 and not np.isscalar(bins[0]) and not np.isscalar(bins[1]):
        x_bins, y_bins = bins
    else:
        x_bins = y_bins = bins
    if weights is None:
        weights = np.ones(len(x))
    if overflow:
        n_x = len(x_bins) - 1
        n_y = len(y_bins) - 1
        xi = digitize_with_overflow(x, x_bins)
        yi = digitize_with_overflow(y, y_bins)
        flat = xi * n_y + yi
        return np.bincount(flat, weights=weights, minlength=n_x * n_y).reshape(n_x, n_y).astype(float)
    else:
        return np.histogram2d(x, y, bins=[x_bins, y_bins], weights=weights, **kwargs)[0]


def bin_geometry(var) -> tuple[np.ndarray, np.ndarray]:
    """Return (bin_centers, bin_widths) for a VariableConfig.

    Centers are ``var.bin_centers``; widths are ``np.diff(var.bin_labels)``,
    which can differ from ``np.diff(var.bins)`` when an overflow display label
    is placed past the physical bin edge.
    """
    centers = np.asarray(var.bin_centers)
    widths  = np.diff(np.asarray(var.bin_labels, dtype=float))
    return centers, widths
