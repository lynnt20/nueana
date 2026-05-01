"""
Histogram utilities with overflow handling.

Conventions
-----------
- Weights must be the first argument to automatically pass it into 
  `np.apply_along_axis` for vectorized systematic uncertainty calculations.
"""
import numpy as np

__all__ = ['get_hist1d', 'get_hist2d']

def get_hist1d(weights=None, data=None, bins=None, overflow=True, **kwargs): 
    """1D histogram with optional overflow handling.
    
    Parameters
    ----------
    weights : np.ndarray, optional
        Per-event weights. If None (default), uses uniform weights of 1.0 for all events.
    data : np.ndarray
        Data values to histogram.
    bins : np.ndarray
        Bin edges.
    overflow : bool, optional
        If True (default), values above bins[-1] are clipped to bins[-1] - 1e-10
        to fold overflow into the last bin. Non-finite values are also assigned to
        edge bins (`+/-inf` to overflow/underflow, `NaN` to overflow).
        If False, uses standard numpy histogram behavior with no clipping.
    **kwargs
        Additional keyword arguments to pass to np.histogram().
    
    Returns
    -------
    np.ndarray
        Histogram counts of shape (len(bins)-1,).
    """
    if weights is None:
        weights = np.ones(len(data))
    if overflow==True:
        cleaned = np.nan_to_num(data, nan=bins[-1] - 1e-10,
                                posinf=bins[-1] - 1e-10,
                                neginf=bins[0])
        clipped = np.clip(cleaned, bins[0], bins[-1] - 1e-10)
        return np.histogram(clipped, bins=bins, weights=weights, **kwargs)[0]
    else:
        return np.histogram(data,bins=bins,weights=weights,**kwargs)[0]

def get_hist2d(weights=None, x=None, y=None, bins=None, overflow=True, **kwargs):
    """2D histogram with optional overflow handling on both axes.
    
    Parameters
    ----------
    weights : np.ndarray, optional
        Per-event weights. If None (default), uses uniform weights of 1.0 for all events.
    x : np.ndarray
        X-axis data values.
    y : np.ndarray
        Y-axis data values.
    bins : np.ndarray
        Bin edges for both axes.
    overflow : bool, optional
        If True (default), values above bins[-1] are clipped to bins[-1] - 1e-10
        on both axes to fold overflow into the last bin. Non-finite values are
        also assigned to edge bins (`+/-inf` to overflow/underflow, `NaN` to overflow).
        If False, uses standard numpy histogram behavior with no clipping.
    **kwargs
        Additional keyword arguments to pass to np.histogram2d().
    
    Returns
    -------
    np.ndarray
        2D histogram counts of shape (len(bins)-1, len(bins)-1).
    """
    if isinstance(bins, (list, tuple)) and len(bins) == 2 and not np.isscalar(bins[0]) and not np.isscalar(bins[1]):
        x_bins, y_bins = bins
    else:
        x_bins = y_bins = bins
    if weights is None:
        weights = np.ones(len(x))
    if overflow==True:
        cy = np.nan_to_num(y, nan=y_bins[-1] - 1e-10,
                           posinf=y_bins[-1] - 1e-10,
                           neginf=y_bins[0])
        cx = np.nan_to_num(x, nan=x_bins[-1] - 1e-10,
                           posinf=x_bins[-1] - 1e-10,
                           neginf=x_bins[0])
        cy = np.clip(cy, y_bins[0], y_bins[-1] - 1e-10)
        cx = np.clip(cx, x_bins[0], x_bins[-1] - 1e-10)
        return np.histogram2d(cx, cy, bins=[x_bins, y_bins], weights=weights, **kwargs)[0]
    else: 
        return np.histogram2d(x, y, bins=[x_bins, y_bins], weights=weights, **kwargs)[0]

