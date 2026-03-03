"""Histogram utilities with overflow handling."""
import numpy as np

def get_hist1d(weights, data, bins): 
    """1D histogram with overflow folded into last bin.
    
    Parameters
    ----------
    weights : np.ndarray
        Per-event weights.
    data : np.ndarray
        Data values to histogram.
    bins : np.ndarray
        Bin edges. Values above bins[-1] are clipped to bins[-1] - 1e-10.
    
    Returns
    -------
    np.ndarray
        Histogram counts of shape (len(bins)-1,).
    """
    clipped = np.clip(data, bins[0], bins[-1] - 1e-10)
    return np.histogram(clipped, bins=bins, weights=weights)[0]

def get_hist2d(weights, x, y, bins):
    """2D histogram with overflow folded into last bin on both axes.
    
    Parameters
    ----------
    weights : np.ndarray
        Per-event weights.
    x : np.ndarray
        X-axis data values.
    y : np.ndarray
        Y-axis data values.
    bins : np.ndarray
        Bin edges for both axes. Values above bins[-1] are clipped to bins[-1] - 1e-10.
    
    Returns
    -------
    np.ndarray
        2D histogram counts of shape (len(bins)-1, len(bins)-1).
    """
    cy = np.clip(y, bins[0], bins[-1] - 1e-10)
    cx = np.clip(x, bins[0], bins[-1] - 1e-10)
    return np.histogram2d(cx, cy, bins=bins, weights=weights)[0]
