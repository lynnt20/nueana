"""Detector geometry and fiducial volume definitions."""
import numpy as np

__all__ = ['whereTPC']

def whereTPC(df,
             xmin=-202.20000000000002,
             xmax= 202.20000000000002,
             ymin=-203.73225000000002,
             ymax= 203.73225000000002,
             zmin=0.0,
             zmax=501.0):
    """Check if coordinates are within TPC boundaries.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with x, y, z coordinate columns.
    xmin, xmax : float
        X-axis boundaries (default: TPC bounds from geometry service).
    ymin, ymax : float
        Y-axis boundaries (default: TPC bounds from geometry service).
    zmin, zmax : float
        Z-axis boundaries (default: TPC bounds from geometry service).
    
    Returns
    -------
    pandas.Series or numpy.ndarray
        Boolean mask indicating if coordinates are within TPC boundaries.
        
    Notes 
    -----
    The default bounds are based on the geometry service for sbndcode v10_14_02_01. 
    Adjust as needed for different geometries or versions.
    """
    return (df.x > xmin) & (df.x < xmax) & (df.y > ymin) & (df.y < ymax) & (df.z > zmin) & (df.z < zmax)