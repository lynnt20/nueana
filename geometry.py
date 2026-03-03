"""Detector geometry and fiducial volume definitions."""
import numpy as np

# Bounds obtained directly from geometry service for sbndcode v10_14_02_01
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
    """
    return (df.x > xmin) & (df.x < xmax) & (df.y > ymin) & (df.y < ymax) & (df.z > zmin) & (df.z < zmax)

def InRealisticFV(df):
    """
    Filter events based on realistic fiducial volume criteria.

    This function applies spatial cuts to determine if events fall within the realistic
    fiducial volume of the detector based on z and y coordinates.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing event data with 'z' and 'y' coordinate columns.

    Returns
    -------
    pandas.Series or numpy.ndarray
        Boolean mask where True indicates the event is within the realistic fiducial volume.
        The condition is satisfied when either:
        - z >= 250 cm AND y < 100 cm, OR
        - z < 250 cm

    Notes
    -----
    The realistic fiducial volume is defined by the logical OR of two conditions:
    1. Events with z-coordinate >= 250 cm must have y-coordinate < 100 cm
    2. All events with z-coordinate < 250 cm are included regardless of y-coordinate
    """
    return (((df.z >= 250) & (df.y < 100)) | (df.z < 250))
