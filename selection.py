import numpy as np

def InSpill(df,spill_start=0.2, spill_end=2.2):
    return (df.slc.barycenterFM.flashTime > spill_start) & (df.slc.barycenterFM.flashTime < spill_end)

def InScore(df,score_cut=0.02):
    return (df.slc.barycenterFM.score > score_cut)

def InRealisticAV(df):
    """
    Filter events based on realistic active volume criteria.

    This function applies spatial cuts to determine if events fall within the realistic
    active volume of the detector based on z and y coordinates.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing event data with 'z' and 'y' coordinate columns.

    Returns
    -------
    pandas.Series or numpy.ndarray
        Boolean mask where True indicates the event is within the realistic active volume.
        The condition is satisfied when either:
        - z >= 250 cm AND y < 100 cm, OR
        - z < 250 cm

    Notes
    -----
    The realistic active volume is defined by the logical OR of two conditions:
    1. Events with z-coordinate >= 250 cm must have y-coordinate < 100 cm
    2. All events with z-coordinate < 250 cm are included regardless of y-coordinate
    """

    return (((df.z >= 250) & (df.y < 100)) | (df.z < 250))

def select(indf, 
           spring=False,
           realisticAV=True,
           spill_start=0.2, 
           spill_end=2.2, 
           score_cut=0.02,
           min_shower_energy=0.5,
           max_track_length=200,
           max_conversion_gap=2,
           min_dedx=1,
           max_dedx=2.5,
           max_opening_angle=0.2):
    """
    Apply selection cuts to neutrino interaction data.
    
    Parameters
    ----------
    indf : pandas.DataFrame
        Input DataFrame with neutrino interaction data
    realisticAV : bool, optional
        Whether to apply realistic active volume cut (default: True)
    spill_start : float, optional
        Minimum flash time for beam spill (default: 0.2)
    spill_end : float, optional
        Maximum flash time for beam spill (default: 2.2)
    score_cut : float, optional
        Minimum flash matching score (default: 0.02)
    min_shower_energy : float, optional
        Minimum primary shower energy in GeV (default: 0.5)
    max_track_length : float, optional
        Maximum track length in cm for muon rejection (default: 200)
    max_conversion_gap : float, optional
        Maximum conversion gap (default: 2)
    min_dedx : float, optional
        Minimum dE/dx on best plane (default: 1)
    max_dedx : float, optional
        Maximum dE/dx on best plane (default: 2.5)
    max_opening_angle : float, optional
        Maximum shower opening angle (default: 0.2)
    
    Returns
    -------
    dict
        Dictionary of DataFrames after each selection cut
    """
    df_dict = {}
    df = indf.copy()
    # ** these cuts done already in makedf
    # * require nuscore > 0.5
    # * require not clear cosmic 
    # * require reco vertex in AV
    # * require that there is a primary shower (at least one pfp w/ trackScore < 0.5)
    if realisticAV:
        df = df[InRealisticAV(df.slc.vertex)]
    df_dict['preselection'] = df
    
    # * require that the matched (many-to-many) is inside the beam spill
    df = df[InSpill(df, spill_start, spill_end) & InScore(df, score_cut)]
    df_dict['flash matching'] = df

    # * require that primary shower > min_shower_energy
    if spring:
        df = df[df.primshw.shw.maxplane_energy > min_shower_energy]
    else:
        df = df[df.primshw.shw.bestplane_energy > min_shower_energy]
    df_dict['shower energy'] = df 

    # * require track length < max_track_length cm
    df = df[np.isnan(df.primtrk.trk.len) | (df.primtrk.trk.len < max_track_length)]
    df = df.drop('primtrk',axis=1,level=0)
    df_dict['muon rejection'] = df

    df = df[df.primshw.shw.conversion_gap < max_conversion_gap]
    df_dict['conversion gap'] = df

    df = df[(df.primshw.shw.bestplane_dEdx > min_dedx) & (df.primshw.shw.bestplane_dEdx < max_dedx)]
    df_dict['dEdx'] = df

    df = df[df.primshw.shw.open_angle < max_opening_angle]
    df_dict['opening angle'] = df

    return df_dict

