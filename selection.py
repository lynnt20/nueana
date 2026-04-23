import numpy as np
import pandas as pd
from . import config
from makedf.util import *
from pyanalib.pandas_helpers import *

from .utils import ensure_lexsorted
from .constants import signal_dict, generic_dict
from .geometry import whereTPC

def InSpill(df,spill_start=0.335, spill_end=0.335+1.6):
    return (df.slc.barycenterFM.flashTime > spill_start) & (df.slc.barycenterFM.flashTime < spill_end)

def InScore(df,score_cut=0.02):
    return (df.slc.barycenterFM.score > score_cut)

def select(indf,
           stage = None,
           savedict = False,
           spring=True,
           realisticFV=True,
           spill_start=0.335, 
           spill_end=0.335+1.6, 
           score_cut=0.02,
           nuscore_cut=0.5,
           pe_cut = 2e3,
           shower_scale=1.17,
           min_shower_energy=0.5,
           max_track_length=200,
           min_conversion_gap=0.001,
           max_conversion_gap=2,
           min_dedx=1.25,
           max_dedx=2.5,
           min_opening_angle=0.03,
           max_opening_angle=0.15,
           min_shower_length=0.1,
           max_shower_length=200,
           min_direction=-1,
           max_direction=1,):
    """
    Apply selection cuts to neutrino interaction data.
    
    Parameters
    ----------
    indf : pandas.DataFrame
        Input DataFrame with neutrino interaction data
    stage : str, optional
        If provided, stop after filling that stage and return immediately.
        Allowed values are: 'preselection', 'flash matching', 'shower energy',
        'muon rejection', 'conversion gap', 'dEdx', 'opening angle', 'shower length'.
        If None (default), applies all cuts and returns all stages.
    savedict : bool, optional
        If False (default), return only the final selected DataFrame (or selected stage).
        If True, return a dictionary with all saved stages.
    spring : bool, optional
        If True (default), use max-plane shower energy for reco-energy definition.
        If False, use best-plane shower energy.
    realisticFV : bool, optional
        Whether to apply realistic active volume cut (default: True)
    spill_start : float, optional
        Minimum flash time for beam spill (default: 0.335)
    spill_end : float, optional
        Maximum flash time for beam spill (default: 1.935)
    score_cut : float, optional
        Minimum flash matching score (default: 0.02)
    nuscore_cut : float, optional
        Minimum neutrino score for preselection (default: 0.5)
    pe_cut : float, optional
        Minimum flash photoelectrons for preselection (default: 2000)
    shower_scale : float, optional
        Scale factor for shower energy, reco->true (default: 1.17)
    min_shower_energy : float, optional
        Minimum primary shower energy in GeV (default: 0.5)
    max_track_length : float, optional
        Maximum track length in cm for muon rejection (default: 200)
    min_conversion_gap : float, optional
        Minimum conversion gap (default: 0.001)
    max_conversion_gap : float, optional
        Maximum conversion gap (default: 2)
    min_dedx : float, optional
        Minimum dE/dx on best plane (default: 1.25)
    max_dedx : float, optional
        Maximum dE/dx on best plane (default: 2.5)
    min_opening_angle : float, optional
        Minimum shower opening angle (default: 0.03)
    max_opening_angle : float, optional
        Maximum shower opening angle (default: 0.15)
    min_shower_length : float, optional
        Minimum shower length in cm (default: 0.1)
    max_shower_length : float, optional
        Maximum shower length in cm (default: 200)
    
    Returns
    -------
    dict or pandas.DataFrame
        If savedict=True, returns dictionary of DataFrames after each selection cut.
        If savedict=False, returns only the final (or stage-specific) DataFrame.
        If stage is provided, returns only up to and including that stage.
    """
    valid_stages = [
        'preselection',
        'flash matching',
        'shower energy',
        'muon rejection',
        'conversion gap',
        'dEdx',
        'opening angle',
        'shower length'
        ]
    if stage is not None and stage not in valid_stages:
        raise ValueError(f"Unknown stage '{stage}'. Valid options: {valid_stages}")

    df_dict = {}
    df = indf.copy()
    if spring:
        df[("primshw","shw","reco_energy",'','','')] = df.primshw.shw.maxplane_energy*shower_scale
    else:
        df[("primshw","shw","reco_energy",'','','')] = df.primshw.shw.bestplane_energy*shower_scale
    
    def save_stage(stage_name, current_df):
        """Save stage to dict if savedict=True and early return if at target stage."""
        if savedict:
            df_dict[stage_name] = current_df
        if stage == stage_name:
            return df_dict if savedict else current_df
        return None
    # ** these cuts done already in makedf
    # * require nuscore > 0.5
    # * require not clear cosmic 
    # * require reco vertex in AV
    # * require that there is a primary shower (at least one pfp w/ trackScore < 0.5)
    if realisticFV:
        df = df[(InFV(df.slc.vertex,det="SBND_nohighyz",inzback=0))]
    df = df[df.slc.barycenterFM.flashPEs > pe_cut]
    df = df[df.slc.nu_score>nuscore_cut]
    df = df[df.slc.is_clear_cosmic==0]
    result = save_stage('preselection', df)
    if result is not None: return result
    
    # * require that the matched (many-to-many) is inside the beam spill
    df = df[InSpill(df, spill_start, spill_end) & InScore(df, score_cut)]
    result = save_stage('flash matching', df)
    if result is not None: return result

    # * require that primary shower > min_shower_energy
    df = df[df.primshw.shw.reco_energy > min_shower_energy]
    df = df[df.primshw.trackScore >0]
    result = save_stage('shower energy', df)
    if result is not None: return result

    # * require track length < max_track_length cm
    df = df[np.isnan(df.primtrk.trk.len) | (df.primtrk.trk.len < max_track_length)]
    result = save_stage('muon rejection', df)
    if result is not None: return result

    df = df[(df.primshw.shw.conversion_gap < max_conversion_gap) & 
            (df.primshw.shw.conversion_gap > min_conversion_gap)]
    result = save_stage('conversion gap', df)
    if result is not None: return result

    df = df[(df.primshw.shw.bestplane_dEdx > min_dedx) & (df.primshw.shw.bestplane_dEdx < max_dedx)]
    result = save_stage('dEdx', df)
    if result is not None: return result

    df = df[(df.primshw.shw.open_angle > min_opening_angle) & 
            (df.primshw.shw.open_angle < max_opening_angle)]
    result = save_stage('opening angle', df)
    if result is not None: return result
    
    df = df[(df.primshw.shw.len < max_shower_length) & 
            (df.primshw.shw.len > min_shower_length)]
    result = save_stage('shower length', df)
    if result is not None: return result
    
    df = df[(df.primshw.shw.dir.z < max_direction) &
            (df.primshw.shw.dir.z > min_direction)]

    return df_dict if savedict else df

def select_sideband(indf, 
                    savedict=False,
                    min_conversion_gap=2,
                    max_conversion_gap=1e3,
                    max_track_length=1e3,
                    min_dedx=3,
                    max_dedx=6,
                    min_opening_angle=0.15,
                    max_opening_angle=1.0,
                    **kwargs):
    """Apply sideband selection cuts with modified default parameters.
    
    This function calls select() with sideband-specific defaults that differ
    from the standard selection. All parameters can be overridden via kwargs.
    
    Parameters
    ----------
    indf : pandas.DataFrame
        Input DataFrame with neutrino interaction data
    savedict : bool, default False
        Whether to return dict of all stages (overrides select's default of True)
    min_conversion_gap : float, default 2
        Minimum conversion gap (overrides select's default of 0.001)
    max_conversion_gap : float, default 1000
        Maximum conversion gap (overrides select's default of 2)
    max_track_length : float, default 1000
        Maximum track length in cm (overrides select's default of 200)
    min_dedx : float, default 3
        Minimum dE/dx on best plane (overrides select's default of 1.25)
    max_dedx : float, default 6
        Maximum dE/dx on best plane (overrides select's default of 2.5)
    min_opening_angle : float, default 0.15
        Minimum shower opening angle (overrides select's default of 0.03)
    max_opening_angle : float, default 1.0
        Maximum shower opening angle (overrides select's default of 0.15)
    **kwargs
        All other parameters are passed to select() and will use its defaults
        unless explicitly specified here.
    
    Returns
    -------
    pandas.DataFrame or dict
        Result from select() with sideband-specific cuts applied.
    """
    df = select(indf,
                savedict=savedict,
                max_track_length=max_track_length,
                min_conversion_gap=min_conversion_gap,
                max_conversion_gap=max_conversion_gap,
                min_dedx=min_dedx,
                max_dedx=max_dedx,
                min_opening_angle=min_opening_angle,
                max_opening_angle=max_opening_angle,
                **kwargs)
    return df


def define_signal(indf: pd.DataFrame, prefix=None):
    """Define signal/background categories for neutrino interactions.
    
    Categorizes events into signal (CC nue) and various background categories
    based on truth information and fiducial volume.
    
    Parameters
    ----------
    indf : pandas.DataFrame
        Input DataFrame with MultiIndex columns containing truth information.
    prefix : str or tuple, optional
        Column prefix to access truth information. If None, uses top-level columns.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with added 'signal' column indicating event category using signal_dict.
    """
    # Keep lexsorted axes for robust multi-index access without forcing a full copy.
    nudf = ensure_lexsorted(ensure_lexsorted(indf, 0), 1)

    if prefix is None:
        mcdf = nudf
    else:
        mcdf = nudf[prefix]

    whereFV = InFV(mcdf.position,det="SBND_nohighyz",inzback=0)
    whereAV = InAV(df=mcdf.position)
    whereCCnue = ((mcdf.iscc==1)  # require CC interaction
                & (abs(mcdf.pdg)==12)  # require neutrino to be a nue
                & (abs(mcdf.e.pdg)==11) # require electron to be the primary (?) 
                & (mcdf.e.genE > 0.5) # require primary electron to deposit ___ MeV
                )

    if "signal" in nudf.columns:
        signal = nudf["signal"].to_numpy(copy=True)
    else:
        signal = np.full(len(nudf), -1, dtype=np.int16)

    # background
    signal[whereFV & (mcdf.iscc==1) & (abs(mcdf.pdg)==14) & (mcdf.npi0>0)] = signal_dict["numuCCpi0"] # numu cc FV
    signal[whereFV & (mcdf.iscc==0) & (mcdf.npi0 > 0)] = signal_dict["NCpi0"] # nc pi0 FV
    signal[whereFV & (mcdf.iscc==1) & (abs(mcdf.pdg)==12)] = signal_dict["othernueCC"] # nue cc FV
    signal[whereFV & (mcdf.iscc==1) & (abs(mcdf.pdg)==14) & (mcdf.npi0 == 0)] = signal_dict["othernumuCC"] # numu cc other FV
    signal[whereFV & (mcdf.iscc==0) & (mcdf.npi0 == 0)] = signal_dict["otherNC"] # nc other FV
    signal[whereAV & (signal < 0)] = signal_dict["nonFV"] # nonFV
    signal[whereAV == False] = signal_dict["dirt"] # dirt
    signal[np.isnan(mcdf.E)] = signal_dict['cosmic']

    signal[whereFV & whereCCnue] = signal_dict["nueCC"]
    nudf["signal"] = signal
    if ((nudf.signal < 0) | (nudf.signal >= len(signal_dict))).any(): 
        print("Warning: unidentified signal/bacgkr channels present.")
    return nudf

def define_generic(indf: pd.DataFrame, prefix=None):
    """Define generic signal/background categories for neutrino interactions.
    
    Categorizes events into broad categories: CC nu, NC nu, non-FV, dirt, cosmic.
    
    Parameters
    ----------
    indf : pandas.DataFrame
        Input DataFrame with MultiIndex columns containing truth information.
    prefix : str or tuple, optional
        Column prefix to access truth information. If None, uses top-level columns.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with added 'signal' column indicating event category using generic_dict.
    """
    # sort by row 
    indf = ensure_lexsorted(indf,0)
    # sort by column make copy to preserve column ordering of original
    nudf = ensure_lexsorted(indf.copy(),1)

    if prefix==None: mcdf = nudf
    else: mcdf = nudf[prefix]

    whereFV = InFV(df=mcdf.position, inzback=0, det="SBND")
    whereAV = InAV(df=mcdf.position)
    
    if "signal" not in nudf.columns: nudf["signal"] = -1    
    # background
    nudf["signal"] = np.where(whereAV == False, generic_dict["dirt"], nudf["signal"]) # dirt    
    nudf["signal"] = np.where(whereAV, generic_dict["nonFV"], nudf['signal']) # nonFV
    nudf["signal"] = np.where(whereFV & (mcdf.iscc==0), generic_dict["NCnu"], nudf["signal"])
    nudf["signal"] = np.where(whereFV & (mcdf.iscc==1), generic_dict["CCnu"], nudf["signal"])
    nudf["signal"] = np.where(np.isnan(mcdf.E), generic_dict['cosmic'], nudf["signal"])

    if ((nudf.signal < 0) | (nudf.signal >= len(generic_dict))).any(): 
        print("Warning: unidentified signal/bacgkr channels present.")
    indf["signal"] = nudf["signal"]
    return indf

