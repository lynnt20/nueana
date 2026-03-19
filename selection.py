import sys; sys.path.append("/exp/sbnd/app/users/lynnt/cafpyana")
from makedf.util import *
from pyanalib.pandas_helpers import *

import numpy as np
import pandas as pd
from .utils import ensure_lexsorted
from .constants import signal_dict, generic_dict
from .geometry import whereTPC

def InSpill(df,spill_start=0.2, spill_end=2.2):
    return (df.slc.barycenterFM.flashTime > spill_start) & (df.slc.barycenterFM.flashTime < spill_end)

def InScore(df,score_cut=0.02):
    return (df.slc.barycenterFM.score > score_cut)

def select(indf,
           stage = None,
           savedict = True,
           spring=False,
           realisticFV=True,
           spill_start=0.2, 
           spill_end=2.2, 
           score_cut=0.02,
           shower_scale=1.25,
           min_shower_energy=0.5,
           max_track_length=200,
           min_conversion_gap=0.001,
           max_conversion_gap=2,
           min_dedx=1.25,
           max_dedx=2.5,
           min_opening_angle=0.03,
           max_opening_angle=0.15,
           min_shower_length=50,
           max_shower_length=200,
           min_shower_density=5,):
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
    realisticFV : bool, optional
        Whether to apply realistic active volume cut (default: True)
    spill_start : float, optional
        Minimum flash time for beam spill (default: 0.2)
    spill_end : float, optional
        Maximum flash time for beam spill (default: 2.2)
    score_cut : float, optional
        Minimum flash matching score (default: 0.02)
    shower_scale : float, optional
        Scale factor for shower energy, reco->true (default: 1.25)
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
    min_opening_angle : float, optional
        Minimum shower opening angle (default: 0.03)
    max_opening_angle : float, optional
        Maximum shower opening angle (default: 0.2)
    min_shower_length : float, optional
        Minimum shower length in cm (default: 50)
    max_shower_length : float, optional
        Maximum shower length in cm (default: 200)
    min_shower_density : float, optional
        Minimum shower density (default: 5)
    
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
        'shower length',
        'shower density'
    ]
    if stage is not None and stage not in valid_stages:
        raise ValueError(f"Unknown stage '{stage}'. Valid options: {valid_stages}")

    df_dict = {}
    df = indf.copy()
    
    def save_stage(stage_name, current_df):
        """Save stage to dict if savedict=True and early return if at target stage."""
        if savedict:
            df_dict[stage_name] = current_df
        if stage == stage_name:
            return df_dict if savedict else current_df
        return None

    shower_var = ("primshw","shw","maxplane_energy") if spring else ("primshw","shw","bestplane_energy")
    add_col = True
    # for col in df.columns:
    #     if "reco_energy" in "_".join(list(col)):
    #         add_col = False
    #         break
    # if add_col: 
    #     df = multicol_add(df,((ensure_lexsorted(df,axis=1)[shower_var])*shower_scale).rename(("primshw","shw","reco_energy")))
    df[("primshw","shw","reco_energy",'','','')] = ensure_lexsorted(df,axis=1)[shower_var]*shower_scale
    # ** these cuts done already in makedf
    # * require nuscore > 0.5
    # * require not clear cosmic 
    # * require reco vertex in AV
    # * require that there is a primary shower (at least one pfp w/ trackScore < 0.5)
    if realisticFV:
        df = df[(InFV(df.slc.vertex,det="SBND_nohighyz",inzback=0))]
    df = df[df.slc.nu_score>0.5]
    result = save_stage('preselection', df)
    if result is not None: return result
    
    # * require that the matched (many-to-many) is inside the beam spill
    df = df[InSpill(df, spill_start, spill_end) & InScore(df, score_cut)]
    result = save_stage('flash matching', df)
    if result is not None: return result

    # * require that primary shower > min_shower_energy
    df = df[df.primshw.shw.reco_energy > min_shower_energy]
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

    df = df[(df.primshw.shw.density > min_shower_density)]
    result = save_stage('shower density', df)
    if result is not None: return result
    
    return df_dict if savedict else df

def select_sideband(indf, 
                    savedict=False,
                    min_conversion_gap=2,
                    max_conversion_gap=1e3,
                    min_dedx=3,
                    max_dedx=6,
                    min_opening_angle=0.2,
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
    min_dedx : float, default 3
        Minimum dE/dx on best plane (overrides select's default of 1.25)
    max_dedx : float, default 6
        Maximum dE/dx on best plane (overrides select's default of 2.5)
    min_opening_angle : float, default 0.2
        Minimum shower opening angle (overrides select's default of 0.03)
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

