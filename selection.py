import sys; sys.path.append("/exp/sbnd/app/users/lynnt/cafpyana")
from makedf.util import *
from pyanalib.pandas_helpers import *

import numpy as np
import pandas as pd
from .utils import ensure_lexsorted
from .constants import signal_dict, generic_dict
from .geometry import whereTPC, InRealisticFV

def InSpill(df,spill_start=0.2, spill_end=2.2):
    return (df.slc.barycenterFM.flashTime > spill_start) & (df.slc.barycenterFM.flashTime < spill_end)

def InScore(df,score_cut=0.02):
    return (df.slc.barycenterFM.score > score_cut)

def select(indf, 
           spring=False,
           realisticFV=True,
           spill_start=0.2, 
           spill_end=2.2, 
           score_cut=0.02,
           shower_scale=1.25,
           min_shower_energy=0.5,
           max_track_length=200,
           max_conversion_gap=2,
           min_dedx=1,
           max_dedx=2.5,
           min_opening_angle=0.03,
           max_opening_angle=0.2):
    """
    Apply selection cuts to neutrino interaction data.
    
    Parameters
    ----------
    indf : pandas.DataFrame
        Input DataFrame with neutrino interaction data
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
    if realisticFV:
        df = df[InRealisticFV(df.slc.vertex)]
    df_dict['preselection'] = df
    
    # * require that the matched (many-to-many) is inside the beam spill
    df = df[InSpill(df, spill_start, spill_end) & InScore(df, score_cut)]
    df_dict['flash matching'] = df

    # * require that primary shower > min_shower_energy
    shower_var = ("primshw","shw","maxplane_energy") if spring else ("primshw","shw","bestplane_energy")
    df = multicol_add(df,((ensure_lexsorted(df,axis=1)[shower_var])*shower_scale).rename(("primshw","shw","reco_energy")))
    df = df[df.primshw.shw.reco_energy > min_shower_energy]
    df_dict['shower energy'] = df 

    # * require track length < max_track_length cm
    df = df[np.isnan(df.primtrk.trk.len) | (df.primtrk.trk.len < max_track_length)]
    # df = df.drop('primtrk',axis=1,level=0)
    df_dict['muon rejection'] = df

    df = df[df.primshw.shw.conversion_gap < max_conversion_gap]
    df_dict['conversion gap'] = df

    df = df[(df.primshw.shw.bestplane_dEdx > min_dedx) & (df.primshw.shw.bestplane_dEdx < max_dedx)]
    df_dict['dEdx'] = df

    df = df[(df.primshw.shw.open_angle > min_opening_angle) & 
            (df.primshw.shw.open_angle < max_opening_angle)]
    df_dict['opening angle'] = df

    return df_dict

def ccnuefilt(df):
    """Filter for CC nue events based on truth information.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with truth information (position, iscc, e.genE, pdg columns).
    
    Returns
    -------
    pandas.Series or numpy.ndarray
        Boolean mask for CC nue events within TPC.
    """
    return whereTPC(df.position) & (df.iscc==1) & (np.isnan(df.e.genE)==False) & (abs(df.pdg)==12)

def remove_ccnue(indf):
    """Remove CC nue events from DataFrame.
    
    Parameters
    ----------
    indf : pandas.DataFrame
        Input DataFrame with MultiIndex columns containing truth information.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with CC nue events removed.
    """
    df = indf.copy()
    bnb_nuecc_idx = df[ccnuefilt(df.slc.truth)].reset_index()[[('__ntuple', '', '', '', '', ''),('entry', '', '', '', '', '')]].drop_duplicates()

    indexes = df.index.names
    df = multicol_merge(bnb_nuecc_idx,
                        df.reset_index(),
                        left_on=[('__ntuple', '', '', '', '', ''),('entry', '', '', '', '', '')],
                        right_on=[('__ntuple', '', '', '', '', ''),('entry', '', '', '', '', '')],
                        how='outer',indicator=True).set_index(indexes)
    print("% of slices dropped: ", np.round(len(df[df._merge =='both'])/len(df)*100,2)) 
    df = df[df._merge == 'right_only']
    df = ensure_lexsorted(df,axis=0)
    df = ensure_lexsorted(df,axis=1)
    df = df.drop(columns=['_merge'])
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
    # sort by row 
    indf = ensure_lexsorted(indf,0)
    # sort by column make copy to preserve column ordering of original
    nudf = ensure_lexsorted(indf.copy(),1)

    if prefix==None: mcdf = nudf
    else: mcdf = nudf[prefix]

    whereFV = InFV(df=mcdf.position, inzback=0, det="SBND") & InRealisticFV(df=mcdf.position)
    whereAV = InAV(df=mcdf.position)
    whereCCnue = ((mcdf.iscc==1)  # require CC interaction
                & (abs(mcdf.pdg)==12)  # require neutrino to be a nue
                & (abs(mcdf.e.pdg)==11) # require electron to be the primary (?) 
                & (mcdf.e.genE > 0.5) # require primary electron to deposit ___ MeV
                )

    if "signal" not in nudf.columns: nudf["signal"] = -1    
    # background
    nudf["signal"] = np.where(whereFV & (mcdf.iscc==1) & (abs(mcdf.pdg)==14) & (mcdf.npi0>0), signal_dict["numuCCpi0"], nudf["signal"]) # numu cc FV
    nudf["signal"] = np.where(whereFV & (mcdf.iscc==0) & (mcdf.npi0 > 0), signal_dict["NCpi0"], nudf["signal"]) # nc pi0 FV
    nudf["signal"] = np.where(whereFV & (mcdf.iscc==1) & (abs(mcdf.pdg)==12), signal_dict["othernueCC"], nudf["signal"]) # nue cc FV
    nudf["signal"] = np.where(whereFV & (mcdf.iscc==1) & (abs(mcdf.pdg)==14) & (mcdf.npi0 == 0), signal_dict["othernumuCC"], nudf["signal"]) # numu cc other FV
    nudf["signal"] = np.where(whereFV & (mcdf.iscc==0) & (mcdf.npi0 == 0), signal_dict["otherNC"], nudf["signal"]) # nc other FV
    nudf["signal"] = np.where(whereAV & (nudf["signal"]<0), signal_dict["nonFV"], nudf['signal']) # nonFV
    nudf["signal"] = np.where(whereAV == False, signal_dict["dirt"], nudf["signal"]) # dirt
    nudf["signal"] = np.where(np.isnan(mcdf.E), signal_dict['cosmic'], nudf["signal"])
    
    nudf["signal"] = np.where(whereFV & whereCCnue, signal_dict["nueCC"], nudf["signal"])
    if ((nudf.signal < 0) | (nudf.signal >= len(signal_dict))).any(): 
        print("Warning: unidentified signal/bacgkr channels present.")
    indf["signal"] = nudf["signal"]
    return indf

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

