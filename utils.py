import pandas as pd 
from .constants import signal_dict

import sys; sys.path.append("/exp/sbnd/app/users/lynnt/cafpyana")
from makedf.util import *
from pyanalib.pandas_helpers import *

n_max_concat=10

# credit for first three functions to Mun! 
def get_n_split(file):
    this_split_df = pd.read_hdf(file, key="split")
    this_n_split = this_split_df.n_split.iloc[0]
    return this_n_split

def print_keys(file):
    with pd.HDFStore(file, mode='r') as store:
        keys = store.keys()       # list of all keys in the file
        print("Keys:", keys)
        
def load_dfs(file, keys2load):
    out_df_dict = {}
    this_n_keys = get_n_split(file) 
    n_concat = min(n_max_concat, this_n_keys)
    for key in keys2load:
        dfs = []  # collect all splits for this key
        for i in range(n_concat):
            this_df = pd.read_hdf(file, key=f"{key}_{i}")
            dfs.append(this_df)
        out_df_dict[key] = pd.concat(dfs, ignore_index=False)
    return out_df_dict

def get_mcexposure_info(file_list): 
    ngates = 0
    pot = 0
    nevents = 0
    for i, file in enumerate(file_list):
        out_df = load_dfs(file,["hdr"])
        hdr_df = out_df["hdr"]
        ngates += hdr_df.reset_index().drop_duplicates(subset=['run','subrun'])['ngenevt'].sum()
        pot += hdr_df.reset_index().pot.sum()
        nevents += len(hdr_df)
    return ngates, pot, nevents

# Defensive: ensure DataFrame axes are fully lexsorted when using MultiIndex
# This avoids pandas PerformanceWarning about indexing past lexsort depth
def ensure_lexsorted(frame, axis):
    # axis: 0 -> index, 1 -> columns
    idx = frame.index if axis == 0 else frame.columns
    if isinstance(idx, pd.MultiIndex) and getattr(idx, "lexsort_depth", 0) < idx.nlevels:
        # sort by all levels (returns a new frame)
        return frame.sort_index(axis=axis)
    return frame

# helper functions for reproducing ccnue art filter logic
def whereTPC(df):
    xmax = 202.20000000000002
    xmin = -202.20000000000002
    ymax = 203.732262000002
    ymin = -203.732262000002
    zmax = 500#.09999999999997
    zmin = 0
    return (df.x > xmin) & (df.x < xmax) & (df.y > ymin) & (df.y < ymax) & (df.z > zmin) & (df.z < zmax)

def ccnuefilt(df):
    return whereTPC(df.position) & (df.iscc==1) & (np.isnan(df.e.genE)==False) & (abs(df.pdg)==12)

def remove_ccnue(indf):
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

def define_signal(indf: pd.DataFrame,prefix=None):
    # sort by row 
    indf = ensure_lexsorted(indf,0)
    # sort by column make copy to preserve column ordering of original
    nudf = ensure_lexsorted(indf.copy(),1)

    if prefix==None: mcdf = nudf
    else: mcdf = nudf[prefix]

    whereFV = InFV(df=mcdf.position, inzback=0, det="SBND")
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