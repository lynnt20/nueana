"""Utility functions."""

import pandas as pd
import numpy as np
import statsmodels as sm

nu_idx_set =  ["ntuple","entry","rec.mc.nu__index"]
slc_idx_set = ["ntuple","entry","rec.slc__index"]

def flatten_df(df: pd.DataFrame): 
    """
    Flattens a multi-index dataframe. Changes column names from "col.sub0.sub1" to "col_sub0_sub1"
    
    Parameters
    ----------
    df: pandas dataframe
        dataframe to be flattened
    
    Returns
    -------
    flat_df: flattened pandas dataframe
    """
    flat_df = df.copy()
    flat_df.reset_index(inplace=True)
    flat_df.columns = ['_'.join(col) for col in flat_df.columns.values]
    flat_df.columns = [col.strip('_') for col in flat_df.columns]
    for col in flat_df.columns:
        if ".." in col:
            flat_df.rename(columns={col:col.replace("..","__")},inplace=True)
    return flat_df

def get_slc(df: pd.DataFrame):
    """
    Returns a dataframe containing only slices (duplicate slices are not dropped, pfps are dropped).
    "Duplicate" slices references slices corresponding to the same neutrino interaction. In other words,
    allows "slice double counting." 
    """
    nodup_df = df.drop_duplicates(subset=slc_idx_set)
    return nodup_df

def get_evt(df: pd.DataFrame):
    """
    Returns a dataframe containing only events (duplicate slices are dropped).
    """
    nodup_df = df.drop_duplicates(subset=nu_idx_set)
    return nodup_df

def get_signal_evt(df: pd.DataFrame):
    """
    Returns a dataframe containing only signal **events** (duplicate slices are dropped).
    Assumes the input dataframe has a "signal" column, where signal=0. 
    """
    # only run on flattened dataframes 
    nodup_df = df.drop_duplicates(subset=nu_idx_set)
    return nodup_df[nodup_df.signal == 0]

def get_backgr_evt(df: pd.DataFrame):
    """
    Returns a dataframe containing only background events (duplicate slices are dropped).
    
    Parameters
    ----------
    df: input dataframe
    
    Returns
    -------
    backgr_df: dataframe containing only background events
    """
    nodup_df = df.drop_duplicates(subset=nu_idx_set)
    slices_df = nodup_df[nodup_df["signal"]!=0]
    return slices_df

def get_signal_slc(df: pd.DataFrame):
    """
    Returns a dataframe containing only signal slices (Duplicate slices are not dropped, pfps are dropped)
    "Duplicate" slices references slices corresponding to the same neutrino interaction. In other words,
    allows "slice double counting."  
    """
    nodup_df = df.drop_duplicates(subset=slc_idx_set)
    return nodup_df[nodup_df.signal==0]

def get_backgr_slc(df: pd.DataFrame):
    """
    Returns a dataframe containing only background slices (Duplicate slices are not dropped, pfps are dropped)
    "Duplicate" slices references slices corresponding to the same neutrino interaction. In other words,
    allows "slice double counting." 
    """
    nodup_df = df.drop_duplicates(subset=slc_idx_set)
    return nodup_df[nodup_df.signal!=0]

def get_slices(df: pd.DataFrame, int_type):
    """
    Returns a dataframe containing slices of a certain interaction type (Duplicate slices are not dropped.)
    "Duplicate" slices references slices corresponding to the same neutrino interaction. In other words,
    allows "slice double counting." 
    
    Parameters
    ----------
    df: input dataframe
    int_type: int
        interaction type
        
    Returns
    -------
    slices_df: dataframe containing only slices of the specified interaction type
    """
    slices_df = df[df["signal"]==int_type]
    return slices_df

def calc_err(passed: list, total: list):
    """
    Calculates the Binomial Error for a pass/fail selection
    
    Parameters
    ----------
    passed: list of ints
        number of events in a bin that passed the selection
    total: list of ints
        original number of events in that bin
    
    Returns
    -------
    err: list of lists
        list of lower and upper errors for each bin
    """
    err = [[],[]]
    eff = passed/total
    for i in range(len(passed)):
        val_passed = passed[i]
        val_tot = total[i]
        interval = sm.stats.proportion.proportion_confint(val_passed,val_tot,method='wilson')
        err[0].append(abs(eff[i]-interval[0]))
        err[1].append(abs(eff[i]-interval[1]))
    return err