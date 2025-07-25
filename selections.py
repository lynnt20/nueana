import pandas as pd
import numpy as np

def cutRecoVtxFV(df: pd.DataFrame, col:str="slc_vertex_",
                 xmin:float=0, xmax:float=190,
                 ymin:float=-190, ymax:float=190,
                 zmin:float=10, zmax:float=480):
    """
    Selects slices with reconstructed vertices inside the FV: x[0,190], y[-190,190], z[10,480].
    
    Parameters
    ----------
    df: input dataframe
    col: str
        column name of the reconstructed vertex position
    xmin: float
        minimum abs(x) value (default 0)
    xmax: float
        maximum abs(x) value (default 190)
    ymin: float
        minimum y value (default -190)
    ymax: float
        maximum y value (default 190)
    zmin: float
        minimum z value (default 10)
    zmax: float
        maximum z value (default 480)
    """
    maskRecoFV = ((abs(df[col+"x"]) > xmin)& (abs(df[col+"x"]) < xmax)
                    & (df[col+"y"] >  ymin)& (df[col+"y"] < ymax) 
                    & (df[col+"z"] >  zmin)& (df[col+"z"] < zmax))
    return df[maskRecoFV]

def cutNotCosmic(df: pd.DataFrame, col:str="slc_is_clear_cosmic"):
    """
    Selects slices that are not clear cosmics.
    
    Parameters
    ----------
    df: input dataframe
    col: str
        column name of the clear cosmic bool
    """
    maskNotCosmic = (df[col]==False)
    return df[maskNotCosmic]

def cutShower(df: pd.DataFrame, col:str="pfp_trackScore", score_cut:float=0.6):
    """
    Selects slices with at least one pfp with trackScore < 0.6. 
    This cut can be computationally intensive.
    Ideally perform this cut **after** cutRecoVtxFV and cutNotCosmic.

    Parameters
    ----------
    df: input dataframe
    col: str
        column name of the track score
    score_cut: float
        cut value for the track score (default 0.6)
    """
    score_df = df.query(col+" > 0") # require that we look at pfps with a sensible track score
    score_df = score_df[(score_df[col] == score_df.groupby(["ntuple","entry","rec.slc__index"])[col].transform("min"))]
    score_df = score_df.query("pfp_trackScore < @score_cut")[["ntuple","entry","rec.slc__index"]] # require that the minimum track score is < 0.6
    return df.merge(score_df,how="inner")

def cutShowerEnergy(df: pd.DataFrame, 
                    shw_col:str="shw_energy",
                    score_col:str="pfp_trackScore",
                    shw_cut_val:float=0.1,
                    score_cut_val:float=0.6):
    """
    Selects slices that have at least one pfp that fulfills the conditions: has a trackScore < 0.6 and reco
    shower energy greater than `cut_val` (in units of GeV).
    
    Parameters
    ----------
    df: input dataframe
    shw_col: str
        column name of the shower energy
    score_col: str
        column name of the track score
    shw_cut_val: float
        cut value for the shower energy
    score_cut_val: float
        cut value for the track score
        
    Returns
    -------
    df: dataframe with slices that have at least one pfp with trackScore < score_cut_val and shower energy > shw_cut_val.
    Returns the original dataframe with only those slices.
    """
    shw_df = df.query("(" + score_col+" < @score_cut_val ) & ("+score_col+" > 0)")
    max_df = shw_df[(shw_df[shw_col] == shw_df.groupby(["ntuple","entry","rec.slc__index"])[shw_col].transform(max))]
    max_df = max_df.query(shw_col+" > @shw_cut_val")
    slc_max_df = max_df[['ntuple','entry','rec.slc__index']].drop_duplicates()
    return df.merge(slc_max_df,how="inner",on=['ntuple','entry','rec.slc__index'])

def cutPreselection(df: pd.DataFrame, 
                    whereRecoVtxFV: bool=True, 
                    whereNotCosmic: bool=True, 
                    whereShower: bool=True):
    """
    Performs all presection cuts: reco vertex in FV, not clear cosmic, and at least one pfp with trackScore < 0.6.
    
    Parameters
    ----------
    df: input dataframe
    whereRecoVtxFV: bool
        if True, performs the reco vertex in FV cut
    whereNotCosmic: bool
        if True, performs the not clear cosmic cut
    whereShower: bool
        if True, performs the at least one pfp with trackScore < 0.6 cut
    
    Returns
    -------
    df: dataframe with preselection cuts applied
    """
    if whereRecoVtxFV:
        df = cutRecoVtxFV(df)
    if whereNotCosmic: 
        df = cutNotCosmic(df)
    if whereShower:
        df = cutShower(df)
    return df

def cutCRUMBS(df: pd.DataFrame, col:str="slc_crumbs_result_score", cut_val:float =-0.15):
    """
    Selects slices with CRUMBS score > -0.15 [or other cut value].
    
    Parameters
    ----------
    df: input dataframe
    col: str
        column name of the CRUMBS score
    cut_val: float
        cut value for the CRUMBS score
    """
    maskCRUMBS = (df[col] > cut_val)
    return df[maskCRUMBS]

def cutContainment(df: pd.DataFrame, whereTrkCont: bool=True, whereShwCont: bool=True, 
                   trk_col: str = "pfp_trk_end_", shw_col: str = "pfp_shw_end_"):
    """
    Selects slices with all pfps contained in the detector volume. Containment is defined as 5 cm within the detector volume.
    To decide whether to use the pfp_trk or pfp_shw end position, pfps with trackScore >= 0.5 are considered tracks, and pfps with trackScore < 0.5 are considered showers.
    
    Parameters
    ----------
    df: input dataframe
    whereTrkCont: bool
        if True, performs the track containment cut (uses pfp_trk_end)
    whereShwCont: bool
        if True, performs the shower containment cut (uses pfp_shw_end)
    trk_col: str
        column name of the track end position
    shw_col: str
        column name of the shower end position
    """
    df["shw_exit"] = 0 
    df["trk_exit"] = 0

    df["trk_exit"] = np.where(((df.pfp_trackScore >= 0.5) & 
                                ((abs(df[trk_col+"x"]) > 195) |
                                    (df[trk_col+"y"] < -195)  | (df[trk_col+"y"] > 195) &
                                    (df[trk_col+"z"] < 5)    | (df[trk_col+"z"] > 495))),
                                    1,df["trk_exit"])
    df["shw_exit"] = np.where(((df.pfp_trackScore < 0.5) & 
                                ((abs(df[shw_col+"x"]) > 195) |
                                    (df[shw_col+"y"] < -195)  | (df[shw_col+"y"] > 195) &
                                    (df[shw_col+"z"] < 5)    | (df[shw_col+"z"] > 495))),
                                    1,df["shw_exit"])
    # sum the number of exiting trks/shws 
    pfp_exit_df = df.groupby(["ntuple","entry","rec.slc__index"]).agg(ntrk_exit = ('trk_exit','sum'),
                                                                      nshw_exit = ('shw_exit','sum')).reset_index()
    # require that there are no exiting trks/shw if specified
    if (whereTrkCont): pfp_exit_df = pfp_exit_df.query("ntrk_exit==0")
    if (whereShwCont): pfp_exit_df = pfp_exit_df.query("nshw_exit==0")

    df = df.merge(pfp_exit_df[["ntuple","entry","rec.slc__index"]],how="right")
    df.drop(columns=["trk_exit","shw_exit"],inplace=True)
    return df