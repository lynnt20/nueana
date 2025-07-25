import pandas as pd
import numpy as np

def shw_energy_fix(df: pd.DataFrame, fix_dEdx: bool=False):
    """
    Fixes the shower energy column in the dataframe.
    The best plane is the plane with the greatest number of hits and positive reconstructed energy. 
    
    Parameters
    ----------
    df: input dataframe
    
    Returns
    -------
    shw_df: dataframe with fixed shower energy column
    """
    
    shw_df = df.copy()
    col_nhits2  = 'shw_plane_I2_nHits'
    col_nhits1  = 'shw_plane_I1_nHits'
    col_nhits0  = 'shw_plane_I0_nHits'
    col_energy2 = 'shw_plane_I2_energy'
    col_energy1 = 'shw_plane_I1_energy'
    col_energy0 = 'shw_plane_I0_energy'
    col_dEdx2 = 'shw_plane_I2_dEdx'
    col_dEdx1 = 'shw_plane_I1_dEdx'
    col_dEdx0 = 'shw_plane_I0_dEdx'
    
    if "pfp_shw_plane_I2_nHits" in df.columns:
        col_nhits2 = 'pfp_' + col_nhits2
        col_nhits1 = 'pfp_' + col_nhits1
        col_nhits0 = 'pfp_' + col_nhits0
        col_energy2 = 'pfp_' + col_energy2
        col_energy1 = 'pfp_' + col_energy1
        col_energy0 = 'pfp_' + col_energy0
    if "pfp_shw_plane_I2_dEdx" in df.columns:
        col_dEdx2 = 'pfp_' + col_dEdx2
        col_dEdx1 = 'pfp_' + col_dEdx1
        col_dEdx0 = 'pfp_' + col_dEdx0        
    elif "shw_plane_I2_nHits" not in df.columns:
        print("Error: Dataframe does not contain shower energy columns")
        return df

    nhits2 = ((shw_df[col_nhits2] >= shw_df[col_nhits1]) & (shw_df[col_nhits2]>= shw_df[col_nhits0]))
    nhits1 = ((shw_df[col_nhits1] >= shw_df[col_nhits2]) & (shw_df[col_nhits1]>= shw_df[col_nhits0]))
    nhits0 = ((shw_df[col_nhits0] >= shw_df[col_nhits2]) & (shw_df[col_nhits0]>= shw_df[col_nhits1]))

    # if energy[plane] is positive
    energy2 = (shw_df[col_energy2] > 0 )
    energy1 = (shw_df[col_energy1] > 0 )
    energy0 = (shw_df[col_energy0] > 0 )

    conditions = [(nhits2 & energy2),
                (nhits1 & energy1),
                (nhits0 & energy0),
                (((nhits2 & energy2)== False) & (energy1) & (shw_df[col_nhits1]>= shw_df[col_nhits0])), # if 2 is invalid, and 1 is positive and 1>0, go with 1 
                (((nhits2 & energy2)== False) & (energy0) & (shw_df[col_nhits0]>= shw_df[col_nhits1])), # if 2 is invalid, and 0 is positive and 0>1, go with 0
                (((nhits1 & energy1)== False) & (energy2) & (shw_df[col_nhits2]>= shw_df[col_nhits0])), # if 1 is invalid, and 2 is positive and 2>0, go with 2 
                (((nhits1 & energy1)== False) & (energy0) & (shw_df[col_nhits0]>= shw_df[col_nhits2])), # if 1 is invalid, and 0 is positive and 0>2, go with 0
                (((nhits0 & energy0)== False) & (energy2) & (shw_df[col_nhits2]>= shw_df[col_nhits1])), # if 0 is invalid, and 2 is positive and 2>1, go with 2              
                (((nhits0 & energy0)== False) & (energy1) & (shw_df[col_nhits1]>= shw_df[col_nhits2])), # if 0 is invalid, and 1 is positive and 1>2, go with 1 
                ((shw_df[col_nhits2]==-5) & (shw_df[col_nhits1]==-5) & (shw_df[col_nhits0]==-5))]
    shw_energy_choices = [ shw_df[col_energy2],
                    shw_df[col_energy1],
                    shw_df[col_energy0],
                    shw_df[col_energy1],
                    shw_df[col_energy0],
                    shw_df[col_energy2],
                    shw_df[col_energy0],
                    shw_df[col_energy2],
                    shw_df[col_energy1],
                    -1]
    shw_plane_choices = [2,1,0,1,0,2,0,2,1,-1]
    if "pfp_shw_plane_I2_dEdx" in df.columns:
        shw_dEdx_choices = [shw_df[col_dEdx2],
                            shw_df[col_dEdx1],
                            shw_df[col_dEdx0],
                            shw_df[col_dEdx1],
                            shw_df[col_dEdx0],
                            shw_df[col_dEdx2],
                            shw_df[col_dEdx0],
                            shw_df[col_dEdx2],
                            shw_df[col_dEdx1],
                            -1]

    shw_df['shw_energy'] = np.select(conditions, shw_energy_choices, default = -1)
    shw_df["shw_plane"] = np.select(conditions, shw_plane_choices, default = -1)
    if ("pfp_shw_plane_I2_dEdx" in df.columns) & (fix_dEdx):
        shw_df["shw_dEdx"] = np.select(conditions, shw_dEdx_choices, default = -1)
        shw_df.drop(columns=[col_dEdx0, col_dEdx1, col_dEdx2],inplace=True)
    shw_df.drop(columns=[col_nhits1, col_nhits2, col_nhits0, col_energy0, col_energy1, col_energy2,],inplace=True)
    return shw_df
