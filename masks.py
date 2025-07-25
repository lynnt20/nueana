import pandas as pd

def maskTrueVtxFv(nuprim_df: pd.DataFrame, 
                  xmin: float = 5, xmax: float = 180, 
                  ymin: float = -180, ymax: float = 180, 
                  zmin: float = 20, zmax: float = 470):
    """
    Returns a mask for true vertex inside the FV: x[5,180], y[-180,180], z[20,470].
    Intended to be used on the multi-index nuprim dataframe.
    """
    whereFV = ((abs(nuprim_df.position.x) > xmin)& (abs(nuprim_df.position.x) < xmax)
                & (nuprim_df.position.y > ymin)  & (nuprim_df.position.y < ymax)
                & (nuprim_df.position.z > zmin)  & (nuprim_df.position.z < zmax))
    return whereFV