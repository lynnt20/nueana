"""File input/output utilities for loading HDF5 data files."""
import pandas as pd

__all__ = ['get_n_split', 'print_keys', 'load_dfs']

# credit for first three functions to Mun! 
def get_n_split(file):
    """Get the number of splits in an HDF5 file.
    
    Parameters
    ----------
    file : str
        Path to HDF5 file.
    
    Returns
    -------
    int
        Number of splits in the file.
    """
    this_split_df = pd.read_hdf(file, key="split")
    this_n_split = this_split_df.n_split.iloc[0]
    return this_n_split

def print_keys(file):
    """Print all keys available in an HDF5 file.
    
    Parameters
    ----------
    file : str
        Path to HDF5 file.
    """
    with pd.HDFStore(file, mode='r') as store:
        keys = store.keys()       # list of all keys in the file
        print("Keys:", keys)
        
def load_dfs(file, keys2load, n_max_concat=10, start_split=0):
    """Load DataFrames from split HDF5 file.
    
    Parameters
    ----------
    file : str
        Path to HDF5 file.
    keys2load : list
        List of key names to load from the file.
    n_max_concat : int, optional
        Maximum number of splits to concatenate (default: 10).
    start_split : int, optional
        Starting split index to load from (default: 0).
    
    Returns
    -------
    dict
        Dictionary mapping key names to concatenated DataFrames.
    """
    out_df_dict = {}
    this_n_keys = get_n_split(file) - start_split
    n_concat = min(n_max_concat, this_n_keys)
    for key in keys2load:
        dfs = []  # collect all splits for this key
        for i in range(start_split, start_split + n_concat):
            this_df = pd.read_hdf(file, key=f"{key}_{i}")
            dfs.append(this_df)
        out_df_dict[key] = pd.concat(dfs, ignore_index=False)
    return out_df_dict