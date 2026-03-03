"""File input/output utilities for loading HDF5 data files."""
import pandas as pd

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
        
def load_dfs(file, keys2load, n_max_concat=10):
    """Load DataFrames from split HDF5 file.
    
    Parameters
    ----------
    file : str
        Path to HDF5 file.
    keys2load : list
        List of key names to load from the file.
    n_max_concat : int, optional
        Maximum number of splits to concatenate (default: 10).
    
    Returns
    -------
    dict
        Dictionary mapping key names to concatenated DataFrames.
    """
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
    """Calculate total MC exposure information from a list of files.
    
    Parameters
    ----------
    file_list : list
        List of file paths to process.
    
    Returns
    -------
    tuple
        (ngates, pot, nevents) - total number of gates, POT, and events.
    """
    ngates = 0
    pot = 0
    nevents = 0
    for i, file in enumerate(file_list):
        out_df = load_dfs(file, ["hdr"])
        hdr_df = out_df["hdr"]
        ngates += hdr_df.reset_index().drop_duplicates(subset=['run','subrun'])['ngenevt'].sum()
        pot += hdr_df.reset_index().pot.sum()
        nevents += len(hdr_df)
    return ngates, pot, nevents
