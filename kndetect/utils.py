import pandas as pd
import numpy as np

def load_pcs(fn, npcs):
    """ Load PC from disk into a Pandas DataFrame
    Parameters
    ----------
    fn: str
        Filename. This file should be known from all machines!
    npcs: int
        Number of principal components to load
    Return
    ----------
    pcs: pd.DataFrame
        All principal components to be considered.
        keys should be PCs names (1, 2, 3, ...),
        values their amplitude at each epoch in the grid.
        Order of PCs when calling pcs.keys() is important.
    """
    comp = pd.read_csv(fn)
    pcs = [] 
    for i in range(npcs):
        pcs.append(comp.iloc[i].values)

    pcs = np.array(pcs)
    return pcs