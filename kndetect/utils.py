import pandas as pd
import numpy as np
import os

def load_pcs(fn=None, npcs=3):
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

    if fn is None: 
        curdir = os.path.dirname(os.path.abspath(__file__))
        fn = os.path.join(curdir, 'data/mixed_pcs.csv')

    comp = pd.read_csv(fn)
    pcs = [] 
    for i in range(npcs):
        pcs.append(comp.iloc[i].values)

    pcs = np.array(pcs)
    return pcs