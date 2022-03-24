import os

import numpy as np
import pandas as pd


def get_data_dir_path():
    """Function to return path to the data folder of kndetect

    Returns
    -------
    data_dir: str
        path to data folder
    """
    curdir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(curdir, "data")

    return data_dir


def load_pcs(fn=None, npcs=3):
    """Load PC from disk into a Pandas DataFrame

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
        data_dir = get_data_dir_path()
        fn = os.path.join(data_dir, "mixed_pcs.csv")

    comp = pd.read_csv(fn)
    pcs = []
    for i in range(npcs):
        pcs.append(comp.iloc[i].values)

    pcs = np.array(pcs)
    return pcs


def snana_ob_type_name(type_no: int):
    """
    Retuens the type name in string for the type numbers in the ZTF dataset.

    Parameters
    ----------
    type_no: int
        type number whose corresponding string is to be fetched.

    Returns
    -------
    str: String with the type name.
    """
    if type_no == 141:
        return "141: 91BG"
    if type_no == 143:
        return "143: Iax"
    if type_no == 145:
        return "145: point Ia"
    if type_no == 149:
        return "149: KN GRANDMA"
    if type_no == 150:
        return "150: KN GW170817"
    if type_no == 151:
        return "151: KN Karsen 2017"
    if type_no == 160:
        return "160: Superluminous SN"
    if type_no == 161:
        return "161: pair instability SN"
    if type_no == 162:
        return "162: ILOT"
    if type_no == 163:
        return "163: CART"
    if type_no == 164:
        return "164: TDE"
    if type_no == 170:
        return "170: AGN"
    if type_no == 180:
        return "180: RRLyrae"
    if type_no == 181:
        return "M 181: dwarf_flares"
    if type_no == 183:
        return "183: PHOEBE"
    if type_no == 190:
        return "190: uLens_BSR"
    if type_no == 191:
        return "191: uLens_Bachelet"
    if type_no == 192:
        return "192: uLens_STRING"
    if type_no == 114:
        return "114: MOSFIT-IIn"
    if type_no == 113:
        return "113: Core collapse Type II pca"
    if type_no == 112:
        return "112: Core collapse Type II"
    if type_no == 102:
        return "102: MOSFIT-Ibc"
    if type_no == 103:
        return "103: Core collapse Type Ibc"
    if type_no == 101:
        return "101: Ia SN"
    if type_no == 0:
        return "0: Unknown"


def get_event_type(
    key, meta_df, meta_key_col_name, meta_type_col_name, fetch_type_name=False
):
    """Function to get the event type of an object.

    Parameters
    ----------
    key: int or list
        object_id for which event type is to be fetched.
        If a list is provided, the values are returned in an array in the same order.
    df_meta: pd.DataFrame
        DataFrame containing columns with names given by`type_col_name` and `meta_key_col_name`
    meta_key_col_name: str
        column name against which keys are to be matched
    meta_type_col_name: str
        column name for event type, in df_meta
    fetch_type_name: bool
        To idtentify id the event type name corresponding to event types are to be returned.

    Returns
    -------
    event_types: list
        list of event types. corresponding to each key.
    event_type_names: str (optional)
        returuns the name of snana event types if fetch_type_name is set to true.
    """
    # drop index?
    if not isinstance(key, list):
        key = [key]

    event_types = []
    event_names = []

    for id in key:
        event_type = (
            meta_df[meta_type_col_name].loc[meta_df[meta_key_col_name] == id].values[0]
        )
        event_types.append(event_type)
        event_names.append(snana_ob_type_name(event_type))

    if fetch_type_name:
        return event_types, event_names

    return event_types
