import os

import numpy as np
from astropy.table import Table
from src.Data import Data

from kndetect.utils import get_data_dir_path


def ztf_ob_type_name(type_no: int):
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


def load_ztf_data(head_path, phot_path):
    """
    loads the ztf test data. The way this file is created, separators (-777) are already dropped.

    Parameters
    ----------
    head_path: str
        path to head file
    phot_path: str
        path to phot file

    Returns
    -------
    object of the Data class
    """

    df_header = Table.read(head_path, format="fits")
    df_phot = Table.read(phot_path, format="fits")
    data_ob = Data(
        df_metadata=df_header,
        df_data=df_phot,
        object_id_col_name="SNID",
        time_col_name="MJD",
        target_col_name="SNTYPE",
        band_col_name="FLT",
        brightness_col_name="FLUXCAL",
        brightness_err_col_name="FLUXCALERR",
        bands=["g", "r"],
    )
    return data_ob


def load_RESSPECT_data(
    phot_df_file_path="/media/biswajit/drive/Kilonova_datasets/RESSPECT"
    "/RESSPECT_PERFECT_LIGHTCURVE.csv",
    meta_df_file_path="/media/biswajit/drive/Kilonova_datasets/RESSPECT/RESSPECT_PERFECT_HEAD.csv",
):
    """
    load RESSPECT simulations for generating PCs

    Parameters
    ----------
    phot_df_file_path: str
        path to data file
    meta_df_file_path: str
        path to header file

    Returns
    -------
    object of Data class
    """
    df_meta_data = Table.read(meta_df_file_path, delimiter=",")
    df_data = Table.read(phot_df_file_path)
    data_ob = Data(
        df_metadata=df_meta_data,
        df_data=df_data,
        object_id_col_name="SNID",
        time_col_name="MJD",
        band_col_name="FLT",
        brightness_col_name="FLUXCAL",
        brightness_err_col_name="FLUXCALERR",
        bands=["u", "g", "r", "i", "z", "Y"],
        target_col_name="type",
    )
    return data_ob


def get_pcs(num_pc_components, pcs_choice="interpolated", normalize_pcs=True):
    """
    function to load PCs

    Parameters
    ----------
    num_pc_components: int
        number of PCs to be used for predictions
    pcs_choice: string
        choice should be either 'interpolated' or 'non-interpolated' pcs.
    normalize_pcs: bool
        normalize pcs so that the maximum absolute value is equal to 1.
        By default, interpolated pcs are always normalized.

    Returns
    -------
    pcs_out: dict
        dictonary which holds pcs corresponding to each band.
    """

    assert (pcs_choice == "interpolated") | (
        pcs_choice == "non-interpolated"
    ), "wrong pcs choice. Choose either interpolated or non-interpolated"
    pc_out = {}

    data_dir = get_data_dir_path()

    if pcs_choice == "interpolated":
        pcs_load = np.load(
            os.path.join(data_dir, "interpolated_mixed_pcs.npy"), allow_pickle=True
        )
        return pcs_load

    if pcs_choice == "non-interpolated":

        pcs_load = np.load(
            os.path.join(data_dir, "mixed_pcs.npy"), allow_pickle=True
        ).item()
        band_choice = "all"

        pc_out = pcs_load[band_choice]
        if normalize_pcs:

            for component_number in range(num_pc_components):

                max_pc_val = np.amax(np.abs(pc_out[component_number]))
                pc_out[component_number] = pc_out[component_number] / max_pc_val

        return pc_out
