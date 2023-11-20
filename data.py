"""
Provides API for data reading
"""
import pandas as pd
import re
import numpy as np

MAC_RE = (
    "^([0-9A-Fa-f]{2}[:-])"
    + "{5}([0-9A-Fa-f]{2})|"
    + "([0-9a-fA-F]{4}\\."
    + "[0-9a-fA-F]{4}\\."
    + "[0-9a-fA-F]{4})$"
)


def get_visible_waps(records: pd.DataFrame, missing_val=-100, wap_re=MAC_RE):
    """
    list of visible waps
    records: for a given df; any number of rows
    """
    # extract col names
    wap_cols = []
    for col in records.columns:
        if re.match(wap_re, col):
            wap_cols.append(col)
    records = records[wap_cols]

    return (
        records.replace(to_replace=missing_val, value=np.nan)
        .dropna(axis=1, how="all")
        .columns
    )


def label2coords_builder(arr: np.array, scale=1):
    """
    Array such that first column is label, and others are coords
    """

    lbl2coords = {}

    for row in arr:
        lbl2coords[int(row[0])] = (
            np.array(
                [
                    *row[1:],
                ]
            )
            / scale
        )

    return lbl2coords


class Floorplan:
    BASEMENT = "engr0"
    OFFICE = "engr1"
    GLOVER = "glover"
    LIBSTUDY = "libstudy"
    SCIENCES = "sciences"
    PATHS = [
        "engr0",
        "engr1",
        "glover",
        "libstudy",
        "sciences",
    ]

    SCALE = {
        "engr0": 14.2,
        "engr1": 17,
        "glover": 19,
        "libstudy": 17.5,
        "sciences": 13.7,
    }


class Devices:
    """
    Device names as given under raw folder
    """

    blu = "BLU"
    htc = "HTC"
    lg = "LG"
    moto = "MOTO"
    op3 = "OP3"
    s7 = "S7"
    devices = ["S7", "BLU", "HTC", "LG", "MOTO", "OP3"]


def build_dataset(
    train_dev: str,
    floorplan: str,
    base_path="EPIC_Lab_Data/Data",
) -> pd.DataFrame:
    """get dataframe for given device and location

    Parameters
    ----------
    train_dev : str
        training device tag as given under Devices class
    floorplan : str
        floorplan tag as given under Floorplan class
    base_path : str, optional
        the base path of data w.r.t. this function call, by default "maril/Data"

    Returns
    -------
    pandas.DataFrame
        dataframe with MAC ID columns, their respective 
        RSSI fingerprints at associated locations
    """

    # split train test
    (train_df, test_df) = (
        pd.read_csv(f"EPIC_Lab_Data/Data/train/{train_dev}_{floorplan}.csv"),
        pd.read_csv(f"EPIC_Lab_Data/Data/test/{train_dev}_{floorplan}.csv"),
    )

    # get mac IDs
    macs = get_visible_waps(train_df, missing_val=-100, wap_re=MAC_RE)

    # dict mapping label to scaled coordinates
    lbl2cords = label2coords_builder(
        train_df[["label", "x", "y"]].values, scale=Floorplan.SCALE[floorplan]
    )

    # scale input values
    train_df[macs] = (train_df[macs] + 100) / 100
    test_df[macs] = (test_df[macs] + 100) / 100

    return train_df, test_df, macs, lbl2cords
