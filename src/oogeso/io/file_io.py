"""This module contains methods to read and save to file"""

import yaml
import pandas as pd
import numpy as np
import logging
#from oogeso.core import electricalsystem
#from core import milp_compute


def _convert_xls_input(df, columns, index_col="id"):
    """Convert from XLSX format input to flat DataFrame"""
    df[columns] = df[columns].fillna(method="ffill")
    if index_col is None:
        df.reset_index(drop=True)

    df2 = df[[index_col, "param_id", "param_value"]].set_index([index_col, "param_id"])
    df2 = df2.squeeze().unstack()
    df2 = df2.dropna(axis=1, how="all")
    df4 = df[columns].set_index(index_col)
    # drop duplicate (indices):
    df4 = df4.loc[~df4.index.duplicated(keep="first")]
    df5 = df4.join(df2)
    return df5


def read_data_from_yaml(filename):
    """Read input data from yaml file"""
    data_dict = None
    with open(filename, "r") as text_file:
        data_dict = yaml.safe_load(text_file)
    # data = create_initdata(data_dict=data_dict)
    return data_dict


def read_profiles_from_hd5(filename, key_actual="actual", key_forecast="forecast"):
    """Read input data profiles from HDF5 file"""
    profiles = {}
    profiles["actual"] = pd.read_hdf(filename, key=key_actual)
    profiles["forecast"] = pd.read_hdf(filename, key=key_forecast)
    return profiles


def save_profiles_to_hd5(filename, profiles):
    """Save profiles to HDF5"""
    for k in profiles:
        profiles[k].to_hdf(filename, key=k, mode="a")
    return


def read_profiles_from_xlsx(
    filename,
    sheet_forecast="profiles",
    sheet_nowcast="profiles_forecast",
    exclude_cols=[],
):
    """Read input data profiles from XLSX to a dicitonary of pandas dataframes"""
    df_profiles = pd.read_excel(
        filename,
        sheet_name=sheet_forecast,
        index_col="timestep",
        usecols=lambda col: col not in exclude_cols,
    )
    df_profiles_forecast = pd.read_excel(
        filename,
        sheet_name=sheet_nowcast,
        index_col="timestep",
        usecols=lambda col: col not in exclude_cols,
    )
    profiles = {"actual": df_profiles, "forecast": df_profiles_forecast}
    return profiles
