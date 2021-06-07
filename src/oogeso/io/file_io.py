"""This module contains methods to read and save to file"""

import yaml
import json
import logging
import pandas as pd
from oogeso.dto.oogeso_subset import EnergySystemData
from oogeso.dto.timeseries import TimeSeries
from ..dto import oogeso_input_data_objects as io_obj
from typing import Dict


def read_data_from_yaml(
    filename, profiles=None, profiles_nowcast=None
) -> EnergySystemData:
    """Read input data from yaml file"""
    data_dict = None
    with open(filename, "r") as text_file:
        data_dict = yaml.safe_load(text_file)

    if profiles is not None:
        logging.debug("Adding profiles to data in yaml file")
        if "profiles" in data_dict:
            logging.warning("Overiding profile data in yaml file")
        data_dict["profiles"] = profiles
    if profiles_nowcast is not None:
        logging.debug("Adding profiles to data in yaml file")
        if "profiles_nowcast" in data_dict:
            logging.warning("Overiding profile data in yaml file")
        data_dict["profiles_nowcast"] = profiles_nowcast
    json_str = json.dumps(data_dict)
    decoder = io_obj.DataclassJSONDecoder()
    energy_system = decoder.decode(json_str)
    return energy_system


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


def read_profiles_from_csv(
    filename_forecasts,
    filename_nowcasts=None,
    timestamp_col="timestamp",
    dayfirst=True,
    exclude_cols=None,
    json_indent=0,
):
    """Read input data profiles from CSV to a dicitonary of pandas dataframes"""
    if exclude_cols is None:
        exclude_cols = []
    df_profiles_forecast = pd.read_csv(
        filename_forecasts,
        index_col=timestamp_col,
        parse_dates=[timestamp_col],
        dayfirst=dayfirst,
        usecols=lambda col: col not in exclude_cols,
    )
    tseries_forecast: Dict[str, TimeSeries] = json.loads(
        df_profiles_forecast.to_json(indent=json_indent, date_format="iso")
    )
    tseries_nowcast = None
    if filename_nowcasts is not None:
        df_profiles_nowcast = pd.read_csv(
            filename_nowcasts,
            index_col=timestamp_col,
            parse_dates=[timestamp_col],
            dayfirst=dayfirst,
            usecols=lambda col: col not in exclude_cols,
        )
        tseries_nowcast: Dict[str, TimeSeries] = json.loads(
            df_profiles_nowcast.to_json(indent=json_indent, date_format="iso")
        )
    es_timeseries = {"profiles": tseries_forecast, "profiles_nowcast": tseries_nowcast}
    return es_timeseries
