"""This module contains methods to read and save to file"""

import yaml
import json
import logging
import pandas as pd
from oogeso.dto.oogeso_input_data_objects import (
    EnergySystemData,
    TimeSeriesData,
    DataclassJSONDecoder,
)
from typing import Dict, List
import datetime


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
    decoder = DataclassJSONDecoder()
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
    df_profiles_nowcast = pd.DataFrame()  # empty dataframe
    if filename_nowcasts is not None:
        df_profiles_nowcast = pd.read_csv(
            filename_nowcasts,
            index_col=timestamp_col,
            parse_dates=[timestamp_col],
            dayfirst=dayfirst,
            usecols=lambda col: col not in exclude_cols,
        )

    # change timestamp to iso string:
    isofromat = "%Y-%m-%dT%H:%M:%SZ"
    df_profiles_forecast.index = df_profiles_forecast.index.map(
        lambda x: datetime.datetime.strftime(x, isofromat)
    )
    df_profiles_nowcast.index = df_profiles_nowcast.index.map(
        lambda x: datetime.datetime.strftime(x, isofromat)
    )

    dict_forecasts = df_profiles_forecast.to_dict()
    dict_nowcasts = df_profiles_nowcast.to_dict()

    es_timeseries = []
    for col in df_profiles_forecast.columns:
        forecast = dict_forecasts[col]
        if col in dict_nowcasts:
            nowcast = dict_nowcasts[col]
        else:
            nowcast = None
        this_timeseries = TimeSeriesData(id=col, data=forecast, data_nowcast=nowcast)
        es_timeseries.append(this_timeseries)
    return es_timeseries
