"""This module contains methods to read and save to file"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import yaml

from oogeso import dto
from oogeso.dto.serialisation import DataclassJSONDecoder

logger = logging.getLogger(__name__)


def read_data_from_yaml(
    filename: Path, profiles: Optional[Dict[str, float]] = None, profiles_nowcast: Optional[Dict[str, float]] = None
) -> dto.EnergySystemData:
    """
    Read input data from yaml file

    :param filename: The yaml file to read
    :param profiles: Overwrite the YAML profiles keyword
    :param profiles_nowcast: Overwrite the YAML profiles_nowcast keyword
    """
    with open(filename, "r") as text_file:
        data_dict = yaml.safe_load(text_file)

    if profiles is not None:
        if "profiles" in data_dict:
            logger.warning("Overiding profile data in yaml file")
        else:
            logger.debug("Adding profiles to data in yaml file")
        data_dict["profiles"] = profiles

    if profiles_nowcast is not None:
        if "profiles_nowcast" in data_dict:
            logger.warning("Overiding profile data in yaml file")
        else:
            logger.debug("Adding profiles to data in yaml file")
        data_dict["profiles_nowcast"] = profiles_nowcast

    json_str = json.dumps(data_dict)
    decoder = DataclassJSONDecoder()
    energy_system = decoder.decode(json_str)
    return energy_system


def read_profiles_from_parquet(filename: Union[Path, str], keys: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Read input data profiles from parquet file

    Reading from the given filename with _<key name><suffix>.
    This is to give legacy support for the hdf5 format where you have keys.
    """
    if not isinstance(filename, Path):
        filename = Path(filename)

    suffix = ""
    if "." in filename.name:
        suffix = filename.suffix

    profiles = {}
    for key in keys:
        profiles[key] = pd.read_parquet(f"{filename.with_suffix('')}_{key}{suffix}")
    return profiles


def save_profiles_to_parquet(filename: Union[Path, str], profiles: Dict[str, pd.DataFrame]) -> None:
    """
    Save profiles to parquet

    Writing to the given filename with _<key name><suffix>.
    This is to give legacy support for the hdf5 format where you have keys.
    """
    if not isinstance(filename, Path):
        filename = Path(filename)

    suffix = ""
    if "." in filename.name:
        suffix = filename.suffix

    for key in profiles:
        profiles[key].to_parquet(f"{filename.with_suffix('')}_{key}{suffix}")
    return


def read_profiles_from_xlsx(
    filename: Path,
    sheet_forecast: str = "profiles",
    sheet_nowcast: str = "profiles_forecast",
    exclude_cols: Optional[List] = None,
) -> Dict[str, pd.DataFrame]:
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

    return {"forecast": df_profiles_forecast, "nowcast": df_profiles_nowcast}
