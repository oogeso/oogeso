"""This module contains methods to read and save to file"""

import yaml
import json
import logging
import pandas as pd
from typing import Optional, Union
from oogeso.dto import EnergySystemData
from oogeso.dto.serialisation import DataclassJSONDecoder
from pathlib import Path

# from oogeso.dto.oogeso_input_data_objects import (
#    EnergySystemData,
#    DataclassJSONDecoder,
# )
from typing import Dict, List

logger = logging.getLogger(__name__)


def read_data_from_yaml(
    filename, profiles=None, profiles_nowcast=None
) -> EnergySystemData:
    """Read input data from yaml file"""
    data_dict = None
    with open(filename, "r") as text_file:
        data_dict = yaml.safe_load(text_file)

    if profiles is not None:
        logger.debug("Adding profiles to data in yaml file")
        if "profiles" in data_dict:
            logger.warning("Overiding profile data in yaml file")
        data_dict["profiles"] = profiles
    if profiles_nowcast is not None:
        logger.debug("Adding profiles to data in yaml file")
        if "profiles_nowcast" in data_dict:
            logger.warning("Overiding profile data in yaml file")
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

    Fixme: Refactor to avoid this complexity if not needed.
    """
    if not isinstance(filename, Path):
        filename =Path(filename)

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

    Fixme: Refactor to avoid this complexity if not needed.
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
    sheet_forecast: str ="profiles",
    sheet_nowcast: str ="profiles_forecast",
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
