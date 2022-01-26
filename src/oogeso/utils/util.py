import logging
from typing import Callable, List, Optional

import pandas as pd

from oogeso import dto
from oogeso.core import devices, networks

logger = logging.getLogger(__name__)


def get_network_from_carrier_name(carrier_name: str) -> Callable:
    map_carrier_to_model = {
        "el": networks.ElNetwork,
        "gas": networks.GasNetwork,
        "oil": networks.OilNetwork,
        "water": networks.WaterNetwork,
        "wellstream": networks.WellStreamNetwork,
        "heat": networks.HeatNetwork,
        "hydrogen": networks.HydrogenNetwork,
    }
    if not carrier_name.lower() in map_carrier_to_model.keys():
        raise NotImplementedError(f"Carrier: {carrier_name} has not implemented a corresponding Network mapping")
    return map_carrier_to_model[carrier_name.lower()]


def get_device_from_model_name(model_name: str) -> Callable:
    map_device_name_to_class = {
        "powersource": devices.PowerSource,
        "powersink": devices.PowerSink,
        "storageel": devices.StorageEl,
        "compressorel": devices.CompressorEl,
        "compressorgas": devices.CompressorGas,
        "electrolyser": devices.Electrolyser,
        "fuelcell": devices.FuelCell,
        "gasheater": devices.GasHeater,
        "gasturbine": devices.GasTurbine,
        "heatpump": devices.HeatPump,
        "pumpoil": devices.PumpOil,
        "pumpwater": devices.PumpWater,
        "separator": devices.Separator,
        "separator2": devices.Separator2,
        "sinkel": devices.SinkEl,
        "sinkheat": devices.SinkHeat,
        "sinkgas": devices.SinkGas,
        "sinkoil": devices.SinkOil,
        "sinkwater": devices.SinkWater,
        "sourceel": devices.SourceEl,
        "sourcegas": devices.SourceGas,
        "sourceoil": devices.SourceOil,
        "sourcewater": devices.SourceWater,
        "storagehydrogen": devices.StorageHydrogen,
        "wellgaslift": devices.WellGasLift,
        "wellproduction": devices.WellProduction,
    }
    if model_name in map_device_name_to_class:
        return map_device_name_to_class[model_name]
    else:
        raise NotImplementedError(f"Device {model_name} has not been implemented.")


def get_class_from_dto(class_str: str) -> Callable:
    """
    Search dto module for a callable that matches the signature given as class str

    Fixme: Replace this with a better division between DTO and core.
    """
    if class_str in dto.__dict__.keys():
        return dto.__dict__[class_str]
    elif class_str.lower() in [x.lower() for x in dto.__dict__.keys()]:
        return [v for k, v in dto.__dict__.items() if k.lower() == class_str.lower()][0]
    elif class_str.lower().replace("_", "") in [x.lower() for x in dto.__dict__.keys()]:
        return [v for k, v in dto.__dict__.items() if k.lower() == class_str.lower().replace("_", "")][0]
    else:
        raise NotImplementedError(f"Model {class_str} has not been implemented.")


def create_time_series_data(
    df_forecast: pd.DataFrame,
    df_nowcast: pd.DataFrame,
    time_start: Optional[str],
    time_end: Optional[str],
    timestep_minutes: int,
    resample_method: str = "linear",
) -> List[dto.TimeSeriesData]:
    """Rearrange and resample pandas timeseries to Oogeso data transfer object

    The input dataframes should have a datetime index
    """

    # Compine forecast and nowcast timeseries into a single dataframe
    df_orig = pd.concat({"forecast": df_forecast, "nowcast": df_nowcast}, axis=1)
    # 1 resample
    resampled = df_orig.resample(
        rule="{}T".format(
            timestep_minutes,
        ),
        closed="left",
        label="left",
    )
    df_new = resampled.mean()

    # If up-sampling, NaN values filed by linear interpolation (or other method)
    # If down-sampling, this does nothing
    df_new = df_new.interpolate(method=resample_method)

    # Select the times of interest
    if time_start is not None:
        mask = df_new.index >= time_start
        df_new = df_new[mask]
    if time_end is not None:
        mask = df_new.index <= time_end
        df_new = df_new[mask]
    if df_new.isna().any().any():
        logger.warning("Profiles contain NA")

    # now, create Oogeso TimeSereriesData object
    profiles = []
    for col in df_new.columns:
        if col[0] == "forecast":
            curve = col[1]
            list_data = df_new.loc[:, ("forecast", curve)].to_list()
            list_data_nowcast = None
            if ("nowcast", curve) in df_new.columns:
                list_data_nowcast = df_new.loc[:, ("nowcast", curve)].to_list()
            new_ts = dto.TimeSeriesData(id=curve, data=list_data, data_nowcast=list_data_nowcast)
            profiles.append(new_ts)
        elif col[0] == "nowcast":
            if ("forecast", col[1]) not in df_new.columns:
                logger.warning(f"Nowcast but no forecast profile for {col[1]}")

    return profiles