from oogeso.dto.oogeso_input_data_objects import TimeSeriesData
from typing import List
import logging
import pandas as pd


def create_timeseriesdata(
    df_forecast: pd.DataFrame,
    df_nowcast: pd.DataFrame,
    time_start: str,
    time_end: str,
    timestep_minutes: int,
    resample_method: str = "linear",
) -> List[TimeSeriesData]:
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
        logging.warning("Profiles contain NA")

    # now, create Oogeso TimeSereriesData object
    profiles = []
    for col in df_new.columns:
        curve = col[1]
        list_data = list(df_new[("forecast", curve)])
        list_data_nowcast = None
        if ("nowcast", curve) in df_new.columns:
            list_data_nowcast = list(df_new[("nowcast", curve)])
        new_ts = TimeSeriesData(
            id=curve, data=list_data, data_nowcast=list_data_nowcast
        )
        profiles.append(new_ts)
    return profiles
