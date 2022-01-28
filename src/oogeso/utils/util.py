import logging
from typing import List, Optional

import pandas as pd

from oogeso import dto

logger = logging.getLogger(__name__)


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
