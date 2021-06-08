from oogeso.dto.oogeso_input_data_objects import TimeSeriesData
from typing import List, Dict
import logging
import pandas as pd


def reshape_timeseries(
    profiles: List[TimeSeriesData],
    time_start: str,
    time_end: str,
    timestep_minutes: int,
) -> Dict[str, pd.DataFrame]:
    """Resample and rearrange time series profiles to the pandas DataFrame format used internally

    Parameters
    ----------
    profiles : list
        list of profiles
    start_time : str
        iso datetime string
    end_time : str
        iso datetime string
    timestep_minutes: int
        timestep interval in minutes in output data
    """
    dfs = []
    for prof in profiles:
        logging.debug("Processing timeseries %s", prof.id)
        df_forecast = pd.DataFrame.from_dict(
            prof.data, columns=["forecast:{}".format(prof.id)], orient="index"
        )
        # return df_forecast
        dfs.append(df_forecast)
        if prof.data_nowcast is not None:
            df_nowcast = pd.DataFrame.from_dict(
                prof.data_nowcast,
                columns=["nowcast:{}".format(prof.id)],
                orient="index",
            )
            dfs.append(df_nowcast)
    df_orig = pd.concat(dfs, axis=1)
    df_orig.index = pd.to_datetime(df_orig.index)
    resampled = df_orig.resample(
        rule="{}T".format(
            timestep_minutes,
        ),
        closed="left",
        label="left",
    )
    df_new = resampled.mean()

    # If up-sampling, NaN values filed by linear interpolation
    # If down-sampling, this does nothing
    df_new = df_new.interpolate(method="linear")

    # Select the times of interest
    if time_start is not None:
        mask = df_new.index >= time_start
        df_new = df_new[mask]
    if time_end is not None:
        mask = df_new.index <= time_end
        df_new = df_new[mask]
    if df_new.isna().any().any():
        logging.warning("Profiles contain NA")

    cols_actual = df_new.columns[df_new.columns.str.startswith("nowcast:")]
    cols_forecast = df_new.columns[df_new.columns.str.startswith("forecast:")]
    if not (cols_actual | cols_forecast).all():
        logging.warning("Profiles should be named 'nowcast:...' or 'forecast:...'")

    # integer timesteps as index:
    timestamps = df_new.index
    df_new = df_new.reset_index()

    prof = {}
    prof["forecast"] = df_new[cols_forecast].copy()
    prof["forecast"].columns = prof["forecast"].columns.str[9:]
    prof["actual"] = df_new[cols_actual].copy()
    prof["actual"].columns = prof["actual"].columns.str[8:]
    prof["forecast"]["TIMESTAMP"] = timestamps
    prof["actual"]["TIMESTAMP"] = timestamps

    return prof


def timeseries_to_dataframes(profiles):
    """Convert timeseries to dataframes used internally"""
    pass


# TODO: decide where to convert between time-series with timestamps to uniform lists
# Inside "core" simulator, or outside
# i.e. should timeseries data within the data transfer object be with profiles as list of numbers, or time-value pairs
