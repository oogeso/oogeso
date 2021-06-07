from oogeso.dto.timeseries import TimeSeries
from typing import Dict
import logging
import pandas as pd


def reshape_timeseries(
    profiles: Dict[str, TimeSeries],
    profiles_nowcast: Dict[str, TimeSeries],
    start_time: str,
    timestep_minutes: int,
) -> Dict[str, pd.DataFrame]:
    """Resample and rearrange time series profiles to the pandas DataFrame format used internally

    Parameters
    ----------
    profiles : dict
        profiles
    profiles_nowcast : dict
        profiles - near real-time updated profile forecast
    start_time : str
        iso datetime string to represent timestep 0
    timestep_minutes: int
        timestep interval in minutes in output data
    """
    df = pd.DataFrame.from_dict(profiles, orient="columns")
    df.index = pd.to_datetime(df.index)
    incuding_nowcast = False
    if profiles_nowcast is not None:
        including_nowcast = True
        df_nc = pd.DataFrame.from_dict(profiles_nowcast, orient="columns")
        df_nc.index = pd.to_datetime(df_nc.index)
        # give them names so they can be split later
        df_nc.columns = "nowcast:" + df_nc.columns
        df.columns = "forecast:" + df.columns
        df = pd.concat([df, df_nc], axis=1)

    resampled = df.resample(
        rule="{}T".format(
            timestep_minutes,
        ),
        closed="left",
        label="left",
    )
    df = resampled.mean()

    # If up-sampling, NaN values filed by linear interpolation
    # If down-sampling, this does nothing
    df = df.interpolate(method="linear")

    # Select the times of interest
    mask = df.index >= start_time
    df = df[mask]

    cols_actual = df.columns[df.columns.str.startswith("nowcast:")]
    cols_forecast = df.columns[df.columns.str.startswith("forecast:")]
    if not (cols_actual | cols_forecast).all():
        logging.warning("Profiles should be named 'nowcast:...' or 'forecast:...'")

    prof = {}
    prof["forecast"] = df[cols_forecast].copy()
    prof["actual"] = None
    if including_nowcast:
        # split nowcast(actual)/forecast profiles:
        prof["forecast"].columns = prof["forecast"].columns.str[9:]
        prof["actual"] = df[cols_actual].copy()
        prof["actual"].columns = prof["actual"].columns.str[8:]

    return prof
