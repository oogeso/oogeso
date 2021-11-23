from pathlib import Path

import oogeso
import oogeso.utils
import oogeso.io
import pandas as pd
import pytest
import tempfile


EXAMPLE_DATA_ROOT_PATH = Path(__file__).parent.parent / "examples"
TEST_DATA_ROOT_PATH = Path(__file__).parent


def test_file_input():
    # Test oogeso object creation from yaml file

    profiles_dfs = oogeso.io.file_io.read_profiles_from_csv(
        filename_forecasts=EXAMPLE_DATA_ROOT_PATH / "testcase2_profiles_forecasts.csv",
        filename_nowcasts=EXAMPLE_DATA_ROOT_PATH / "testcase2_profiles_nowcasts.csv",
        timestamp_col="timestamp",
        exclude_cols=["timestep"],
    )
    profiles_json = oogeso.utils.create_timeseriesdata(
        profiles_dfs["forecast"],
        profiles_dfs["nowcast"],
        time_start="",
        time_end="",
        timestep_minutes=15,
    )
    data0 = oogeso.io.read_data_from_yaml(EXAMPLE_DATA_ROOT_PATH / "test case2.yaml")
    data0.profiles = profiles_json

    # If not failed above, it's OK
    assert True


def test_write_read_parquet():
    idx = pd.date_range(start="2020-01-01", end="2021-01-01", freq="D")

    profiles = {
        "forecast": pd.DataFrame(index=idx, columns=["forecast"], data=1),
        "actual": pd.DataFrame(index=idx, columns=["actual"], data=3),

    }

    filename = tempfile.mkstemp(suffix=".parquet")[1]
    oogeso.io.save_profiles_to_parquet(
        filename=filename,
        profiles=profiles
    )

    profiles_out = oogeso.io.read_profiles_from_parquet(
        filename=filename,
        keys=["actual", "forecast"]
    )

    for key in ["actual", "forecast"]:
        # Asserting that the DataFrames are indeed equal after a round-trip to Parquet.
        assert profiles[key].equals(profiles_out[key])


def test_parquet_profiles():

    profiles_dfs = oogeso.io.read_profiles_from_csv(
        filename_forecasts=EXAMPLE_DATA_ROOT_PATH / "testcase2_profiles_forecasts.csv",
        filename_nowcasts=EXAMPLE_DATA_ROOT_PATH / "testcase2_profiles_nowcasts.csv",
        timestamp_col="timestamp",
        exclude_cols=["timestep"],
    )

    tmp_file = tempfile.mkstemp(suffix=".parquet")[1]

    oogeso.io.file_io.save_profiles_to_parquet(
        filename=tmp_file, profiles=profiles_dfs
    )

    profiles_dfs2 = oogeso.io.file_io.read_profiles_from_parquet(
        filename=tmp_file, keys=["forecast", "nowcast"]
    )
    assert isinstance(profiles_dfs2["forecast"], pd.DataFrame)
    assert isinstance(profiles_dfs2["nowcast"], pd.DataFrame)
