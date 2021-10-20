import oogeso
import oogeso.core.util
import oogeso.io.file_io
import pandas as pd
import pytest


def test_file_input():
    # Test oogeso object creation from yaml file

    profiles_dfs = oogeso.io.file_io.read_profiles_from_csv(
        filename_forecasts="examples/testcase2_profiles_forecasts.csv",
        filename_nowcasts="examples/testcase2_profiles_nowcasts.csv",
        timestamp_col="timestamp",
        exclude_cols=["timestep"],
    )
    profiles_json = oogeso.core.util.create_timeseriesdata(
        profiles_dfs["forecast"],
        profiles_dfs["nowcast"],
        time_start=None,
        time_end=None,
        timestep_minutes=15,
    )
    data0 = oogeso.io.file_io.read_data_from_yaml("examples/test case2.yaml")
    data0.profiles = profiles_json

    # If not failed above, it's OK
    assert True


def test_hdf_profiles():
    try:
        import tables
    except ImportError as e:
        print("Pytables not installed")
        pytest.skip("pytables not installed")

    profiles_dfs = oogeso.io.file_io.read_profiles_from_csv(
        filename_forecasts="examples/testcase2_profiles_forecasts.csv",
        filename_nowcasts="examples/testcase2_profiles_nowcasts.csv",
        timestamp_col="timestamp",
        exclude_cols=["timestep"],
    )
    oogeso.io.file_io.save_profiles_to_hd5(
        filename="tests/profiles.hd5", profiles=profiles_dfs
    )

    profiles_dfs2 = oogeso.io.file_io.read_profiles_from_hd5(
        filename="tests/profiles.hd5"
    )
    assert isinstance(profiles_dfs2["forecast"], pd.DataFrame)
    assert isinstance(profiles_dfs2["nowcast"], pd.DataFrame)
