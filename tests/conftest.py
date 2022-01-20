import json
from pathlib import Path

import pytest

import oogeso
import oogeso.dto.serialisation
import oogeso.io
from oogeso.utils.util import create_time_series_data

EXAMPLE_DATA_ROOT_PATH = Path(__file__).parent.parent / "examples"
TEST_DATA_ROOT_PATH = Path(__file__).parent / "test_data"


@pytest.fixture
def testcase1_data() -> oogeso.dto.EnergySystemData:
    """
    Simple test case, electric only
    """
    return oogeso.io.read_data_from_yaml(TEST_DATA_ROOT_PATH / "testdata1.yaml")


@pytest.fixture
def testcase1_expected_result() -> oogeso.dto.EnergySystemData:
    """Expected results for the testcase 2."""
    with open(TEST_DATA_ROOT_PATH / "testdata1_resultobject.json", "r", encoding="utf8") as infile:
        res_expected = json.load(infile, cls=oogeso.dto.serialisation.OogesoResultJSONDecoder)
    return res_expected


@pytest.fixture
def testcase2_data() -> oogeso.dto.EnergySystemData:
    """
    Medium test case
    """
    data = oogeso.io.read_data_from_yaml(TEST_DATA_ROOT_PATH / "testcase2_inputdata.yaml")

    profiles_dfs = oogeso.io.read_profiles_from_csv(
        filename_forecasts=TEST_DATA_ROOT_PATH / "testcase2_profiles_forecasts.csv",
        filename_nowcasts=TEST_DATA_ROOT_PATH / "testcase2_profiles_nowcasts.csv",
        timestamp_col="timestamp",
        exclude_cols=["timestep"],
    )
    profiles_json = create_time_series_data(
        profiles_dfs["forecast"], profiles_dfs["nowcast"], time_start=None, time_end=None, timestep_minutes=15
    )
    data.profiles = [x for x in profiles_json if x.id in ["wind", "demand"]]

    return data


@pytest.fixture
def testcase2_expected_result() -> oogeso.dto.EnergySystemData:
    """Expected results for the testcase 2."""
    with open(TEST_DATA_ROOT_PATH / "testcase2_resultobject.json", "r", encoding="utf8") as infile:
        res_expected = json.load(infile, cls=oogeso.dto.serialisation.OogesoResultJSONDecoder)
    return res_expected


@pytest.fixture
def leogo_test_data() -> oogeso.dto.EnergySystemData:
    """
    Big and complex test case - the LEOGO case
    """

    data = oogeso.io.read_data_from_yaml(TEST_DATA_ROOT_PATH / "leogo_reference_platform.yaml")
    profiles_dfs = oogeso.io.read_profiles_from_csv(
        filename_forecasts=TEST_DATA_ROOT_PATH / "leogo_profiles_forecasts.csv",
        filename_nowcasts=TEST_DATA_ROOT_PATH / "leogo_profiles_nowcasts.csv",
        timestamp_col="timestamp",
        exclude_cols=["timestep"],
    )
    profiles_json = create_time_series_data(
        profiles_dfs["forecast"],
        profiles_dfs["nowcast"],
        time_start=None,
        time_end=None,
        timestep_minutes=5,
        resample_method="linear",
    )
    data.profiles = profiles_json

    # Case A with wind power:
    gts = ["Gen1", "Gen2", "Gen3"]
    wells = ["wellL1", "wellL2"]
    hydrogendevices = ["electrolyser", "fuelcell", "h2storage"]
    for dev in data.devices:
        if dev.id in gts:
            dev.start_stop.is_on_init = True
        if dev.id == "wind":
            dev.flow_max = 24
        if dev.id == "battery":
            dev.include = 0
        if dev.id in wells:
            dev.flow_min = dev.flow_max
        if dev.id in hydrogendevices:
            dev.include = 0
    return data


@pytest.fixture
def leogo_expected_result() -> oogeso.dto.EnergySystemData:
    """
    Expected results for the Leogo case
    """
    with open(TEST_DATA_ROOT_PATH / "leogo_resultobject.json", "r", encoding="utf8") as infile:
        res_expected = json.load(infile, cls=oogeso.dto.serialisation.OogesoResultJSONDecoder)
    return res_expected
