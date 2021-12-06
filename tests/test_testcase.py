import json
from pathlib import Path

import pyomo.environ as pyo
import pytest

import oogeso
import oogeso.dto.serialisation
import oogeso.io
from oogeso.dto.oogeso_output_data_objects import SimulationResult

TEST_DATA_ROOT_PATH = Path(__file__).parent
EXAMPLE_DATA_ROOT_PATH = Path(__file__).parent.parent / "examples"


def test_integration_simplecase():
    pytest.skip("Test case not yet implemented")


def test_integration_case1():
    # test using data fraom file
    data = oogeso.io.read_data_from_yaml(TEST_DATA_ROOT_PATH / "testdata1.yaml")
    simulator = oogeso.Simulator(data)
    assert isinstance(simulator, oogeso.Simulator)
    assert isinstance(simulator.optimiser, oogeso.OptimisationModel)

    # Continue test only if cbc executable is present
    opt = pyo.SolverFactory("cbc")
    if not opt.available():
        pytest.skip("CBC executable not found. Skipping test.")

    res = simulator.runSimulation("cbc")

    assert (res.device_flow["dem", "el", "in"] == 15).all()
    assert (res.device_flow["source1", "el", "out"] == 15).all()
    assert (res.device_is_prep == 0).all()
    assert (res.device_is_on == 1).all()
    assert (res.device_starting == 0).all()
    assert (res.device_stopping == 0).all()
    assert (res.edge_flow["el1"] == 15).all()
    assert (res.edge_loss["el1"] == 0).all()
    assert (res.penalty.unstack("device")["source1"] == 3.75).all()
    assert (res.penalty.unstack("device")["dem"] == 0).all()
    assert (res.el_reserve == 5).all()


def test_integration_case2():
    data = oogeso.io.read_data_from_yaml(EXAMPLE_DATA_ROOT_PATH / "testcase2_inputdata.yaml")

    profiles_dfs = oogeso.io.read_profiles_from_csv(
        filename_forecasts="testcase2_profiles_forecasts.csv",
        filename_nowcasts="testcase2_profiles_nowcasts.csv",
        timestamp_col="timestamp",
        exclude_cols=["timestep"],
    )
    profiles_json = oogeso.utils.create_timeseriesdata(
        profiles_dfs["forecast"], profiles_dfs["nowcast"], time_start=None, time_end=None, timestep_minutes=15
    )
    data.profiles = [x for x in profiles_json if x.id in ["wind", "demand"]]

    simulator = oogeso.Simulator(data)
    assert isinstance(simulator, oogeso.Simulator)
    assert isinstance(simulator.optimiser, oogeso.OptimisationModel)

    # Continue test only if cbc executable is present
    opt = pyo.SolverFactory("cbc")
    if not opt.available():
        pytest.skip("CBC executable not found. Skipping test.")

    res_computed = simulator.runSimulation("cbc")
    # Read expected results from file:
    with open("testcase2_resultobject.json", "r") as infile:
        res_expected: SimulationResult = json.load(infile, cls=oogeso.dto.serialisation.OogesoResultJSONDecoder)

    # Check that results are as expected.
    assert (res_computed.device_flow == res_expected.device_flow).all()
    assert (res_computed.device_is_on == res_expected.device_is_on).all()
    assert (res_computed.device_is_prep == res_expected.device_is_prep).all()
    assert (res_computed.device_starting == res_expected.device_starting).all()
    assert (res_computed.device_stopping == res_expected.device_stopping).all()
    assert (res_computed.edge_flow == res_expected.edge_flow).all()
    assert (res_computed.edge_loss == res_expected.edge_loss).all()
    assert (res_computed.el_reserve == res_expected.el_reserve).all()
    assert (res_computed.el_backup == res_expected.el_backup).all()
    assert (res_computed.penalty == res_expected.penalty).all()
    assert (res_computed.terminal_flow == res_expected.terminal_flow).all()
    assert (res_computed.terminal_pressure == res_expected.terminal_pressure).all()
    assert (res_computed.co2_intensity == res_expected.co2_intensity).all()
    assert (res_computed.co2_rate == res_expected.co2_rate).all()
    assert (res_computed.co2_rate_per_dev == res_expected.co2_rate_per_dev).all()


def test_integration_case_leogo():
    pytest.skip("Leogo test case not yet implemented")
