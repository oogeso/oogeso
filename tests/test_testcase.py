import json
from pathlib import Path

import pyomo.environ as pyo
import pytest
import numpy as np

import oogeso
import oogeso.dto.serialisation
import oogeso.io
from oogeso.utils.util import create_timeseriesdata
from oogeso.dto.oogeso_output_data_objects import SimulationResult

TEST_DATA_ROOT_PATH = Path(__file__).parent
# EXAMPLE_DATA_ROOT_PATH = Path(__file__).parent.parent / "examples"


def test_integration_case1():
    # Simple test case, electric only
    data = oogeso.io.read_data_from_yaml(TEST_DATA_ROOT_PATH / "testdata1.yaml")
    simulator = oogeso.Simulator(data)
    assert isinstance(simulator, oogeso.Simulator)
    assert isinstance(simulator.optimiser, oogeso.OptimisationModel)

    # Continue test only if cbc executable is present
    opt = pyo.SolverFactory("cbc")
    if not opt.available():
        pytest.skip("CBC executable not found. Skipping test case 1.")

    res = simulator.runSimulation("cbc", timerange=[0, 4])

    assert (res.device_flow["dem", "el", "in"] == 15).all()
    assert (res.device_flow["source1", "el", "out"] == 15).all()
    assert (res.device_is_prep == 0).all()
    assert (res.device_is_on == 1).all()
    assert res.device_starting is None
    assert res.device_stopping is None
    assert (res.edge_flow["el1"] == 15).all()
    assert (res.edge_loss["el1"] == 0).all()
    assert (res.penalty.unstack("device")["source1"] == 3.75).all()
    assert (res.penalty.unstack("device")["dem"] == 0).all()
    assert (res.el_reserve == 5).all()


def test_integration_case2():
    # Medium test case
    data = oogeso.io.read_data_from_yaml(TEST_DATA_ROOT_PATH / "testcase2_inputdata.yaml")

    profiles_dfs = oogeso.io.read_profiles_from_csv(
        filename_forecasts=TEST_DATA_ROOT_PATH / "testcase2_profiles_forecasts.csv",
        filename_nowcasts=TEST_DATA_ROOT_PATH / "testcase2_profiles_nowcasts.csv",
        timestamp_col="timestamp",
        exclude_cols=["timestep"],
    )
    profiles_json = create_timeseriesdata(
        profiles_dfs["forecast"], profiles_dfs["nowcast"], time_start=None, time_end=None, timestep_minutes=15
    )
    data.profiles = [x for x in profiles_json if x.id in ["wind", "demand"]]

    simulator = oogeso.Simulator(data)
    assert isinstance(simulator, oogeso.Simulator)
    assert isinstance(simulator.optimiser, oogeso.OptimisationModel)

    # Continue test only if cbc executable is present
    opt = pyo.SolverFactory("cbc")
    if not opt.available():
        pytest.skip("CBC executable not found. Skipping test case 2.")

    res_computed = simulator.runSimulation("cbc", timerange=[0, 90])
    # Read expected results from file:
    with open(TEST_DATA_ROOT_PATH / "testcase2_resultobject.json", "r", encoding="utf8") as infile:
        res_expected: SimulationResult = json.load(infile, cls=oogeso.dto.serialisation.OogesoResultJSONDecoder)

    # Check that results are as expected.
    assert np.allclose(res_computed.device_flow, res_expected.device_flow)
    assert (res_computed.device_is_on == res_expected.device_is_on).all()
    assert (res_computed.device_is_prep == res_expected.device_is_prep).all()
    assert (res_computed.device_starting == res_expected.device_starting).all()
    assert (res_computed.device_stopping == res_expected.device_stopping).all()
    assert np.allclose(res_computed.edge_flow, res_expected.edge_flow)
    assert np.allclose(res_computed.edge_loss, res_expected.edge_loss)
    assert np.allclose(res_computed.el_reserve, res_expected.el_reserve)
    assert np.allclose(res_computed.el_backup, res_expected.el_backup)
    assert np.allclose(res_computed.penalty, res_expected.penalty)
    assert np.allclose(res_computed.terminal_flow, res_expected.terminal_flow)
    assert np.allclose(res_computed.terminal_pressure, res_expected.terminal_pressure)
    assert np.allclose(res_computed.co2_rate, res_expected.co2_rate)
    assert np.allclose((res_computed.co2_rate_per_dev - res_expected.co2_rate_per_dev).astype(float), 0)


def test_integration_case_leogo():
    # Big and complex test case - the LEOGO case

    data = oogeso.io.read_data_from_yaml(TEST_DATA_ROOT_PATH / "leogo_reference_platform.yaml")
    profiles_dfs = oogeso.io.read_profiles_from_csv(
        filename_forecasts=TEST_DATA_ROOT_PATH / "leogo_profiles_forecasts.csv",
        filename_nowcasts=TEST_DATA_ROOT_PATH / "leogo_profiles_nowcasts.csv",
        timestamp_col="timestamp",
        exclude_cols=["timestep"],
    )
    profiles_json = create_timeseriesdata(
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

    simulator = oogeso.Simulator(data)

    opt = pyo.SolverFactory("cbc")
    if not opt.available():
        pytest.skip("CBC executable not found. Skipping test case 2.")

    res_computed = simulator.runSimulation("cbc", timerange=[0, 40])
    # Read expected results from file:
    with open(TEST_DATA_ROOT_PATH / "leogo_resultobject.json", "r", encoding="utf8") as infile:
        res_expected: SimulationResult = json.load(infile, cls=oogeso.dto.serialisation.OogesoResultJSONDecoder)

    # Check that results are as expected.
    assert np.allclose(res_computed.device_flow, res_expected.device_flow)
    assert (res_computed.device_is_on == res_expected.device_is_on).all()
    assert (res_computed.device_is_prep == res_expected.device_is_prep).all()
    assert (res_computed.device_starting == res_expected.device_starting).all()
    assert (res_computed.device_stopping == res_expected.device_stopping).all()
    assert np.allclose(res_computed.edge_flow, res_expected.edge_flow)
    assert np.allclose(res_computed.edge_loss, res_expected.edge_loss)
    assert np.allclose(res_computed.el_reserve, res_expected.el_reserve)
    assert np.allclose(res_computed.el_backup, res_expected.el_backup, atol=1e-6)
    assert np.allclose(res_computed.penalty, res_expected.penalty)
    assert np.allclose(res_computed.terminal_flow, res_expected.terminal_flow)
    assert np.allclose(res_computed.terminal_pressure, res_expected.terminal_pressure)
    assert np.allclose(res_computed.co2_rate, res_expected.co2_rate)
    assert np.allclose((res_computed.co2_rate_per_dev - res_expected.co2_rate_per_dev).astype(float), 0, atol=1e-6)

