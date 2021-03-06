import pandas as pd
import pyomo.environ as pyo
import pytest

import oogeso
from oogeso.dto import (
    CarrierElData,
    DevicePowerSinkData,
    DevicePowerSourceData,
    EdgeElData,
    EnergySystemData,
    NodeData,
    OptimisationParametersData,
    TimeSeriesData,
)


def _make_test_data() -> EnergySystemData:
    parameters = OptimisationParametersData(
        objective="penalty",
        time_delta_minutes=5,
        planning_horizon=3,
        optimisation_timesteps=1,
        forecast_timesteps=1,
    )
    carriers = [CarrierElData(id="el")]
    nodes = [NodeData(id="node1"), NodeData(id="node2")]
    edges = [EdgeElData(id="edge1_2", node_from="node1", node_to="node2")]
    dev1 = DevicePowerSourceData(id="source1", node_id="node1", flow_max=20, penalty_function=([0, 20], [0, 5]))
    dev2 = DevicePowerSinkData(id="demand", node_id="node2", flow_min=15, flow_max=15, profile="demand")
    devices = [dev1, dev2]
    prof_demand = TimeSeriesData(id="demand", data=[1, 1, 1, 1], data_nowcast=[1.1, 1.1, 1.1, 1.1])
    profiles = [prof_demand]
    energy_system_data = EnergySystemData(
        parameters=parameters, carriers=carriers, nodes=nodes, edges=edges, devices=devices, profiles=profiles
    )
    return energy_system_data


def test_simulator_create():
    energy_system_data = _make_test_data()
    sim_obj = oogeso.Simulator(energy_system_data)
    assert isinstance(sim_obj, oogeso.Simulator)
    assert isinstance(sim_obj.optimiser, oogeso.OptimisationModel)
    assert isinstance(sim_obj.profiles["forecast"], pd.DataFrame)
    assert isinstance(sim_obj.profiles["nowcast"], pd.DataFrame)


@pytest.mark.skipif(not pyo.SolverFactory("cbc").available(), reason="Skipping test because CBC is not available.")
def test_simulator_runsim():
    energy_system_data = _make_test_data()
    sim_obj = oogeso.Simulator(energy_system_data)
    with pytest.raises(Exception):
        _ = sim_obj.run_simulation(solver="wrong_solver_name", time_range=(0, 2))

    sim_res = sim_obj.run_simulation(solver="cbc", time_limit=1)
    assert sim_res.device_flow["source1", "el", "out", 0] == 15 * 1.1
    assert sim_res.device_flow["demand", "el", "in", 0] == 15 * 1.1
    assert sim_res.edge_flow["edge1_2", 0] == 15 * 1.1
