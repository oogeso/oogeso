import pytest
import pandas as pd
import oogeso
from oogeso.dto import (
    EnergySystemData,
    OptimisationParametersData,
    NodeData,
    EdgeElData,
    CarrierElData,
    DevicePowersinkData,
    DevicePowersourceData,
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
    nodes = [NodeData("node1"), NodeData("node2")]
    edges = [EdgeElData("edge1_2", node_from="node1", node_to="node2")]
    dev1 = DevicePowersourceData(
        id="source1", node_id="node1", flow_max=20, penalty_function=[[0, 20], [0, 5]]
    )
    dev2 = DevicePowersinkData(
        id="demand", node_id="node2", flow_min=15, flow_max=15, profile="demand"
    )
    devices = [dev1, dev2]
    prof_demand = TimeSeriesData(
        id="demand", data=[1, 1, 1, 1], data_nowcast=[1.1, 1.1, 1.1, 1.1]
    )
    profiles = [prof_demand]
    energy_system_data = EnergySystemData(
        parameters, carriers, nodes, edges, devices, profiles=profiles
    )
    return energy_system_data


def test_simulator_create():
    energy_system_data = _make_test_data()
    sim_obj = oogeso.Simulator(energy_system_data)
    assert isinstance(sim_obj, oogeso.Simulator)
    assert isinstance(sim_obj.optimiser, oogeso.OptimisationModel)
    assert isinstance(sim_obj.profiles["forecast"], pd.DataFrame)
    assert isinstance(sim_obj.profiles["nowcast"], pd.DataFrame)


def test_simulator_runsim():
    energy_system_data = _make_test_data()
    sim_obj = oogeso.Simulator(energy_system_data)
    with pytest.raises(RuntimeError):
        sim_res = sim_obj.runSimulation(solver="wrong_solver_name", timerange=[0, 2])

    # single timestep:
    sim_res = sim_obj.runSimulation(solver="cbc", timelimit=[0, 1])
    assert sim_res.device_flow["source1", "el", "out", 0] == 15 * 1.1
    assert sim_res.device_flow["demand", "el", "in", 0] == 15 * 1.1
    assert sim_res.edge_flow["edge1_2", 0] == 15 * 1.1
