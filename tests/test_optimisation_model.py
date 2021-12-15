import pandas as pd
import pyomo.environ as pyo
import pyomo.opt as pyopt
import pytest

import oogeso
import oogeso.io
from oogeso.dto.oogeso_input_data_objects import EnergySystemData


def make_test_data() -> EnergySystemData:
    parameters = oogeso.dto.OptimisationParametersData(
        objective="penalty",
        time_delta_minutes=5,
        planning_horizon=3,
        optimisation_timesteps=1,
        forecast_timesteps=1,
    )
    carriers = [oogeso.dto.CarrierElData(id="el")]
    nodes = [oogeso.dto.NodeData("node1"), oogeso.dto.NodeData("node2")]
    edges = [oogeso.dto.EdgeElData("edge1_2", node_from="node1", node_to="node2")]
    dev1 = oogeso.dto.DevicePowerSourceData(
        id="source1", node_id="node1", flow_max=20, penalty_function=([0, 20], [0, 5])
    )
    dev2 = oogeso.dto.DevicePowerSinkData(id="demand", node_id="node2", flow_min=15, flow_max=15, profile="demand")
    devices = [dev1, dev2]
    prof_demand = oogeso.dto.TimeSeriesData(id="demand", data=[1, 2, 3, 4], data_nowcast=[1.1, 2.1, 3.1, 4.1])
    profiles = [prof_demand]
    energy_system_data = oogeso.dto.EnergySystemData(parameters, carriers, nodes, edges, devices, profiles=profiles)
    return energy_system_data


def test_optimiser_create():
    """Test that the creation of an oogeso optimisation model object is ok"""
    energy_system_data = make_test_data()
    optimisation_model = oogeso.OptimisationModel(data=energy_system_data)
    assert isinstance(optimisation_model, oogeso.OptimisationModel)
    assert isinstance(optimisation_model, pyo.ConcreteModel)
    assert optimisation_model.setHorizon == [0, 1, 2]
    assert optimisation_model.setDevice == ["source1", "demand"]

    el_consumers = optimisation_model.getDevicesInout(carrier_in="el")
    assert el_consumers == ["demand"]
    el_producers = optimisation_model.getDevicesInout(carrier_out="el")
    assert el_producers == ["source1"]


def test_optimiser_extractvalues():
    """Check that extracting variable values works without error"""
    energy_system_data = make_test_data()
    optimisation_model = oogeso.OptimisationModel(data=energy_system_data)
    var_values = optimisation_model.extract_all_variable_values()
    assert isinstance(var_values["varDeviceFlow"], pd.Series)
    # Value should be the same as initial value (0) since it has not been solved yet
    assert var_values["varDeviceFlow"]["demand", "el", "in", 0] == 0


def test_optimiser_updatemodel():
    """Test that the method for updating an oogeso optimisation models"""

    energy_system_data = make_test_data()
    # convert profile data to dictionary of dataframes needed internally:
    profiles_df = {"forecast": pd.DataFrame(), "nowcast": pd.DataFrame()}
    for pr in energy_system_data.profiles:
        profiles_df["forecast"][pr.id] = pr.data
        profiles_df["nowcast"][pr.id] = pr.data_nowcast

    optimisation_model = oogeso.OptimisationModel(data=energy_system_data)

    optimisation_model.updateOptimisationModel(timestep=0, profiles=profiles_df, first=True)

    optimisation_model.updateOptimisationModel(timestep=1, profiles=profiles_df, first=False)

    # Selecting timestep outside data should give error
    # 3 timesteps in each optimisation, so with profile of length 4, we can only
    # update for timestep 0 and 1 without error
    with pytest.raises(KeyError):
        optimisation_model.updateOptimisationModel(timestep=2, profiles=profiles_df, first=False)


def test_optimiser_compute():
    """ "Check that objective function expressions are valid"""
    energy_system_data = make_test_data()
    optimisation_model = oogeso.OptimisationModel(data=energy_system_data)
    pyomo_model = optimisation_model

    # co2 should be zero in this case
    avg_co2 = optimisation_model.compute_CO2(pyomo_model, devices=["source1"])
    assert avg_co2 == 0

    # co2 intenisty should be None as no export of oil/gas
    int_co2 = optimisation_model.compute_CO2_intensity(pyomo_model)
    assert int_co2 is None

    # No start-stop penalty defined, so should be zero
    penalty_start = optimisation_model.compute_startup_penalty(pyomo_model)
    assert penalty_start == 0

    # No storage, so depletion cost should be zero
    cost_storagedepl = optimisation_model.compute_costForDepletedStorage(pyomo_model)
    assert cost_storagedepl == 0

    # No operating cost defined so should be zero
    op_cost = optimisation_model.compute_operatingCosts(pyomo_model)
    assert op_cost == 0

    # No devices represent export, so should be zero
    export_volume = optimisation_model.compute_export(pyomo_model, value="volume", carriers=None, timesteps=None)
    assert export_volume == 0

    # No oil/gas export  volume should give zero
    export_oilgas_volume = optimisation_model.compute_oilgas_export(pyomo_model, timesteps=None)
    assert export_oilgas_volume == 0

    # No export so therefore no export revenue
    export_revenue = optimisation_model.compute_exportRevenue(pyomo_model)
    assert export_revenue == 0


@pytest.mark.skipif(not pyo.SolverFactory("cbc").available(), reason="Skipping test because CBC is not available.")
def test_optimisation_solve():
    """Check that it solves simple problem with CBC solver"""

    pyo.SolverFactory("cbc")

    energy_system_data = make_test_data()
    model = oogeso.OptimisationModel(data=energy_system_data)

    sol = model.solve(solver="cbc")

    # Solves and finds optimal solution
    assert sol.solver.status == pyopt.SolverStatus.ok
    assert sol.solver.termination_condition == pyopt.TerminationCondition.optimal
    # profiles were not updated before solving, so are all 1
    assert pyo.value(model.varDeviceFlow["demand", "el", "in", 0]) == 15 * 1
    assert pyo.value(model.varDeviceFlow["source1", "el", "out", 0]) == 15 * 1

    # Check that variable extraction works as well, with correct value:
    var_values = model.extract_all_variable_values()
    assert isinstance(var_values["varDeviceFlow"], pd.Series)
    assert var_values["varDeviceFlow"]["demand", "el", "in", 0] == 15 * 1
