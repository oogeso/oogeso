import numpy as np
import pyomo.environ as pyo
import pyomo.opt as pyopt
import pytest

import oogeso
import oogeso.dto.serialisation
import oogeso.io
from oogeso import dto


@pytest.mark.skipif(not pyo.SolverFactory("cbc").available(), reason="Skipping test because CBC is not available.")
def test_simulator_run(testcase1_data):
    simulator = oogeso.Simulator(testcase1_data)

    sol = simulator.optimiser.solve(solver="cbc", time_limit=20)
    assert sol.solver.termination_condition == pyopt.TerminationCondition.optimal, "Optimisation with CBC failed"
    res = simulator.run_simulation("cbc", time_range=(0, 1), time_limit=20)
    assert res.device_flow.loc[("dem", "el", "in", 0)] == res.device_flow.loc[("source1", "el", "out", 0)]


@pytest.mark.skipif(not pyo.SolverFactory("cbc").available(), reason="Skipping test because CBC is not available.")
def test_integration_case1(testcase1_data: dto.EnergySystemData):

    simulator = oogeso.Simulator(testcase1_data)
    assert isinstance(simulator, oogeso.Simulator)
    assert isinstance(simulator.optimiser, oogeso.OptimisationModel)

    res = simulator.run_simulation("cbc", time_range=(0, 4))

    assert (res.device_flow["dem", "el", "in"] == 15).all()
    assert (res.device_flow["source1", "el", "out"] == 15).all()
    assert (res.device_is_prep == 0).all()
    assert (res.device_is_on == 1).all()
    assert res.device_starting.empty
    assert res.device_stopping.empty
    assert (res.edge_flow["el1"] == 15).all()
    assert (res.edge_loss["el1"] == 0).all()
    assert (res.penalty.unstack("device")["source1"] == 3.75).all()
    assert (res.penalty.unstack("device")["dem"] == 0).all()
    assert (res.el_reserve == 5).all()


@pytest.mark.skipif(not pyo.SolverFactory("cbc").available(), reason="Skipping test because CBC is not available.")
def test_integration_case2(testcase2_data: dto.EnergySystemData, testcase2_expected_result: dto.SimulationResult):

    simulator = oogeso.Simulator(data=testcase2_data)
    assert isinstance(simulator, oogeso.Simulator)
    assert isinstance(simulator.optimiser, oogeso.OptimisationModel)

    res_computed = simulator.run_simulation("cbc", time_range=(0, 90))

    # Check that results are as expected.
    assert np.allclose(res_computed.device_flow, testcase2_expected_result.device_flow)
    assert (res_computed.device_is_on == testcase2_expected_result.device_is_on).all()
    assert (res_computed.device_is_prep == testcase2_expected_result.device_is_prep).all()
    assert (res_computed.device_starting == testcase2_expected_result.device_starting).all()
    assert (res_computed.device_stopping == testcase2_expected_result.device_stopping).all()
    assert np.allclose(res_computed.edge_flow, testcase2_expected_result.edge_flow)
    assert np.allclose(res_computed.edge_loss, testcase2_expected_result.edge_loss)
    assert np.allclose(res_computed.el_reserve, testcase2_expected_result.el_reserve)
    assert np.allclose(res_computed.el_backup, testcase2_expected_result.el_backup)
    assert np.allclose(res_computed.penalty, testcase2_expected_result.penalty)
    assert np.allclose(res_computed.terminal_flow, testcase2_expected_result.terminal_flow)
    assert np.allclose(res_computed.terminal_pressure, testcase2_expected_result.terminal_pressure)
    assert np.allclose(res_computed.co2_rate, testcase2_expected_result.co2_rate)
    assert np.allclose((res_computed.co2_rate_per_dev - testcase2_expected_result.co2_rate_per_dev).astype(float), 0)


@pytest.mark.skipif(not pyo.SolverFactory("cbc").available(), reason="Skipping test because CBC is not available.")
def test_integration_case_leogo(leogo_test_data: dto.EnergySystemData, leogo_expected_result: dto.SimulationResult):

    simulator = oogeso.Simulator(leogo_test_data)

    res_computed = simulator.run_simulation("cbc", time_range=(0, 40))

    # Check that results are as expected.
    deviation = res_computed.device_flow-leogo_expected_result.device_flow
    print(deviation[deviation!=0])
    assert res_computed.device_flow.shape == leogo_expected_result.device_flow.shape
    assert np.allclose(res_computed.device_flow, leogo_expected_result.device_flow)
    assert (res_computed.device_is_on == leogo_expected_result.device_is_on).all()
    assert (res_computed.device_is_prep == leogo_expected_result.device_is_prep).all()
    assert (res_computed.device_starting == leogo_expected_result.device_starting).all()
    assert (res_computed.device_stopping == leogo_expected_result.device_stopping).all()
    assert np.allclose(res_computed.edge_flow, leogo_expected_result.edge_flow)
    assert np.allclose(res_computed.edge_loss, leogo_expected_result.edge_loss)
    assert np.allclose(res_computed.el_reserve, leogo_expected_result.el_reserve)
    assert np.allclose(res_computed.el_backup, leogo_expected_result.el_backup, atol=1e-6)
    assert np.allclose(res_computed.penalty, leogo_expected_result.penalty)
    assert np.allclose(res_computed.terminal_flow, leogo_expected_result.terminal_flow)
    assert np.allclose(res_computed.terminal_pressure, leogo_expected_result.terminal_pressure)
    assert np.allclose(res_computed.co2_rate, leogo_expected_result.co2_rate)
    assert np.allclose(
        (res_computed.co2_rate_per_dev - leogo_expected_result.co2_rate_per_dev).astype(float), 0, atol=1e-6
    )


@pytest.mark.skipif(not pyo.SolverFactory("cbc").available(), reason="Skipping test because CBC is not available.")
def test_integration_startstop():
    # Test that minimum online/offline time constraint, and startup penalty work as expected
    parameters = oogeso.dto.OptimisationParametersData(
        objective="penalty", time_delta_minutes=10, planning_horizon=20, optimisation_timesteps=20, forecast_timesteps=0
    )
    carriers = [oogeso.dto.CarrierElData(id="el")]
    edges = []
    nodes = [oogeso.dto.NodeData(id="node1")]
    penalty1 = [[0, 6], [1.1, 1.2]]
    penalty2 = [[0, 6], [2.1, 2.2]]
    sec_per_timestep = 60 * 10  # 10 minutes
    start_stop = {
        1: oogeso.dto.StartStopData(
            is_on_init=True, minimum_time_on_minutes=0, minimum_time_off_minutes=0, delay_start_minutes=0
        ),
        2: oogeso.dto.StartStopData(
            is_on_init=True, minimum_time_on_minutes=20, minimum_time_off_minutes=0, delay_start_minutes=0
        ),
        3: oogeso.dto.StartStopData(
            is_on_init=True, minimum_time_on_minutes=80, minimum_time_off_minutes=0, delay_start_minutes=0
        ),
        4: oogeso.dto.StartStopData(
            is_on_init=True, minimum_time_on_minutes=0, minimum_time_off_minutes=0, delay_start_minutes=0
        ),
        5: oogeso.dto.StartStopData(
            is_on_init=True, minimum_time_on_minutes=0, minimum_time_off_minutes=20, delay_start_minutes=0
        ),
        6: oogeso.dto.StartStopData(
            is_on_init=True, minimum_time_on_minutes=0, minimum_time_off_minutes=80, delay_start_minutes=0
        ),
        7: oogeso.dto.StartStopData(is_on_init=True, delay_start_minutes=0, penalty_start=3 * sec_per_timestep),
        8: oogeso.dto.StartStopData(is_on_init=True, delay_start_minutes=0, penalty_start=2.3 * sec_per_timestep * 4),
    }
    expected_is_on = {
        1: [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        2: [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        3: [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        4: [1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        5: [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        6: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        7: [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],  # start penalty is small
        8: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],  # start penalty is high
    }
    profiles = [
        oogeso.dto.TimeSeriesData(
            id="prof_load1", data=[3, 3, 3, 9, 4, 4, 4, 4, 4, 10, 5, 5, 5, 5, 5, 11, 11, 11, 11, 11, 11]
        ),
        oogeso.dto.TimeSeriesData(
            id="prof_load2", data=[9, 9, 9, 3, 10, 10, 10, 10, 10, 4, 4, 4, 4, 4, 11, 5, 5, 5, 5, 5, 5]
        ),
        oogeso.dto.TimeSeriesData(
            id="prof_load7", data=[3, 3, 3, 9, 4, 4, 4, 4, 10, 10, 5, 5, 5, 5, 5, 5, 11, 11, 11, 11, 11]
        ),
    ]
    prof_id = {
        1: "prof_load1",
        2: "prof_load1",
        3: "prof_load1",
        4: "prof_load2",
        5: "prof_load2",
        6: "prof_load2",
        7: "prof_load7",
        8: "prof_load7",
    }

    # minimum on/of times
    for test_num in range(1, 7):
        devices = [
            oogeso.dto.DevicePowerSinkData(
                id="load", node_id="node1", profile=prof_id[test_num], flow_min=1, flow_max=10
            ),
            oogeso.dto.DevicePowerSourceData(
                id="gen1",
                node_id="node1",
                flow_min=1,
                flow_max=6,
                start_stop=start_stop[test_num],
                penalty_function=penalty1,
            ),
            oogeso.dto.DevicePowerSourceData(
                id="gen2",
                node_id="node1",
                flow_min=1,
                flow_max=6,
                start_stop=start_stop[test_num],
                penalty_function=penalty2,
            ),
        ]
        test_data = oogeso.dto.EnergySystemData(
            parameters=parameters, carriers=carriers, nodes=nodes, edges=edges, devices=devices, profiles=profiles
        )
        simulator = oogeso.Simulator(test_data)
        sim_res = simulator.run_simulation(solver="cbc", time_range=None)
        assert (sim_res.device_is_on["gen2"] == expected_is_on[test_num]).all()

    # startup penalty
    for test_num in range(7, 9):
        devices = [
            oogeso.dto.DevicePowerSinkData(
                id="load", node_id="node1", profile=prof_id[test_num], flow_min=1, flow_max=10
            ),
            oogeso.dto.DevicePowerSourceData(
                id="gen1",
                node_id="node1",
                flow_min=1,
                flow_max=6,
                start_stop=start_stop[test_num],
                penalty_function=penalty1,
            ),
            oogeso.dto.DevicePowerSourceData(
                id="gen2",
                node_id="node1",
                flow_min=1,
                flow_max=6,
                start_stop=start_stop[test_num],
                penalty_function=penalty2,
            ),
        ]
        test_data = oogeso.dto.EnergySystemData(
            parameters=parameters, carriers=carriers, nodes=nodes, edges=edges, devices=devices, profiles=profiles
        )
        simulator = oogeso.Simulator(test_data)
        sim_res = simulator.run_simulation(solver="cbc", time_range=None)
        assert (sim_res.device_is_on["gen2"] == expected_is_on[test_num]).all()
