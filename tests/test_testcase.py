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
    assert res.device_starting is None
    assert res.device_stopping is None
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
