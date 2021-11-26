import pytest
from pyomo import environ as pyo, opt as pyopt

import oogeso
from tests.test_testcase import TEST_DATA_ROOT_PATH


@pytest.mark.skipif(not pyo.SolverFactory("cbc").available(), reason="Skipping test because CBC is not available.")
def test_simulator_run():
    data = oogeso.io.read_data_from_yaml(TEST_DATA_ROOT_PATH / "testdata1.yaml")
    simulator = oogeso.Simulator(data)
    # Continue test only if cbc executable is present
    opt = pyo.SolverFactory("cbc")

    sol = simulator.optimiser.solve(solver="cbc", timelimit=20)
    assert (
        sol.solver.termination_condition == pyopt.TerminationCondition.optimal
    ), "Optimisation with CBC failed"
    res = simulator.runSimulation("cbc", timerange=(0, 1), timelimit=20)
    assert (
        res.device_flow.loc[("dem", "el", "in", 0)]
        == res.device_flow.loc[("source1", "el", "out", 0)]
    )
