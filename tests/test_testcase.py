import oogeso
import oogeso.io
import pyomo.opt as pyopt
import pyomo.environ as pyo
import pytest


def test_simulator_create():
    data = oogeso.io.read_data_from_yaml("tests/testdata1.yaml")
    simulator = oogeso.Simulator(data)
    # If not failed above, it's OK
    assert isinstance(simulator, oogeso.Simulator)
    assert isinstance(simulator.optimiser, oogeso.OptimisationModel)
    # assert isinstance(simulator.optimiser.pyomo_instance)


def test_simulator_run():
    data = oogeso.io.read_data_from_yaml("tests/testdata1.yaml")
    simulator = oogeso.Simulator(data)
    # Continue test only if cbc executable is present
    opt = pyo.SolverFactory("cbc")
    if not opt.available():
        pytest.skip("CBC executable not found. Skipping test")

    sol = simulator.optimiser.solve(solver="cbc", timelimit=20)
    assert (
        sol.solver.termination_condition == pyopt.TerminationCondition.optimal
    ), "Optimisation with CBC failed"
    res = simulator.runSimulation("cbc", timerange=[0, 1], timelimit=20)
    # print(res.dfDeviceFlow)
    assert (
        res.dfDeviceFlow.loc[("dem", "el", "in", 0)]
        == res.dfDeviceFlow.loc[("source1", "el", "out", 0)]
    )
