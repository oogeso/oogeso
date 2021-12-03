from pathlib import Path
import oogeso
import oogeso.io
import pyomo.environ as pyo
import pyomo.opt as pyopt
import pytest

TEST_DATA_ROOT_PATH = Path(__file__).parent


def test_simulator_create():
    data = oogeso.io.read_data_from_yaml(TEST_DATA_ROOT_PATH / "testdata1.yaml")
    simulator = oogeso.Simulator(data)
    # If not failed above, it's OK
    assert isinstance(simulator, oogeso.Simulator)
    assert isinstance(simulator.optimiser, oogeso.OptimisationModel)
    # assert isinstance(simulator.optimiser.pyomo_instance)


def test_integration_simplecase():
    pytest.skip("Test case not yet implemented")


def test_integration_case1():
    # test using data fraom file
    data = oogeso.io.read_data_from_yaml(TEST_DATA_ROOT_PATH / "testdata1.yaml")
    simulator = oogeso.Simulator(data)
    # If not failed above, it's OK
    assert isinstance(simulator, oogeso.Simulator)
    assert isinstance(simulator.optimiser, oogeso.OptimisationModel)

    # Continue test only if cbc executable is present
    opt = pyo.SolverFactory("cbc")
    if not opt.available():
        pytest.skip("CBC executable not found. Skipping test.")

    res = simulator.runSimulation("cbc")

    # TODO: Check results are as expected.


def test_integration_case_leogo():
    pytest.skip("Leogo test case not yet implemented")
