from pathlib import Path
import oogeso
import oogeso.io

TEST_DATA_ROOT_PATH = Path(__file__).parent


def test_simulator_create():
    data = oogeso.io.read_data_from_yaml(TEST_DATA_ROOT_PATH / "testdata1.yaml")
    simulator = oogeso.Simulator(data)
    # If not failed above, it's OK
    assert isinstance(simulator, oogeso.Simulator)
    assert isinstance(simulator.optimiser, oogeso.OptimisationModel)
    # assert isinstance(simulator.optimiser.pyomo_instance)
