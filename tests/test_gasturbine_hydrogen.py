import pyomo.environ as pyo
import pytest

import oogeso
import oogeso.dto as dto
from oogeso.core.devices import GasTurbine

from .conftest import TEST_DATA_ROOT_PATH


@pytest.fixture
def testcase_gasturbine_hydrogen_data() -> oogeso.dto.base.EnergySystemData:
    """
    Simple test case, electric only
    """
    return oogeso.io.read_data_from_yaml(TEST_DATA_ROOT_PATH / "testdata_gasturbine_hydrogen.yaml")


def test_gasturbine_hydrogen_model():
    dev_obj = GasTurbine()
    # TODO add tests
    assert 1==0



@pytest.mark.skipif(not pyo.SolverFactory("cbc").available(), reason="Skipping test because CBC is not available.")
def test_gasturbine_hydrogen_simple_simulation(testcase_gasturbine_hydrogen_data: dto.EnergySystemData):
    my_sim = oogeso.Simulator(testcase_gasturbine_hydrogen_data)
    res = my_sim.run_simulation("cbc", time_range=(0, 4))
    # TODO add tests
    assert 1==0

