import pyomo.environ as pyo
import pytest

import oogeso
import oogeso.dto as dto

from .conftest import TEST_DATA_ROOT_PATH


@pytest.fixture
def testcase_gasturbine_hydrogen_data() -> oogeso.dto.base.EnergySystemData:
    """
    Simple test case, electric only
    """
    return oogeso.io.read_data_from_yaml(TEST_DATA_ROOT_PATH / "testdata_gasturbine_hydrogen.yaml")


def test_gasturbine_hydrogen_model(testcase_gasturbine_hydrogen_data: dto.EnergySystemData):
    my_sim = oogeso.Simulator(testcase_gasturbine_hydrogen_data)
    dev_gtg = my_sim.optimiser.all_devices["gtg"]
    assert dev_gtg.dev_data.hydrogen_blend_max == 0.2


@pytest.mark.skipif(not pyo.SolverFactory("cbc").available(), reason="Skipping test because CBC is not available.")
def test_gasturbine_hydrogen_simple_simulation(testcase_gasturbine_hydrogen_data: dto.EnergySystemData):
    my_sim = oogeso.Simulator(testcase_gasturbine_hydrogen_data)
    res = my_sim.run_simulation("cbc", time_range=(0, 4))

    hydrogen_fraction = res.device_flow["gtg", "hydrogen", "in", 0] / (
        res.device_flow["gtg", "hydrogen", "in", 0] + res.device_flow["gtg", "gas", "in", 0]
    )
    assert hydrogen_fraction == 0.2  # max value

    assert res.device_flow["gtg", "gas", "in", 0] == pytest.approx(0.5433526, rel=1e-6)
    assert res.device_flow["gtg", "hydrogen", "in", 0] == pytest.approx(0.13583815, rel=1e-6)
    assert res.device_flow["gtg", "carbon", "out", 0] == pytest.approx(1.2714451, rel=1e-6)
