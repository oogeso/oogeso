import pyomo.environ as pyo
import pytest

import oogeso
import oogeso.dto as dto

from .conftest import TEST_DATA_ROOT_PATH


@pytest.fixture
def testcase_gaslinepack_data() -> oogeso.dto.base.EnergySystemData:
    """
    Simple test case, electric only
    """
    return oogeso.io.read_data_from_yaml(TEST_DATA_ROOT_PATH / "testdata_gas_line_pack.yaml")


def test_gaslinepack_model(testcase_gaslinepack_data: dto.EnergySystemData):
    my_sim = oogeso.Simulator(testcase_gaslinepack_data)
    dev = my_sim.optimiser.all_devices["gas_linepack"]
    assert dev.dev_data.volume_m3 == 12825


@pytest.mark.skipif(not pyo.SolverFactory("cbc").available(), reason="Skipping test because CBC is not available.")
def test_gas_linepack_simulation(testcase_gaslinepack_data: dto.EnergySystemData):
    my_sim = oogeso.Simulator(testcase_gaslinepack_data)
    res = my_sim.run_simulation("cbc", time_range=(0, 4))

    assert res.device_storage_energy["gas_linepack", 4] == pytest.approx(225, rel=1e-6)
