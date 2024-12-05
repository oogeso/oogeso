import pyomo.environ as pyo
import pytest

import oogeso
import oogeso.dto as dto

from .conftest import TEST_DATA_ROOT_PATH


@pytest.fixture
def testcase_hydrogenstorage_data() -> oogeso.dto.base.EnergySystemData:
    """
    Simple test case with hydrogen storage
    """
    return oogeso.io.read_data_from_yaml(TEST_DATA_ROOT_PATH / "testdata_hydrogen_storage.yaml")


def test_hydrogenstorage_model(testcase_hydrogenstorage_data: dto.EnergySystemData):
    my_sim = oogeso.Simulator(testcase_hydrogenstorage_data)
    dev = my_sim.optimiser.all_devices["h2tank"]

    assert dev.dev_data.compressor_pressure_max == 35

    tank_state = my_sim.optimiser.all_devices["h2tank"].get_tank_state(my_sim.optimiser)

    assert tank_state["volume_m3"] == pytest.approx(706.2857142857143, rel=1e-6)


@pytest.mark.skipif(not pyo.SolverFactory("cbc").available(), reason="Skipping test because CBC is not available.")
def test_hydrogen_storage_simulation(testcase_hydrogenstorage_data: dto.EnergySystemData):
    my_sim = oogeso.Simulator(testcase_hydrogenstorage_data)
    res = my_sim.run_simulation("cbc", time_range=(0, 8))

    # Compressor electricity demand:
    assert res.device_flow["h2tank", "el", "in", 0] == pytest.approx(0.685492, rel=1e-6)

    # Storage level:
    assert res.device_storage_energy["h2tank", 7] == pytest.approx(29511.265, rel=1e-6)
