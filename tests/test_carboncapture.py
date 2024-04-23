import pyomo.environ as pyo
import pytest

import oogeso
import oogeso.dto as dto
from oogeso.core.devices import CarbonCapture

from .conftest import TEST_DATA_ROOT_PATH


@pytest.fixture
def testcase_carboncapture_data() -> oogeso.dto.base.EnergySystemData:
    """
    Simple test case, electric only
    """
    return oogeso.io.read_data_from_yaml(TEST_DATA_ROOT_PATH / "testdata_carboncapture.yaml")


def test_carboncapture_model():
    # dev_obj = CarbonCapture()

    egr = 0.4
    ccr = 0.9
    flow_in_co2 = 1
    frac_co2 = CarbonCapture.co2frac(egr=egr)
    specific_heat_demand = CarbonCapture.specific_heat_demand(frac_co2=frac_co2)
    specific_el_demand = CarbonCapture.specific_electricity_demand(frac_co2=frac_co2)
    heat_demand = CarbonCapture.required_heat(flow_in_co2=1, egr=egr, ccr=ccr)
    el_demand = CarbonCapture.required_electricity(flow_in_co2=flow_in_co2, egr=egr, ccr=ccr)
    co2_captured = flow_in_co2 * (1 - egr) * ccr

    assert frac_co2 == (12.88 * egr + 0.7874) / 100  # = 0.059394
    if frac_co2 < 0.0489:
        assert specific_heat_demand == -3.57 * frac_co2 + 4.1551
        assert specific_el_demand == 0.16 * frac_co2 + 0.0232
    else:
        assert specific_heat_demand == -2.11 * frac_co2 + 4.0837
        assert specific_el_demand == 0.009 * frac_co2 + 0.0305
    assert heat_demand == co2_captured * specific_heat_demand
    assert el_demand == co2_captured * specific_el_demand


@pytest.mark.skipif(not pyo.SolverFactory("cbc").available(), reason="Skipping test because CBC is not available.")
def test_carboncapture_simple_simulation(testcase_carboncapture_data: dto.EnergySystemData):
    my_sim = oogeso.Simulator(testcase_carboncapture_data)
    res = my_sim.run_simulation("cbc", time_range=(0, 4))
    assert (res.device_flow["ccs", "carbon", "in"] == 2.34).all()
    assert (res.device_flow["ccs", "carbon", "out"] == 0.1404).all()
