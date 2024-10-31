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

    # egr = 0.4
    ccr = 0.9
    flow_in_co2 = 1
    # frac_co2 = CarbonCapture.co2frac(egr=egr)
    # specific_heat_demand = CarbonCapture.specific_heat_demand(frac_co2=frac_co2)
    # specific_el_demand = CarbonCapture.specific_electricity_demand(frac_co2=frac_co2)
    heat_demand = CarbonCapture.required_heat(flow_in_co2=1, ccr=ccr, specific_demand_MW_per_kgCO2=0.4)
    el_demand = CarbonCapture.required_electricity(
        flow_in_co2=flow_in_co2, ccr=ccr, specific_demand_MW_per_kgCO2=0.032
    ) + CarbonCapture.required_electricity_compressor(flow_in_co2=flow_in_co2, energy_demand_MJ_per_kg=0.3)
    capture_heat_demand_MJ_per_kgCO2 = 0.4  # MJ/kgCO2
    capture_el_demand_MJ_per_kgCO2 = 0.032  # MJ/kgCO2
    compressor_el_demand_MJ_per_kgCO2 = 0.30  # MJ/kgCO2
    # co2_captured = flow_in_co2 * (1 - egr) * ccr

    assert el_demand == (capture_el_demand_MJ_per_kgCO2 * ccr + compressor_el_demand_MJ_per_kgCO2) * flow_in_co2
    assert heat_demand == capture_heat_demand_MJ_per_kgCO2 * flow_in_co2 * ccr


@pytest.mark.skipif(not pyo.SolverFactory("cbc").available(), reason="Skipping test because CBC is not available.")
def test_carboncapture_simple_simulation(testcase_carboncapture_data: dto.EnergySystemData):
    my_sim = oogeso.Simulator(testcase_carboncapture_data)
    res = my_sim.run_simulation("cbc", time_range=(0, 4))
    assert (res.device_flow["ccs", "carbon", "in"] == 3.66795).all()
    assert (res.device_flow["ccs", "carbon", "out"] == 0.366795).all()
