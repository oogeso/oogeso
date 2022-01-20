import sys

import pandas as pd
import pytest

import oogeso
from oogeso import dto

try:
    from oogeso import plots as op
except ImportError:
    pass


@pytest.mark.skipif(
    not all([module in sys.modules for module in ["matplotlib", "plotly", "seaborn"]]),
    reason="Plotting modules has not been installed",
)
def test_plot_co2_intensity(leogo_expected_result: dto.SimulationResult):
    sim_result = leogo_expected_result
    op.plot_CO2_intensity(sim_result)


@pytest.mark.skipif(
    not all([module in sys.modules for module in ["matplotlib", "plotly", "seaborn"]]),
    reason="Plotting modules has not been installed",
)
def test_plot_co2_rate(leogo_expected_result: dto.SimulationResult):
    sim_result = leogo_expected_result
    op.plot_CO2_rate(sim_result)


@pytest.mark.skipif(
    not all([module in sys.modules for module in ["matplotlib", "plotly", "seaborn"]]),
    reason="Plotting modules has not been installed",
)
def test_plot_co2_rate_per_dev(leogo_test_data: dto.EnergySystemData, leogo_expected_result: dto.SimulationResult):
    sim_result = leogo_expected_result
    optimisation_model = oogeso.OptimisationModel(leogo_test_data)
    op.plot_CO2_rate_per_dev(sim_result, optimisation_model)


@pytest.mark.skipif(
    not all([module in sys.modules for module in ["matplotlib", "plotly", "seaborn"]]),
    reason="Plotting modules has not been installed",
)
def test_plot_device_power_energy(
    testcase2_data: dto.EnergySystemData, testcase2_expected_result: dto.SimulationResult
):
    sim_result = testcase2_expected_result
    optimisation_model = oogeso.OptimisationModel(testcase2_data)
    op.plot_device_power_energy(sim_result, optimisation_model, dev="battery")


@pytest.mark.skipif(
    not all([module in sys.modules for module in ["matplotlib", "plotly", "seaborn"]]),
    reason="Plotting modules has not been installed",
)
def test_plot_device_power_energy_nostorage(
    testcase1_data: dto.EnergySystemData, testcase1_expected_result: dto.SimulationResult
):
    sim_result = testcase1_expected_result
    optimisation_model = oogeso.OptimisationModel(testcase1_data)
    op.plot_device_power_energy(sim_result, optimisation_model, dev="dem")


@pytest.mark.skipif(
    not all([module in sys.modules for module in ["matplotlib", "plotly", "seaborn"]]),
    reason="Plotting modules has not been installed",
)
def test_plot_device_power_flow_pressure(
    leogo_test_data: dto.EnergySystemData, leogo_expected_result: dto.SimulationResult
):
    sim_result = leogo_expected_result
    optimisation_model = oogeso.OptimisationModel(leogo_test_data)
    op.plot_device_power_flow_pressure(sim_result, optimisation_model, dev="REC")


@pytest.mark.skipif(
    not all([module in sys.modules for module in ["matplotlib", "plotly", "seaborn"]]),
    reason="Plotting modules has not been installed",
)
def test_plot_device_profile(leogo_test_data: dto.EnergySystemData, leogo_expected_result: dto.SimulationResult):
    sim_result = leogo_expected_result
    optimisation_model = oogeso.OptimisationModel(leogo_test_data)
    op.plot_device_profile(sim_result, optimisation_model, devs=["wind"], include_forecasts=True)
    op.plot_device_profile(
        sim_result, optimisation_model, devs=["Gen1", "Gen2", "Gen3"], include_on_off=True, include_prep=True
    )


@pytest.mark.skipif(
    not all([module in sys.modules for module in ["matplotlib", "plotly", "seaborn"]]),
    reason="Plotting modules has not been installed",
)
def test_plot_df():
    df = pd.DataFrame()
    df["time"] = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    df["value"] = [1, 2, 3, 4, 5, 3, 2, 1, 1, 0]
    df["class"] = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    op.plot_df(df, id_var="class")


@pytest.mark.skipif(
    not all([module in sys.modules for module in ["matplotlib", "plotly", "seaborn"]]),
    reason="Plotting modules has not been installed",
)
def test_plot_el_backup(leogo_expected_result: dto.SimulationResult):
    sim_result = leogo_expected_result
    op.plot_el_backup(sim_result, showMargin=True)


@pytest.mark.skipif(
    not all([module in sys.modules for module in ["matplotlib", "plotly", "seaborn"]]),
    reason="Plotting modules has not been installed",
)
def test_plot_export_revenue(leogo_expected_result: dto.SimulationResult):
    sim_result = leogo_expected_result
    op.plot_export_revenue(sim_result)


@pytest.mark.skipif(
    not all([module in sys.modules for module in ["matplotlib", "plotly", "seaborn"]]),
    reason="Plotting modules has not been installed",
)
def test_plot_gas_turbine_efficiency():
    op.plot_gas_turbine_efficiency()


@pytest.mark.skipif(
    not all([module in sys.modules for module in ["matplotlib", "plotly", "seaborn"]]),
    reason="Plotting modules has not been installed",
)
def test_plot_network(leogo_test_data: dto.EnergySystemData, leogo_expected_result: dto.SimulationResult):
    simulator = oogeso.Simulator(data=leogo_test_data)
    simulator.result_object = leogo_expected_result
    op.plot_network(simulator, timestep=1)
    # If no errors, it's OK


@pytest.mark.skipif(
    not all([module in sys.modules for module in ["matplotlib", "plotly", "seaborn"]]),
    reason="Plotting modules has not been installed",
)
def test_plot_profiles(leogo_test_data: dto.EnergySystemData):
    op.plot_profiles(leogo_test_data.profiles)


@pytest.mark.skipif(
    not all([module in sys.modules for module in ["matplotlib", "plotly", "seaborn"]]),
    reason="Plotting modules has not been installed",
)
def test_plot_reservek(leogo_test_data: dto.EnergySystemData, leogo_expected_result: dto.SimulationResult):
    sim_result = leogo_expected_result
    optimisation_model = oogeso.OptimisationModel(leogo_test_data)
    op.plot_reserve(sim_result, optimisation_model)


@pytest.mark.skipif(
    not all([module in sys.modules for module in ["matplotlib", "plotly", "seaborn"]]),
    reason="Plotting modules has not been installed",
)
def test_plot_sum_power_mix(leogo_test_data: dto.EnergySystemData, leogo_expected_result: dto.SimulationResult):
    sim_result = leogo_expected_result
    optimisation_model = oogeso.OptimisationModel(leogo_test_data)
    op.plot_sum_power_mix(sim_result, optimisation_model, carrier="el")
