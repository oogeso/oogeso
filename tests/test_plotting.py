import oogeso
from oogeso import plots as op
from oogeso import dto
from oogeso.core.optimiser import OptimisationModel


def test_plot_co2_intensity(leogo_expected_result: dto.SimulationResult):
    sim_result = leogo_expected_result
    op.plot_CO2_intensity(sim_result)


def test_plot_co2_rate(leogo_expected_result: dto.SimulationResult):
    sim_result = leogo_expected_result
    op.plot_CO2_rate(sim_result)


def test_plot_co2_rate_per_dev(leogo_test_data: dto.EnergySystemData, leogo_expected_result: dto.SimulationResult):
    sim_result = leogo_expected_result
    optimisation_model = oogeso.OptimisationModel(leogo_test_data)
    op.plot_CO2_rate_per_dev(sim_result, optimisation_model)


def test_plot_device_power_energy(leogo_test_data: dto.EnergySystemData, leogo_expected_result: dto.SimulationResult):
    sim_result = leogo_expected_result
    optimisation_model = oogeso.OptimisationModel(leogo_test_data)
    op.plot_device_power_energy(sim_result, optimisation_model, dev="Gen1")


def test_plot_device_power_flow_pressure(
    leogo_test_data: dto.EnergySystemData, leogo_expected_result: dto.SimulationResult
):
    sim_result = leogo_expected_result
    optimisation_model = oogeso.OptimisationModel(leogo_test_data)
    op.plot_device_power_flow_pressure(sim_result, optimisation_model, dev="REC")


def test_plot_device_profile(leogo_test_data: dto.EnergySystemData, leogo_expected_result: dto.SimulationResult):
    sim_result = leogo_expected_result
    optimisation_model = oogeso.OptimisationModel(leogo_test_data)
    op.plot_device_profile(sim_result, optimisation_model, devs=["wind"], include_forecasts=True)
    op.plot_device_profile(
        sim_result, optimisation_model, devs=["Gen1", "Gen2", "Gen3"], include_on_off=True, include_prep=True
    )


def test_plot_df(leogo_expected_result: dto.SimulationResult):
    sim_result = leogo_expected_result
    df = sim_result.device_flow
    df = df[:, :, "el", "out"].unstack("device")
    op.plot_df(df, id_var="device")


def test_plot_el_backup(leogo_expected_result: dto.SimulationResult):
    sim_result = leogo_expected_result
    op.plot_el_backup(sim_result, showMargin=True)


def test_plot_el_backup2(leogo_expected_result: dto.SimulationResult):
    sim_result = leogo_expected_result
    op.plot_el_backup2(sim_result)


def test_plot_export_revenue(leogo_expected_result: dto.SimulationResult):
    sim_result = leogo_expected_result
    op.plot_export_revenue(sim_result)


def test_plot_export_revenue():
    op.plot_gas_turbine_efficiency()


def test_plot_network(leogo_test_data: dto.EnergySystemData, leogo_expected_result: dto.SimulationResult):
    simulator = oogeso.Simulator(data=leogo_test_data)
    simulator.result_object = leogo_expected_result
    op.plot_network(simulator, timestep=1)
    # If no errors, it's OK


def test_plot_profiles(leogo_test_data: dto.EnergySystemData):
    op.plot_profiles(leogo_test_data.profiles)


def test_plot_reservek(leogo_test_data: dto.EnergySystemData, leogo_expected_result: dto.SimulationResult):
    sim_result = leogo_expected_result
    optimisation_model = oogeso.OptimisationModel(leogo_test_data)
    op.plot_reserve(sim_result, optimisation_model)


def test_plot_sum_power_mix(leogo_test_data: dto.EnergySystemData, leogo_expected_result: dto.SimulationResult):
    sim_result = leogo_expected_result
    optimisation_model = oogeso.OptimisationModel(leogo_test_data)
    op.plot_sum_power_mix(sim_result, optimisation_model, carrier="el")
