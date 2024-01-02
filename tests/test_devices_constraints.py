import pyomo.environ as pyo

from oogeso import dto
from oogeso.core import OptimisationModel

dev_data_generic = {
    "id": "the_id",
    "node_id": "the_node",
    "name": "the_name",
    "include": True,
    "profile": "the_profile",
    "flow_min": 10,
    "flow_max": 20,
    "max_ramp_up": None,
    "max_ramp_down": None,
    "op_cost": 0,
}
startstop_data = dto.StartStopData(
    is_on_init=False,
    penalty_start=1,
    penalty_stop=0,
    delay_start_minutes=30,
    minimum_time_on_minutes=0,
    minimum_time_off_minutes=0,
)
data_profile = dto.TimeSeriesData(
    id="the_profile",
    data=[1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
    data_nowcast=[1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
)

# These tests, one for each device type, checks that the construction of Pyomo model
# constraints associated with the deviecs goes through without error.
# To test this, it is necessary to create a complete optimisation model with some
# necessary node, edge, carrier data as well.


def _build_lp_problem_with_single_dev(dev_data: dto.DeviceData):
    """Builds a minimal Pyomo model with a single device"""
    parameters = dto.OptimisationParametersData(
        objective="penalty",
        time_delta_minutes=15,
        planning_horizon=4,
        optimisation_timesteps=2,
        forecast_timesteps=2,
    )
    carriers = [
        dto.CarrierElData(id="el", reserve_storage_minutes=30),
        dto.CarrierGasData(
            id="gas",
            co2_content=2.34,
            Pb_basepressure_MPa=0.1,
            R_individual_gas_constant=500,
            Tb_basetemp_K=288,
            Z_compressibility=0.9,
            energy_value=40,
            k_heat_capacity_ratio=1.27,
            rho_density=0.8,
        ),
        dto.CarrierHeatData(id="heat"),
        dto.CarrierOilData(id="oil", darcy_friction=0.02, rho_density=900, viscosity=0.002),
        dto.CarrierWaterData(id="water", darcy_friction=0.01, rho_density=900, viscosity=0.01),
        dto.CarrierHydrogenData(id="hydrogen"),
        dto.CarrierCarbonData(id="carbon"),
        dto.CarrierWellStreamData(
            id="wellstream",
            darcy_friction=0.01,
            rho_density=900,
            viscosity=0.01,
            water_cut=0.6,
            gas_oil_ratio=500,
        ),
    ]

    nodes = [
        dto.NodeData(id="the_node"),
        dto.NodeData(id="another_node1"),
        dto.NodeData(id="another_node2"),
    ]
    # edges are needed to set nominal pressure, needed for compressor devices
    edges = [
        dto.EdgeGasData(
            id="edge1g",
            node_from="the_node",
            node_to="another_node1",
            pressure_from=1,
            pressure_to=1,
        ),
        dto.EdgeGasData(
            id="edge2g",
            node_from="another_node2",
            node_to="the_node",
            pressure_from=1,
            pressure_to=1,
        ),
        dto.EdgeWaterData(
            id="edge1w",
            node_from="the_node",
            node_to="another_node1",
            pressure_from=1,
            pressure_to=1,
        ),
        dto.EdgeWaterData(
            id="edge2w",
            node_from="another_node2",
            node_to="the_node",
            pressure_from=1,
            pressure_to=1,
        ),
        dto.EdgeOilData(
            id="edge1o",
            node_from="the_node",
            node_to="another_node1",
            pressure_from=1,
            pressure_to=1,
        ),
        dto.EdgeOilData(
            id="edge2o",
            node_from="another_node2",
            node_to="the_node",
            pressure_from=1,
            pressure_to=1,
        ),
    ]
    devs = [dev_data]
    profiles = [data_profile]
    energy_system_data = dto.EnergySystemData(
        parameters=parameters,
        carriers=carriers,
        nodes=nodes,
        edges=edges,
        devices=devs,
        profiles=profiles,
    )

    # Create and initialize optimisation model. This will create the constraints
    optimisation_model = OptimisationModel(energy_system_data)

    # Chack that all method calls works without error (not caring about return value)
    _device_method_calls(optimisation_model)

    return optimisation_model


def _device_method_calls(optimistion_model: OptimisationModel):
    """Calls all common device methods"""
    dev_obj = optimistion_model.all_devices["the_id"]
    timesteps = pyo.Set(initialize=[0])
    timesteps.construct()
    # Check that these calls don't give errors
    dev_obj.compute_cost_for_depleted_storage(optimistion_model, timesteps)
    dev_obj.compute_el_reserve(optimistion_model, t=0)
    dev_obj.compute_export(
        optimistion_model,
        value="volume",
        carriers=["oil", "gas", "el"],
        timesteps=timesteps,
    )
    dev_obj.compute_operating_costs(optimistion_model, timesteps)
    dev_obj.compute_penalty(optimistion_model, timesteps)
    dev_obj.compute_startup_penalty(optimistion_model, timesteps)
    dev_obj.get_flow_var(optimistion_model, t=0)
    dev_obj.get_max_flow(optimistion_model, t=0)
    dev_obj.get_flow_upper_bound()
    dev_obj.set_flow_upper_bound([data_profile])


def test_powersource_constraints():
    dev_data = dto.DevicePowerSourceData(
        **dev_data_generic, start_stop=startstop_data, penalty_function=([0, 50], [1, 20])
    )
    optimisation_model = _build_lp_problem_with_single_dev(dev_data)
    assert isinstance(optimisation_model, OptimisationModel)


def test_powersink_constraints():
    dev_data = dto.DevicePowerSinkData(**dev_data_generic)
    optimisation_model = _build_lp_problem_with_single_dev(dev_data)
    assert isinstance(optimisation_model, OptimisationModel)


def test_compressor_el_constraints():
    dev_data = dto.DeviceCompressorElData(**dev_data_generic, eta=0.6, Q0=5, temp_in=300)
    optimisation_model = _build_lp_problem_with_single_dev(dev_data)
    assert isinstance(optimisation_model, OptimisationModel)


def test_compressor_gas_constraints():
    dev_data = dto.DeviceCompressorGasData(**dev_data_generic, eta=0.6, Q0=5, temp_in=300)
    optimisation_model = _build_lp_problem_with_single_dev(dev_data)
    assert isinstance(optimisation_model, OptimisationModel)


def test_electrolyser_constraints():
    dev_data = dto.DeviceElectrolyserData(**dev_data_generic, eta=0.6, eta_heat=0.5)
    optimisation_model = _build_lp_problem_with_single_dev(dev_data)
    assert isinstance(optimisation_model, OptimisationModel)


def test_fuelcell_constraints():
    dev_data = dto.DeviceFuelCellData(**dev_data_generic, eta=0.6, eta_heat=0.5)
    optimisation_model = _build_lp_problem_with_single_dev(dev_data)
    assert isinstance(optimisation_model, OptimisationModel)


def test_gasheater_constraints():
    dev_data = dto.DeviceGasHeaterData(**dev_data_generic, eta=0.8)
    optimisation_model = _build_lp_problem_with_single_dev(dev_data)
    assert isinstance(optimisation_model, OptimisationModel)


def test_gasturbine_constraints():
    dev_data = dto.DeviceGasTurbineData(**dev_data_generic, start_stop=startstop_data, fuel_A=1, fuel_B=2, eta_heat=0.5)
    optimisation_model = _build_lp_problem_with_single_dev(dev_data)
    assert isinstance(optimisation_model, OptimisationModel)


def test_heatpump_constraints():
    dev_data = dto.DeviceHeatPumpData(**dev_data_generic, eta=3.0)
    optimisation_model = _build_lp_problem_with_single_dev(dev_data)
    assert isinstance(optimisation_model, OptimisationModel)


def test_pump_oil_constraints():
    dev_data = dto.DevicePumpOilData(**dev_data_generic, eta=0.6)
    optimisation_model = _build_lp_problem_with_single_dev(dev_data)
    assert isinstance(optimisation_model, OptimisationModel)


def test_pump_water_constraints():
    dev_data = dto.DevicePumpWaterData(**dev_data_generic, eta=0.6)
    optimisation_model = _build_lp_problem_with_single_dev(dev_data)
    assert isinstance(optimisation_model, OptimisationModel)


def test_separator_constraints():
    dev_data = dto.DeviceSeparatorData(**dev_data_generic, heat_demand_factor=0.5)
    optimisation_model = _build_lp_problem_with_single_dev(dev_data)
    assert isinstance(optimisation_model, OptimisationModel)


def test_separator2_constraints():
    dev_data = dto.DeviceSeparator2Data(**dev_data_generic, heat_demand_factor=0.5)
    optimisation_model = _build_lp_problem_with_single_dev(dev_data)
    assert isinstance(optimisation_model, OptimisationModel)


def test_sink_el_constraints():
    dev_data = dto.DeviceSinkElData(**dev_data_generic)
    optimisation_model = _build_lp_problem_with_single_dev(dev_data)
    assert isinstance(optimisation_model, OptimisationModel)


def test_sink_gas_constraints():
    dev_data = dto.DeviceSinkGasData(**dev_data_generic)
    optimisation_model = _build_lp_problem_with_single_dev(dev_data)
    assert isinstance(optimisation_model, OptimisationModel)


def test_sink_heat_constraints():
    dev_data = dto.DeviceSinkHeatData(**dev_data_generic)
    optimisation_model = _build_lp_problem_with_single_dev(dev_data)
    assert isinstance(optimisation_model, OptimisationModel)


def test_sink_oil_constraints():
    dev_data = dto.DeviceSinkOilData(**dev_data_generic)
    optimisation_model = _build_lp_problem_with_single_dev(dev_data)
    assert isinstance(optimisation_model, OptimisationModel)


def test_sink_water_constraints():
    dev_data = dto.DeviceSinkWaterData(**dev_data_generic)
    optimisation_model = _build_lp_problem_with_single_dev(dev_data)
    assert isinstance(optimisation_model, OptimisationModel)


def test_source_el_constraints():
    dev_data = dto.DeviceSourceElData(**dev_data_generic)
    optimisation_model = _build_lp_problem_with_single_dev(dev_data)
    assert isinstance(optimisation_model, OptimisationModel)


def test_source_gas_constraints():
    dev_data = dto.DeviceSourceGasData(**dev_data_generic, naturalpressure=10)
    optimisation_model = _build_lp_problem_with_single_dev(dev_data)
    assert isinstance(optimisation_model, OptimisationModel)


def test_source_oil_constraints():
    dev_data = dto.DeviceSourceOilData(**dev_data_generic, naturalpressure=10)
    optimisation_model = _build_lp_problem_with_single_dev(dev_data)
    assert isinstance(optimisation_model, OptimisationModel)


def test_source_water_constraints():
    dev_data = dto.DeviceSourceWaterData(**dev_data_generic, naturalpressure=1)
    optimisation_model = _build_lp_problem_with_single_dev(dev_data)
    assert isinstance(optimisation_model, OptimisationModel)


def test_storage_el_constraints():
    dev_data = dto.DeviceStorageElData(**dev_data_generic)
    optimisation_model = _build_lp_problem_with_single_dev(dev_data)
    assert isinstance(optimisation_model, OptimisationModel)


def test_storage_hydrogen_constraints():
    dev_data = dto.DeviceStorageHydrogenData(**dev_data_generic)
    optimisation_model = _build_lp_problem_with_single_dev(dev_data)
    assert isinstance(optimisation_model, OptimisationModel)


def test_well_gas_lift_constraints():
    dev_data = dto.DeviceWellGasLiftData(
        **dev_data_generic, separator_pressure=10, gas_oil_ratio=500, water_cut=0.6, f_inj=2, injection_pressure=20
    )
    optimisation_model = _build_lp_problem_with_single_dev(dev_data)
    assert isinstance(optimisation_model, OptimisationModel)


def test_well_production_constraints():
    dev_data = dto.DeviceWellProductionData(**dev_data_generic, wellhead_pressure=10)
    optimisation_model = _build_lp_problem_with_single_dev(dev_data)
    assert isinstance(optimisation_model, OptimisationModel)
