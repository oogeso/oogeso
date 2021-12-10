from oogeso import dto
from oogeso.core import devices

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

# Only Powersource and PowerSink are used in electric-only modelling


def test_powersource():
    startstop_data = dto.StartStopData(
        is_on_init=False,
        penalty_start=1,
        penalty_stop=0,
        delay_start_minutes=30,
        minimum_time_on_minutes=0,
        minimum_time_off_minutes=0,
    )
    dev_data = dto.DevicePowerSourceData(
        **dev_data_generic,
        start_stop=startstop_data,
        penalty_function=([0, 50], [1, 20]),
    )
    carrier_data_dict = {}
    obj = devices.Powersource(dev_data, carrier_data_dict)
    assert isinstance(obj, devices.Powersource)
    assert obj.dev_data.node_id == "the_node"
    assert obj.dev_data.penalty_function == ([0, 50], [1, 20])


def test_powersink():
    dev_data = dto.DevicePowerSinkData(**dev_data_generic, reserve_factor=0)
    carrier_data_dict = {}
    obj = devices.PowerSink(dev_data, carrier_data_dict)
    assert isinstance(obj, devices.PowerSink)
    assert obj.dev_data.reserve_factor == 0


# The following device models are only used in multi-energy modelling:


def test_gasturbine():
    startstop_data = dto.StartStopData(
        is_on_init=False,
        penalty_start=1,
        penalty_stop=0,
        delay_start_minutes=30,
        minimum_time_on_minutes=0,
        minimum_time_off_minutes=0,
    )
    dev_data = dto.DeviceGasTurbineData(**dev_data_generic, start_stop=startstop_data)
    carrier_data_dict = {}
    obj = devices.GasTurbine(dev_data, carrier_data_dict)
    assert isinstance(obj, devices.GasTurbine)
    assert obj.dev_data.reserve_factor == 1  # default value
    assert obj.dev_data.start_stop.penalty_start == 1


def test_compressor_el():
    dev_data = dto.DeviceCompressorElData(**dev_data_generic, eta=0.6, Q0=0.5, temp_in=300)
    carrier_data_dict = {}
    obj = devices.CompressorEl(dev_data, carrier_data_dict)
    assert isinstance(obj, devices.CompressorEl)
    assert obj.dev_data.eta == 0.6


def test_compressor_gas():
    dev_data = dto.DeviceCompressorGasData(**dev_data_generic, eta=0.6, Q0=0.5, temp_in=300)
    carrier_data_dict = {}
    obj = devices.CompressorGas(dev_data, carrier_data_dict)
    assert isinstance(obj, devices.CompressorGas)
    assert obj.dev_data.eta == 0.6


def test_pump_oil():
    dev_data = dto.DevicePumpOilData(**dev_data_generic, eta=0.6)
    carrier_data_dict = {}
    obj = devices.PumpOil(dev_data, carrier_data_dict)
    assert isinstance(obj, devices.PumpOil)
    assert obj.dev_data.eta == 0.6


def test_pump_water():
    dev_data = dto.DevicePumpWaterData(**dev_data_generic, eta=0.6)
    carrier_data_dict = {}
    obj = devices.PumpWater(dev_data, carrier_data_dict)
    assert isinstance(obj, devices.PumpWater)
    assert obj.dev_data.eta == 0.6


def test_separator():
    dev_data = dto.DeviceSeparatorData(**dev_data_generic, el_demand_factor=0.1, heat_demand_factor=0.5)
    carrier_data_dict = {}
    obj = devices.Separator(dev_data, carrier_data_dict)
    assert isinstance(obj, devices.Separator)
    assert obj.dev_data.heat_demand_factor == 0.5


def test_separator2():
    dev_data = dto.DeviceSeparator2Data(**dev_data_generic, el_demand_factor=0.1, heat_demand_factor=0.5)
    carrier_data_dict = {}
    obj = devices.Separator2(dev_data, carrier_data_dict)
    assert isinstance(obj, devices.Separator2)
    assert obj.dev_data.heat_demand_factor == 0.5


def test_well_production():
    dev_data = dto.DeviceWellProductionData(**dev_data_generic, wellhead_pressure=5)
    carrier_data_dict = {}
    obj = devices.WellProduction(dev_data, carrier_data_dict)
    assert isinstance(obj, devices.WellProduction)
    assert obj.dev_data.wellhead_pressure == 5


def test_well_gaslift():
    dev_data = dto.DeviceWellGasLiftData(
        **dev_data_generic,
        gas_oil_ratio=500,
        water_cut=0.5,
        f_inj=0.7,
        injection_pressure=25,
        separator_pressure=5,
    )
    carrier_data_dict = {}
    obj = devices.WellGasLift(dev_data, carrier_data_dict)
    assert isinstance(obj, devices.WellGasLift)
    assert obj.dev_data.injection_pressure == 25


def test_sink_gas():
    dev_data = dto.DeviceSinkGasData(**dev_data_generic, price={"gas": 100})
    carrier_data_dict = {}
    obj = devices.SinkGas(dev_data, carrier_data_dict)
    assert isinstance(obj, devices.SinkGas)
    assert obj.dev_data.flow_max == 20
    assert obj.dev_data.price == {"gas": 100}


def test_sink_oil():
    dev_data = dto.DeviceSinkOilData(**dev_data_generic, price={"oil": 100})
    carrier_data_dict = {}
    obj = devices.SinkOil(dev_data, carrier_data_dict)
    assert isinstance(obj, devices.SinkOil)
    assert obj.dev_data.flow_max == 20
    assert obj.dev_data.price == {"oil": 100}


def test_sink_water():
    dev_data = dto.DeviceSinkWaterData(**dev_data_generic, flow_avg=None, max_accumulated_deviation=None)
    carrier_data_dict = {}
    obj = devices.SinkWater(dev_data, carrier_data_dict)
    assert isinstance(obj, devices.SinkWater)
    assert obj.dev_data.flow_max == 20


def test_sink_heat():
    dev_data = dto.DeviceSinkHeatData(**dev_data_generic)
    carrier_data_dict = {}
    obj = devices.SinkHeat(dev_data, carrier_data_dict)
    assert isinstance(obj, devices.SinkHeat)
    assert obj.dev_data.flow_max == 20


def test_sink_el():
    dev_data = dto.DeviceSinkElData(**dev_data_generic)
    carrier_data_dict = {}
    obj = devices.SinkEl(dev_data, carrier_data_dict)
    assert isinstance(obj, devices.SinkEl)
    assert obj.dev_data.flow_max == 20


def test_source_gas():
    dev_data = dto.DeviceSourceGasData(**dev_data_generic, naturalpressure=15)
    carrier_data_dict = {}
    obj = devices.SourceGas(dev_data, carrier_data_dict)
    assert isinstance(obj, devices.SourceGas)
    assert obj.dev_data.flow_max == 20
    assert obj.dev_data.naturalpressure == 15


def test_source_water():
    dev_data = dto.DeviceSourceWaterData(**dev_data_generic, naturalpressure=15)
    carrier_data_dict = {}
    obj = devices.SourceWater(dev_data, carrier_data_dict)
    assert isinstance(obj, devices.SourceWater)
    assert obj.dev_data.flow_max == 20
    assert obj.dev_data.naturalpressure == 15


def test_source_el():
    # source_el is identical to powersource
    dev_data = dto.DeviceSourceElData(**dev_data_generic, co2em=1.5)
    carrier_data_dict = {}
    obj = devices.SourceEl(dev_data, carrier_data_dict)
    assert isinstance(obj, devices.SourceEl)
    assert obj.dev_data.flow_max == 20
    assert obj.dev_data.co2em == 1.5
    assert obj.dev_data.reserve_factor == 1


def test_storage_el():
    dev_data = dto.DeviceStorageElData(
        **dev_data_generic,
        E_max=10,
        E_min=0.2,
        E_cost=0,
        eta=0.9,
        target_profile="mytarget",
        E_end=5,
        E_init=2.5,
    )
    carrier_data_dict = {}
    obj = devices.StorageEl(dev_data, carrier_data_dict)
    assert isinstance(obj, devices.StorageEl)
    assert obj.dev_data.flow_max == 20
    assert obj.dev_data.E_end == 5


def test_storage_hydrogen():
    dev_data = dto.DeviceStorageHydrogenData(
        **dev_data_generic,
        E_max=10,
        E_min=0.2,
        E_cost=0,
        eta=0.9,
        target_profile="mytarget",
        E_init=4,
    )
    carrier_data_dict = {}
    obj = devices.StorageHydrogen(dev_data, carrier_data_dict)
    assert isinstance(obj, devices.StorageHydrogen)
    assert obj.dev_data.flow_max == 20
    assert obj.dev_data.target_profile == "mytarget"


def test_storage_gasheater():
    dev_data = dto.DeviceGasHeaterData(**dev_data_generic)
    carrier_data_dict = {}
    obj = devices.GasHeater(dev_data, carrier_data_dict)
    assert isinstance(obj, devices.GasHeater)
    assert obj.dev_data.flow_max == 20


def test_storage_heatpump():
    dev_data = dto.DeviceHeatPumpData(**dev_data_generic, eta=3)
    carrier_data_dict = {}
    obj = devices.HeatPump(dev_data, carrier_data_dict)
    assert isinstance(obj, devices.HeatPump)
    assert obj.dev_data.flow_max == 20
    assert obj.dev_data.eta == 3


def test_storage_electrolyser():
    dev_data = dto.DeviceElectrolyserData(**dev_data_generic, eta=0.5, eta_heat=0.3)
    carrier_data_dict = {}
    obj = devices.Electrolyser(dev_data, carrier_data_dict)
    assert isinstance(obj, devices.Electrolyser)
    assert obj.dev_data.flow_max == 20
    assert obj.dev_data.eta == 0.5


def test_storage_fuelcell():
    dev_data = dto.DeviceFuelCellData(**dev_data_generic, eta=0.5, eta_heat=0.3)
    carrier_data_dict = {}
    obj = devices.FuelCell(dev_data, carrier_data_dict)
    assert isinstance(obj, devices.FuelCell)
    assert obj.dev_data.flow_max == 20
    assert obj.dev_data.eta == 0.5
    assert obj.carrier_in == ["hydrogen"]
    assert obj.carrier_out == ["el", "heat"]
