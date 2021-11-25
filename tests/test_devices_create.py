from oogeso.core import devices
from oogeso import dto


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

# Only Powersource and Powersink are used in electric-only modelling


def test_powersource():
    startstop_data = dto.StartStopData(
        is_on_init=False,
        penalty_start=1,
        penalty_stop=0,
        delay_start_minutes=30,
        minimum_time_on_minutes=0,
        minimum_time_off_minutes=0,
    )
    dev_data = dto.DevicePowersourceData(
        **dev_data_generic,
        start_stop=startstop_data,
        penalty_function=[[0, 50], [1, 20]],
    )
    carrier_data_dict = {}
    obj = devices.Powersource(dev_data, carrier_data_dict)
    assert isinstance(obj, devices.Powersource)
    assert obj.dev_data.node_id == "the_node"
    assert obj.dev_data.penalty_function == [[0, 50], [1, 20]]


def test_powersink():
    dev_data = dto.DevicePowersinkData(**dev_data_generic, reserve_factor=0)
    carrier_data_dict = {}
    obj = devices.Powersink(dev_data, carrier_data_dict)
    assert isinstance(obj, devices.Powersink)
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
    dev_data = dto.DeviceGasturbineData(**dev_data_generic, start_stop=startstop_data)
    carrier_data_dict = {}
    obj = devices.Gasturbine(dev_data, carrier_data_dict)
    assert isinstance(obj, devices.Gasturbine)
    assert obj.dev_data.reserve_factor == 1  # default value
    assert obj.dev_data.start_stop.penalty_start == 1


def test_compressor_el():
    dev_data = dto.DeviceCompressor_elData(
        **dev_data_generic, eta=0.6, Q0=0.5, temp_in=300
    )
    carrier_data_dict = {}
    obj = devices.Compressor_el(dev_data, carrier_data_dict)
    assert isinstance(obj, devices.Compressor_el)
    assert obj.dev_data.eta == 0.6


def test_compressor_gas():
    dev_data = dto.DeviceCompressor_gasData(
        **dev_data_generic, eta=0.6, Q0=0.5, temp_in=300
    )
    carrier_data_dict = {}
    obj = devices.Compressor_gas(dev_data, carrier_data_dict)
    assert isinstance(obj, devices.Compressor_gas)
    assert obj.dev_data.eta == 0.6


def test_pump_oil():
    dev_data = dto.DevicePump_oilData(**dev_data_generic, eta=0.6)
    carrier_data_dict = {}
    obj = devices.Pump_oil(dev_data, carrier_data_dict)
    assert isinstance(obj, devices.Pump_oil)
    assert obj.dev_data.eta == 0.6


def test_pump_water():
    dev_data = dto.DevicePump_waterData(**dev_data_generic, eta=0.6)
    carrier_data_dict = {}
    obj = devices.Pump_water(dev_data, carrier_data_dict)
    assert isinstance(obj, devices.Pump_water)
    assert obj.dev_data.eta == 0.6


def test_separator():
    dev_data = dto.DeviceSeparatorData(
        **dev_data_generic, el_demand_factor=0.1, heat_demand_factor=0.5
    )
    carrier_data_dict = {}
    obj = devices.Separator(dev_data, carrier_data_dict)
    assert isinstance(obj, devices.Separator)
    assert obj.dev_data.heat_demand_factor == 0.5


def test_separator2():
    dev_data = dto.DeviceSeparator2Data(
        **dev_data_generic, el_demand_factor=0.1, heat_demand_factor=0.5
    )
    carrier_data_dict = {}
    obj = devices.Separator2(dev_data, carrier_data_dict)
    assert isinstance(obj, devices.Separator2)
    assert obj.dev_data.heat_demand_factor == 0.5


def test_well_production():
    dev_data = dto.DeviceWell_productionData(**dev_data_generic, wellhead_pressure=5)
    carrier_data_dict = {}
    obj = devices.Well_production(dev_data, carrier_data_dict)
    assert isinstance(obj, devices.Well_production)
    assert obj.dev_data.wellhead_pressure == 5


def test_well_gaslift():
    dev_data = dto.DeviceWell_gasliftData(
        **dev_data_generic,
        gas_oil_ratio=500,
        water_cut=0.5,
        f_inj=0.7,
        injection_pressure=25,
        separator_pressure=5,
    )
    carrier_data_dict = {}
    obj = devices.Well_gaslift(dev_data, carrier_data_dict)
    assert isinstance(obj, devices.Well_gaslift)
    assert obj.dev_data.injection_pressure == 25


def test_sink_gas():
    dev_data = dto.DeviceSink_gasData(**dev_data_generic, price={"gas": 100})
    carrier_data_dict = {}
    obj = devices.Sink_gas(dev_data, carrier_data_dict)
    assert isinstance(obj, devices.Sink_gas)
    assert obj.dev_data.flow_max == 20
    assert obj.dev_data.price == {"gas": 100}


def test_sink_oil():
    dev_data = dto.DeviceSink_oilData(**dev_data_generic, price={"oil": 100})
    carrier_data_dict = {}
    obj = devices.Sink_oil(dev_data, carrier_data_dict)
    assert isinstance(obj, devices.Sink_oil)
    assert obj.dev_data.flow_max == 20
    assert obj.dev_data.price == {"oil": 100}


def test_sink_water():
    dev_data = dto.DeviceSink_waterData(
        **dev_data_generic, flow_avg=None, max_accumulated_deviation=None
    )
    carrier_data_dict = {}
    obj = devices.Sink_water(dev_data, carrier_data_dict)
    assert isinstance(obj, devices.Sink_water)
    assert obj.dev_data.flow_max == 20


def test_sink_heat():
    dev_data = dto.DeviceSink_heatData(**dev_data_generic)
    carrier_data_dict = {}
    obj = devices.Sink_heat(dev_data, carrier_data_dict)
    assert isinstance(obj, devices.Sink_heat)
    assert obj.dev_data.flow_max == 20


def test_sink_el():
    dev_data = dto.DeviceSink_elData(**dev_data_generic)
    carrier_data_dict = {}
    obj = devices.Sink_el(dev_data, carrier_data_dict)
    assert isinstance(obj, devices.Sink_el)
    assert obj.dev_data.flow_max == 20


def test_source_gas():
    dev_data = dto.DeviceSource_gasData(**dev_data_generic, naturalpressure=15)
    carrier_data_dict = {}
    obj = devices.Source_gas(dev_data, carrier_data_dict)
    assert isinstance(obj, devices.Source_gas)
    assert obj.dev_data.flow_max == 20
    assert obj.dev_data.naturalpressure == 15


def test_source_water():
    dev_data = dto.DeviceSource_waterData(**dev_data_generic, naturalpressure=15)
    carrier_data_dict = {}
    obj = devices.Source_water(dev_data, carrier_data_dict)
    assert isinstance(obj, devices.Source_water)
    assert obj.dev_data.flow_max == 20
    assert obj.dev_data.naturalpressure == 15


def test_source_el():
    # source_el is identical to powersource
    dev_data = dto.DeviceSource_elData(**dev_data_generic, co2em=1.5)
    carrier_data_dict = {}
    obj = devices.Source_el(dev_data, carrier_data_dict)
    assert isinstance(obj, devices.Source_el)
    assert obj.dev_data.flow_max == 20
    assert obj.dev_data.co2em == 1.5
    assert obj.dev_data.reserve_factor == 1


def test_storage_el():
    dev_data = dto.DeviceStorage_elData(
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
    obj = devices.Storage_el(dev_data, carrier_data_dict)
    assert isinstance(obj, devices.Storage_el)
    assert obj.dev_data.flow_max == 20
    assert obj.dev_data.E_end == 5


def test_storage_hydrogen():
    dev_data = dto.DeviceStorage_hydrogenData(
        **dev_data_generic,
        E_max=10,
        E_min=0.2,
        E_cost=0,
        eta=0.9,
        target_profile="mytarget",
        E_init=4,
    )
    carrier_data_dict = {}
    obj = devices.Storage_hydrogen(dev_data, carrier_data_dict)
    assert isinstance(obj, devices.Storage_hydrogen)
    assert obj.dev_data.flow_max == 20
    assert obj.dev_data.target_profile == "mytarget"


def test_storage_gasheater():
    dev_data = dto.DeviceGasheaterData(**dev_data_generic)
    carrier_data_dict = {}
    obj = devices.Gasheater(dev_data, carrier_data_dict)
    assert isinstance(obj, devices.Gasheater)
    assert obj.dev_data.flow_max == 20


def test_storage_heatpump():
    dev_data = dto.DeviceHeatpumpData(**dev_data_generic, eta=3)
    carrier_data_dict = {}
    obj = devices.Heatpump(dev_data, carrier_data_dict)
    assert isinstance(obj, devices.Heatpump)
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
    dev_data = dto.DeviceFuelcellData(**dev_data_generic, eta=0.5, eta_heat=0.3)
    carrier_data_dict = {}
    obj = devices.Fuelcell(dev_data, carrier_data_dict)
    assert isinstance(obj, devices.Fuelcell)
    assert obj.dev_data.flow_max == 20
    assert obj.dev_data.eta == 0.5
    assert obj.carrier_in == ["hydrogen"]
    assert obj.carrier_out == ["el", "heat"]
