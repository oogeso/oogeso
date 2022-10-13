from typing import Dict, Optional

from pydantic import Field

from oogeso.dto import DeviceData
from oogeso.dto.types import ModelType


class DevicePowerSourceData(DeviceData):
    reserve_factor: float = 1  # not used capacity contributes fully to spinning reserve
    model: ModelType = ModelType.POWER_SOURCE


class DevicePowerSinkData(DeviceData):
    model: ModelType = ModelType.POWER_SINK


class DeviceStorageElData(DeviceData):
    E_max: float = 0  # MWh storage capacity
    E_min: float = 0
    E_end: Optional[float] = None  # required storage level at end of horzion
    E_cost: Optional[float] = None  # cost for non-ful storage
    eta: float = 1  # efficiency
    target_profile: Optional[str] = None
    E_init: Optional[float] = 0
    model: ModelType = ModelType.STORAGE_EL


class DeviceSourceElData(DeviceData):
    co2em: Optional[float] = None
    op_cost: Optional[float] = None
    reserve_factor: float = 1  # not used capacity contributes fully to spinning reserve
    model: ModelType = ModelType.SOURCE_EL


class DeviceSourceGasData(DeviceData):
    naturalpressure: float = None
    model: ModelType = ModelType.SOURCE_GAS


class DeviceSourceOilData(DeviceData):
    naturalpressure: float = None
    model: ModelType = ModelType.SOURCE_OIL


class DeviceSourceWaterData(DeviceData):
    naturalpressure: float = None
    model: ModelType = ModelType.SOURCE_WATER


class DeviceSinkElData(DeviceData):
    model: ModelType = ModelType.SINK_EL


class DeviceSinkHeatData(DeviceData):
    model: ModelType = ModelType.SINK_HEAT


class DeviceSinkGasData(DeviceData):
    price: Dict[str, float] = Field(default_factory=lambda: {})
    model: ModelType = ModelType.SINK_GAS


class DeviceSinkOilData(DeviceData):
    price: Dict[str, float] = Field(default_factory=lambda: {})
    model: ModelType = ModelType.SINK_OIL


class DeviceSinkWaterData(DeviceData):
    price: Dict[str, float] = Field(default_factory=lambda: {})
    flow_avg: Optional[float] = None  # required average flow
    max_accumulated_deviation: Optional[float] = None  # buffer size (max accumulated deviation from average)
    model: ModelType = ModelType.SINK_WATER


class DeviceCompressorElData(DeviceData):
    eta: float = None  # efficiency
    Q0: float = None  # nominal flow rate used in linearisation
    temp_in: float = None  # inlet temperature
    model: ModelType = ModelType.COMPRESSOR_EL


class DeviceCompressorGasData(DeviceData):
    eta: float = None  # efficiency
    Q0: float = None  # nominal flow rate used in linearisation
    temp_in: float = None  # inlet temperature
    model: ModelType = ModelType.COMPRESSOR_GAS


class DeviceElectrolyserData(DeviceData):
    eta: float = None  # efficiency
    eta_heat: float = None  # heat recovery efficiency
    model: ModelType = ModelType.ELECTROLYSER


class DeviceFuelCellData(DeviceData):
    eta: float = None  # efficiency
    eta_heat: float = None  # heat recovery efficiency
    model: ModelType = ModelType.FUEL_CELL


class DeviceGasHeaterData(DeviceData):
    eta: float = None  # efficiency
    model: ModelType = ModelType.GAS_HEATER


class DeviceGasTurbineData(DeviceData):
    fuel_A: float = None
    fuel_B: float = None
    eta_heat: float = None
    #    is_on_init: bool = False
    #    startup_cost: float = None
    #    startup_delay: float = None  # Minutes from activation to power delivery
    #    shutdown_cost: float = None
    reserve_factor: float = 1  # not used capacity contributes fully to spinning reserve
    model: ModelType = ModelType.GAS_TURBINE


class DevicePumpData(DeviceData):
    eta: float = None
    model: ModelType = ModelType.PUMP


class DeviceHeatPumpData(DevicePumpData):
    eta: float = None
    model: ModelType = ModelType.HEAT_PUMP


class DevicePumpOilData(DevicePumpData):
    eta: float = None  # efficiency
    model: ModelType = ModelType.PUMP_OIL


class DevicePumpWaterData(DevicePumpData):
    eta: float = None
    model: ModelType = ModelType.PUMP_WATER


class DeviceSeparatorData(DeviceData):
    heat_demand_factor: float = None  # heat demand factor
    model: ModelType = ModelType.SEPARATOR


class DeviceSeparator2Data(DeviceData):
    heat_demand_factor: float = None  # heat demand factor
    model: ModelType = ModelType.SEPARATOR2


class DeviceStorageHydrogenData(DeviceData):
    E_max: float = 0  # MWh storage capacity (maximum stored energy)
    E_min: float = 0
    eta: float = 1  # efficiency
    target_profile: Optional[str] = None  # target profile for use of (seasonal) storage
    E_cost: float = 0  # cost for depleting storage
    E_init: float = 0
    model: ModelType = ModelType.STORAGE_HYDROGEN


class DeviceWellProductionData(DeviceData):
    wellhead_pressure: float = None  # 2 # MPa
    model: ModelType = ModelType.WELL_PRODUCTION


class DeviceWellGasLiftData(DeviceData):
    gas_oil_ratio: float = None  # 500
    water_cut: float = None  # 0.6
    f_inj: float = None  # 220 # gas injection rate as fraction of production rate
    injection_pressure: float = None  # 20 # MPa
    separator_pressure: float = None  # 2 # MPa
    model: ModelType = ModelType.WELL_GAS_LIFT
