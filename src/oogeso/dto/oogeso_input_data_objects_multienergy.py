from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from oogeso import dto

# Device types defined as part of "basic":
# DevicePowerSourceData
# DeviceSink_elData
# DeviceStorage_elData


@dataclass
class DeviceSourceElData(dto.DeviceData):
    co2em: Optional[float] = None
    op_cost: Optional[float] = None
    reserve_factor: float = 1  # not used capacity contributes fully to spinning reserve


@dataclass
class DeviceSourceGasData(dto.DeviceData):
    naturalpressure: float = None


@dataclass
class DeviceSourceOilData(dto.DeviceData):
    naturalpressure: float = None


@dataclass
class DeviceSourceWaterData(dto.DeviceData):
    naturalpressure: float = None


@dataclass
class DeviceSinkElData(dto.DeviceData):
    pass


@dataclass
class DeviceSinkHeatData(dto.DeviceData):
    pass


@dataclass
class DeviceSinkGasData(dto.DeviceData):
    price: field(default_factory=lambda: {}) = None


@dataclass
class DeviceSinkOilData(dto.DeviceData):
    price: field(default_factory=lambda: {}) = None


@dataclass
class DeviceSinkWaterData(dto.DeviceData):
    price: field(default_factory=lambda: {}) = None
    flow_avg: Optional[float] = None  # required average flow
    max_accumulated_deviation: Optional[float] = None  # buffer size (max accumulated deviation from average)


@dataclass
class DeviceCompressorElData(dto.DeviceData):
    eta: float = None  # efficiency
    Q0: float = None  # nominal flow rate used in linearisation
    temp_in: float = None  # inlet temperature


@dataclass
class DeviceCompressorGasData(dto.DeviceData):
    eta: float = None  # efficiency
    Q0: float = None  # nominal flow rate used in linearisation
    temp_in: float = None  # inlet temperature


@dataclass
class DeviceElectrolyserData(dto.DeviceData):
    eta: float = None  # efficiency
    eta_heat: float = None  # heat recovery efficiency


@dataclass
class DeviceFuelCellData(dto.DeviceData):
    eta: float = None  # efficiency
    eta_heat: float = None  # heat recovery efficiency


@dataclass
class DeviceGasHeaterData(dto.DeviceData):
    eta: float = None  # efficiency


@dataclass
class DeviceGasTurbineData(dto.DeviceData):
    fuel_A: float = None
    fuel_B: float = None
    eta_heat: float = None
    reserve_factor: float = 1  # not used capacity contributes fully to spinning reserve


@dataclass
class DevicePumpData(dto.DeviceData):
    eta: float = None


@dataclass
class DeviceHeatPumpData(DevicePumpData):
    eta: float = None


@dataclass
class DevicePumpOilData(DevicePumpData):
    eta: float = None  # efficiency


@dataclass
class DevicePumpWaterData(DevicePumpData):
    eta: float = None


@dataclass
class DeviceSeparatorData(dto.DeviceData):
    el_demand_factor: float = None  # electricity demand factor
    heat_demand_factor: float = None  # heat demand factor


@dataclass
class DeviceSeparator2Data(dto.DeviceData):
    el_demand_factor: float = None  # electricity demand factor
    heat_demand_factor: float = None  # heat demand factor


@dataclass
class DeviceStorageHydrogenData(dto.DeviceData):
    E_max: float = 0  # MWh storage capacity (maximum stored energy)
    E_min: float = 0
    eta: float = 1  # efficiency
    target_profile: Optional[str] = None  # target profile for use of (seasonal) storage
    E_cost: float = 0  # cost for depleting storage
    E_init: float = 0


@dataclass
class DeviceWellProductionData(dto.DeviceData):
    wellhead_pressure: float = None  # 2 # MPa


@dataclass
class DeviceWellGasLiftData(dto.DeviceData):
    gas_oil_ratio: float = None  # 500
    water_cut: float = None  # 0.6
    f_inj: float = None  # 220 # gas injection rate as fraction of production rate
    injection_pressure: float = None  # 20 # MPa
    separator_pressure: float = None  # 2 # MPa


@dataclass
class EdgeHeatData(dto.EdgeData):
    # Heat loss in MW as function of energy transfer in MW:
    power_loss_function: Optional[Tuple[List[float], List[float]]] = None


@dataclass
class EdgeHydrogenData(dto.EdgeData):
    bidirectional: bool = False


@dataclass
class EdgeFluidData(dto.EdgeData):
    # wellstream, oil, water, gas
    pressure_from: float = None
    pressure_to: float = None
    diameter_mm: float = None
    temperature_K: float = None
    height_m: float = 0
    num_pipes: Optional[int] = None
    bidirectional: bool = False
    # allovable relative deviation of pressure from nominal values,
    pressure_from_maxdeviation: Optional[float] = None
    pressure_to_maxdeviation: Optional[float] = None


@dataclass
class EdgeGasData(EdgeFluidData):
    pass


@dataclass
class EdgeOilData(EdgeFluidData):
    pass


@dataclass
class EdgeWellstreamData(EdgeFluidData):
    pass


@dataclass
class EdgeWaterData(EdgeFluidData):
    pass


@dataclass
class CarrierHeatData(dto.CarrierData):
    pass


@dataclass
class CarrierHydrogenData(dto.CarrierData):
    energy_value: float = 13  # MJ/Sm3 (calorific value) -> 13 MJ/Sm3


@dataclass
class CarrierFluidData(dto.CarrierData):
    # Todo: Consider if we need one DTO for each fluid type, or if we can do with one DTO.
    rho_density: float  # kg/m3 -> 900 kg/m3
    viscosity: Optional[float] = None  # kg/(m s) -> 0.0026 kg/(m s)
    G_gravity: Optional[float] = None
    Z_compressibility: Optional[float] = None
    Tb_basetemp_K: Optional[float] = None
    Pb_basepressure_MPa: Optional[float] = None
    pressure_method: Optional[str] = None


@dataclass
class CarrierGasData(CarrierFluidData):
    co2_content: float = None  # kg/Sm3 - see SSB 2016 report -> 2.34 kg/Sm3
    R_individual_gas_constant: float = None  # J/(kg K) -> 500 J/kgK
    energy_value: float = None  # MJ/Sm3 (calorific value) -> 40 MJ/Sm3
    k_heat_capacity_ratio: float = None  # 1.27
    pressure_method: Optional[str] = "weymouth"  # pressure drop calculation


@dataclass
class CarrierWellStreamData(CarrierFluidData):
    darcy_friction: float = None  # 0.02
    water_cut: float = None
    gas_oil_ratio: float = None


@dataclass
class CarrierOilData(CarrierFluidData):
    darcy_friction: float = None  # 0.02
    pressure_method: Optional[str] = "darcy-weissbach"  # pressure drop calculation


@dataclass
class CarrierWaterData(CarrierFluidData):
    darcy_friction: float = None  # 0.02
    pressure_method: Optional[str] = "darcy-weissbach"  # pressure drop calculation
