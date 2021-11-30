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
class DeviceFuelcellData(dto.DeviceData):
    eta: float = None  # efficiency
    eta_heat: float = None  # heat recovery efficiency


@dataclass
class DeviceGasHeaterData(dto.DeviceData):
    pass


@dataclass
class DeviceGasTurbineData(dto.DeviceData):
    fuel_A: float = None
    fuel_B: float = None
    eta_heat: float = None
    #    is_on_init: bool = False
    #    startup_cost: float = None
    #    startup_delay: float = None  # Minutes from activation to power delivery
    #    shutdown_cost: float = None
    reserve_factor: float = 1  # not used capacity contributes fully to spinning reserve


@dataclass
class DeviceHeatPumpData(dto.DeviceData):
    eta: float = None


@dataclass
class DevicePumpOilData(dto.DeviceData):
    eta: float = None  # efficiency


@dataclass
class DevicePumpWaterData(dto.DeviceData):
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
class CarrierGasData(dto.CarrierData):
    co2_content: float  # kg/Sm3 - see SSB 2016 report -> 2.34 kg/Sm3
    G_gravity: float  # 0.6
    Pb_basepressure_MPa: float  # MPa -> 0.101 # MPa
    R_individual_gas_constant: float  # J/(kg K) -> 500 J/kgK
    Tb_basetemp_K: float  # K -> 288 K = 15 degC
    Z_compressibility: float  # 0.9
    energy_value: float  # MJ/Sm3 (calorific value) -> 40 MJ/Sm3
    k_heat_capacity_ratio: float  # 1.27
    rho_density: float  # kg/m3 -> 0.84 kg/m3
    pressure_method: Optional[str] = "weymouth"  # pressure drop calculation


@dataclass
class CarrierWellstreamData(dto.CarrierData):
    darcy_friction: float = None  # 0.02
    rho_density: float = None  # kg/m3 -> 900 kg/m3
    viscosity: float = None  # kg/(m s) -> 0.0026 kg/(m s)
    pressure_method: Optional[str] = None
    water_cut: float = None
    gas_oil_ratio: float = None


@dataclass
class CarrierOilData(dto.CarrierData):
    darcy_friction: float = None  # 0.02
    rho_density: float = None  # kg/m3 -> 900 kg/m3
    viscosity: float = None  # kg/(m s) -> 0.0026 kg/(m s)
    pressure_method: Optional[str] = "darcy-weissbach"  # pressure drop calculation


@dataclass
class CarrierWaterData(dto.CarrierData):
    darcy_friction: float = None  # 0.02
    rho_density: float = None  # kg/m3 -> 900 kg/m3
    viscosity: float = None  # kg/(m s) -> 0.0026 kg/(m s)
    pressure_method: Optional[str] = "darcy-weissbach"  # pressure drop calculation
