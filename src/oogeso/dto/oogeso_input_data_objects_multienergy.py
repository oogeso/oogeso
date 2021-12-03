from dataclasses import dataclass, is_dataclass, asdict, field
from typing import List, Optional, Tuple, Any, Dict, Union
import re
from .oogeso_input_data_objects import *


# Device types defined as part of "basic":
# DevicePowerSourceData
# DeviceSink_elData
# DeviceStorage_elData


@dataclass
class DeviceSource_elData(DeviceData):
    co2em: Optional[float] = None
    op_cost: Optional[float] = None
    reserve_factor: float = 1  # not used capacity contributes fully to spinning reserve


@dataclass
class DeviceSource_gasData(DeviceData):
    naturalpressure: float = None


@dataclass
class DeviceSource_waterData(DeviceData):
    naturalpressure: float = None


@dataclass
class DeviceSink_elData(DeviceData):
    pass


@dataclass
class DeviceSink_heatData(DeviceData):
    pass


@dataclass
class DeviceSink_gasData(DeviceData):
    price: field(default_factory=lambda: {}) = None


@dataclass
class DeviceSink_oilData(DeviceData):
    price: field(default_factory=lambda: {}) = None


@dataclass
class DeviceSink_waterData(DeviceData):
    price: field(default_factory=lambda: {}) = None
    flow_avg: Optional[float] = None  # required average flow
    max_accumulated_deviation: Optional[
        float
    ] = None  # buffer size (max accumulated deviation from average)


@dataclass
class DeviceCompressor_elData(DeviceData):
    eta: float = None  # efficiency
    Q0: float = None  # nominal flow rate used in linearisation
    temp_in: float = None  # inlet temperature


@dataclass
class DeviceCompressor_gasData(DeviceData):
    eta: float = None  # efficiency
    Q0: float = None  # nominal flow rate used in linearisation
    temp_in: float = None  # inlet temperature


@dataclass
class DeviceElectrolyserData(DeviceData):
    eta: float = None  # efficiency
    eta_heat: float = None  # heat recovery efficiency


@dataclass
class DeviceFuelcellData(DeviceData):
    eta: float = None  # efficiency
    eta_heat: float = None  # heat recovery efficiency


@dataclass
class DeviceGasheaterData(DeviceData):
    pass


@dataclass
class DeviceGasturbineData(DeviceData):
    fuel_A: float = None
    fuel_B: float = None
    eta_heat: float = None
    #    is_on_init: bool = False
    #    startup_cost: float = None
    #    startup_delay: float = None  # Minutes from activation to power delivery
    #    shutdown_cost: float = None
    reserve_factor: float = 1  # not used capacity contributes fully to spinning reserve


@dataclass
class DeviceHeatpumpData(DeviceData):
    eta: float = None


@dataclass
class DevicePump_oilData(DeviceData):
    eta: float = None  # efficiency


@dataclass
class DevicePump_waterData(DeviceData):
    eta: float = None


@dataclass
class DeviceSeparatorData(DeviceData):
    el_demand_factor: float = None  # electricity demand factor
    heat_demand_factor: float = None  # heat demand factor


@dataclass
class DeviceSeparator2Data(DeviceData):
    el_demand_factor: float = None  # electricity demand factor
    heat_demand_factor: float = None  # heat demand factor


@dataclass
class DeviceStorage_hydrogenData(DeviceData):
    E_max: float = 0  # MWh storage capacity (maximum stored energy)
    E_min: float = 0
    eta: float = 1  # efficiency
    target_profile: Optional[str] = None  # target profile for use of (seasonal) storage
    E_cost: float = 0  # cost for depleting storage
    E_init: float = 0


@dataclass
class DeviceWell_productionData(DeviceData):
    wellhead_pressure: float = None  # 2 # MPa


@dataclass
class DeviceWell_gasliftData(DeviceData):
    gas_oil_ratio: float = None  # 500
    water_cut: float = None  # 0.6
    f_inj: float = None  # 220 # gas injection rate as fraction of production rate
    injection_pressure: float = None  # 20 # MPa
    separator_pressure: float = None  # 2 # MPa


@dataclass
class EdgeHeatData(EdgeData):
    # Heat loss in MW as function of energy transfer in MW:
    power_loss_function: Optional[Tuple[List[float], List[float]]] = None


@dataclass
class EdgeHydrogenData(EdgeData):
    bidirectional: bool = False


@dataclass
class EdgeFluidData(EdgeData):
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
class CarrierHeatData(CarrierData):
    pass


@dataclass
class CarrierHydrogenData(CarrierData):
    energy_value: float = 13  # MJ/Sm3 (calorific value) -> 13 MJ/Sm3


@dataclass
class CarrierGasData(CarrierData):
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
class CarrierWellstreamData(CarrierData):
    darcy_friction: float = None  # 0.02
    rho_density: float = None  # kg/m3 -> 900 kg/m3
    viscosity: float = None  # kg/(m s) -> 0.0026 kg/(m s)
    pressure_method: Optional[str] = None
    water_cut: float = None
    gas_oil_ratio: float = None


@dataclass
class CarrierOilData(CarrierData):
    darcy_friction: float = None  # 0.02
    rho_density: float = None  # kg/m3 -> 900 kg/m3
    viscosity: float = None  # kg/(m s) -> 0.0026 kg/(m s)
    pressure_method: Optional[str] = "darcy-weissbach"  # pressure drop calculation


@dataclass
class CarrierWaterData(CarrierData):
    darcy_friction: float = None  # 0.02
    rho_density: float = None  # kg/m3 -> 900 kg/m3
    viscosity: float = None  # kg/(m s) -> 0.0026 kg/(m s)
    pressure_method: Optional[str] = "darcy-weissbach"  # pressure drop calculation
