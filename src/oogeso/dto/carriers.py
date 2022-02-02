from typing import Optional

from oogeso import dto
from oogeso.dto.types import PressureMethodType


class CarrierElData(dto.CarrierData):
    powerflow_method: str = "transport"  # "transport","dc_pf"
    reference_node: str = None  # reference node for dc-pf electrical voltage angles
    # required (globally) spinning reserve (MW), -1=no limit
    el_reserve_margin: float = -1
    # required backup margin (MW), -1=no limit
    el_backup_margin: Optional[float] = -1  # MW, -1=no limit
    # minutes, how long stored energy must be sustained to count as reserve:
    reserve_storage_minutes: Optional[int] = None


class CarrierHeatData(dto.CarrierData):
    pass


class CarrierHydrogenData(dto.CarrierData):
    energy_value: float = 13  # MJ/Sm3 (calorific value) -> 13 MJ/Sm3


class CarrierFluidData(dto.CarrierData):
    # Todo: Consider if we need one DTO for each fluid type, or if we can do with one DTO.
    rho_density: float  # kg/m3 -> 900 kg/m3
    viscosity: Optional[float] = None  # kg/(m s) -> 0.0026 kg/(m s)
    G_gravity: Optional[float] = None
    Z_compressibility: Optional[float] = None
    Tb_basetemp_K: Optional[float] = None
    Pb_basepressure_MPa: Optional[float] = None
    pressure_method: Optional[str] = None


class CarrierGasData(CarrierFluidData):
    co2_content: float = None  # kg/Sm3 - see SSB 2016 report -> 2.34 kg/Sm3
    R_individual_gas_constant: float = None  # J/(kg K) -> 500 J/kgK
    energy_value: float = None  # MJ/Sm3 (calorific value) -> 40 MJ/Sm3
    k_heat_capacity_ratio: float = None  # 1.27
    pressure_method: PressureMethodType = PressureMethodType.WEYMOUTH  # pressure drop calculation


class CarrierWellStreamData(CarrierFluidData):
    darcy_friction: float = None  # 0.02
    water_cut: float = None
    gas_oil_ratio: float = None


class CarrierOilData(CarrierFluidData):
    darcy_friction: float = None  # 0.02
    pressure_method: PressureMethodType = PressureMethodType.DARCY_WEISSBACH  # pressure drop calculation


class CarrierWaterData(CarrierFluidData):
    darcy_friction: float = None  # 0.02
    pressure_method: PressureMethodType = PressureMethodType.DARCY_WEISSBACH  # pressure drop calculation
