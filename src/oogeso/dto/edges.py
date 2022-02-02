from typing import List, Optional, Tuple, Union

from oogeso.dto import EdgeData
from oogeso.dto.types import CarrierType


class EdgeElData(EdgeData):
    resistance: float = 0  # ohm per km
    reactance: float = 0  # ohm per km
    # Voltage for line (single value) or transformer (tuple)
    voltage: Union[float, Tuple[float, float]] = None  # kV.
    # Power loss in MW as function of power transfer in MW:
    power_loss_function: Optional[Tuple[List[float], List[float]]] = None
    carrier: CarrierType = CarrierType.EL


class EdgeHeatData(EdgeData):
    # Heat loss in MW as function of energy transfer in MW:
    power_loss_function: Optional[Tuple[List[float], List[float]]] = None
    carrier: CarrierType = CarrierType.HEAT


class EdgeHydrogenData(EdgeData):
    bidirectional: bool = False
    carrier: CarrierType = CarrierType.HYDROGEN


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
    carrier: CarrierType = CarrierType.FLUID


class EdgeGasData(EdgeFluidData):
    carrier: CarrierType = CarrierType.GAS


class EdgeOilData(EdgeFluidData):
    carrier: CarrierType = CarrierType.OIL


class EdgeWellstreamData(EdgeFluidData):
    carrier: CarrierType = CarrierType.WELLSTREAM


class EdgeWaterData(EdgeFluidData):
    carrier: CarrierType = CarrierType.WATER
