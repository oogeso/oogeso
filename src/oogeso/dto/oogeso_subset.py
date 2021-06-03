import json
from dataclasses import dataclass, is_dataclass, asdict, field
from typing import List, Optional, Tuple, Any, Dict, Union
import re

"""
Energy system data - consists of:
- list of nodes
- list of energy carriers with their properties ("el", "gas", etc.)
- list of edges (of different types) (power cables, pipes)
- list of devices (of different types)
- optimisation parameters

Time-series profiles (e.g. for variable energy demand) are provied separately.
"""


@dataclass
class NodeData:
    id: str


@dataclass
class DeviceData:  # Parent class - use subclasses instead
    id: str
    node_id: str
    name: str = ""
    include: Optional[bool] = True
    profile: Optional[str] = None  # reference to time-series
    flow_min: Optional[float] = None  # Energy or fluid flow limit
    flow_max: Optional[float] = None
    max_ramp_down: Optional[float] = None
    max_ramp_up: Optional[float] = None
    reserve_factor: float = 0  # contribution to electrical spinning reserve
    op_cost: Optional[float] = None
    model: str = field(init=False)  # model name is derived from class name

    def __post_init__(self):
        # Model name is given by class name. Eg "gasturbine" <-> "DeviceGasturbineData"
        # This field is added to know (when serialized) which subclass the device belongs to
        modelname = re.search("Device(.+?)Data", self.__class__.__name__).group(1)
        self.model = modelname.lower()


@dataclass
class DevicePowersourceData(DeviceData):
    power_to_penalty_data: Tuple[
        List[float], List[float]
    ] = None  # Penalty may be fuel, emissions, cost and combinations of these
    reserve_factor: float = 1  # not used capacity contributes fully to spinning reserve


@dataclass
class DeviceSource_elData(DeviceData):
    co2em: Optional[float] = None
    op_cost: Optional[float] = None
    reserve_factor: float = 1  # not used capacity contributes fully to spinning reserve


@dataclass
class DeviceSink_elData(
    DeviceData
):  # Electricity demand (use flow_min, flow_max, profile)
    pass


@dataclass
class DeviceSink_heatData(
    DeviceData
):  # Electricity demand (use flow_min, flow_max, profile)
    pass


@dataclass
class DeviceGasturbineData(DeviceData):
    fuel_A: float = None
    fuel_B: float = None
    eta_heat: float = None
    is_on_init: bool = False
    startup_delay: float = None  # Minutes from activation to power delivery
    startup_cost: float = None
    shutdown_cost: float = None
    reserve_factor: float = 1  # not used capacity contributes fully to spinning reserve


@dataclass
class DeviceHeatpumpData(DeviceData):
    pass


@dataclass
class DeviceStorage_elData(DeviceData):
    E_max: float = 0  # MWh storage capacity
    E_min: float = 0
    E_end: Optional[float] = None  # required storage level at end of horzion
    E_cost: Optional[float] = None  # cost for non-ful storage
    eta: float = 1  # efficiency
    target_profile: Optional[str] = None


@dataclass
class EdgeData:
    id: str
    node_from: str
    node_to: str
    length_km: float
    flow_max: float = None  # Maximum flow (MW or Sm3/s)
    bidirectional: Optional[bool] = True
    include: bool = True  # whether to include object in problem formulation
    model: str = field(init=False)

    def __post_init__(self):
        modelname = re.search("Edge(.+?)Data", self.__class__.__name__).group(1)
        self.model = modelname.lower()


@dataclass
class EdgeElData(EdgeData):
    resistance: float = 0  # ohm per km
    reactance: float = 0  # ohm per km
    # Voltage for line (single value) or transformer (tuple)
    voltage: Union[float, Tuple] = None  # kV.
    power_loss_function: Optional[Tuple[List[float], List[float]]] = None


@dataclass
class EdgeHeatData(EdgeData):
    power_loss_function: Optional[Tuple[List[float], List[float]]] = None


@dataclass
class CarrierData:
    id: str


@dataclass
class CarrierElData(CarrierData):
    powerflow_method: str = "dc_pf"  # "transport","dc_pf"


@dataclass
class OptimisationParametersData:
    objective: str  # name of objective function to use
    time_delta_minutes: int  # minutes per timestep
    planning_horizon: int  # timesteps in each rolling optimisation
    optimisation_timesteps: int  # timesteps between each optimisation
    forecast_timesteps: int  # timesteps beyond which forecast (instead of nowcast) profile is used
    time_reserve_minutes: int  # minutes, how long stored energy must be sustained to count as reserve
    co2_tax: float  # currency/kgCO2
    el_reserve_margin: float  # MWm -1=no limit
    max_pressure_deviation: float  # global limit for allowable relative pressure deviation from nominal
    reference_node: str = None  # node used as reference for electrical voltage angle
    el_backup_margin: Optional[float] = -1  # MW, -1=no limit
    emission_intensity_max: Optional[float] = -1  # kgCO2/Sm3oe, -1=no limit
    emission_rate_max: Optional[float] = -1  # kgCO2/hour, -1=no limit


@dataclass
class EnergySystemData:
    parameters: OptimisationParametersData
    carriers: List[CarrierData]
    nodes: List[NodeData]
    edges: List[EdgeData]
    devices: Dict[str, DeviceData]
