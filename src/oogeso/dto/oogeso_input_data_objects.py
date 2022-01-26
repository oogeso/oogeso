import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union


@dataclass
class NodeData:
    # unique identifier:
    id: str


@dataclass
class StartStopData:
    is_on_init: bool = False  # Initial on/off status
    penalty_start: float = 0  # Startup "cost"
    penalty_stop: float = 0  # Shutdown "cost"
    delay_start_minutes: int = 0  # Delay in minutes from activation to online
    minimum_time_on_minutes: float = 0  # Minimum on-time in minutes once started
    minimum_time_off_minutes: float = 0  # Minimum off-time in minutes once stopped


@dataclass
class DeviceData:  # Parent class - use subclasses instead
    id: str  # unique identifier
    node_id: str  # reference to node identifier
    name: str = ""
    include: Optional[bool] = True
    profile: Optional[str] = None  # reference to time-series
    flow_min: Optional[float] = None  # Energy or fluid flow limit
    flow_max: Optional[float] = None
    # Ramp rates are given as change relative to capacity per minute, 1=100%/min:
    max_ramp_down: Optional[float] = None
    max_ramp_up: Optional[float] = None
    start_stop: Optional[StartStopData] = None
    reserve_factor: float = 0  # contribution to electrical spinning reserve
    op_cost: Optional[float] = None
    # Penalty may be fuel, emissions, cost and combinations of these
    penalty_function: Optional[Tuple[List[float], List[float]]] = None
    model: str = field(init=False)  # model name is derived from class name

    def __post_init__(self):
        # Model name is given by class name. Eg "gasturbine" <-> "DeviceGasturbineData"
        # This field is added to know (when serialized) which subclass the device belongs to
        modelname = re.search("Device(.+?)Data", self.__class__.__name__).group(1)
        self.model = modelname.lower()


@dataclass
class DevicePowerSourceData(DeviceData):
    reserve_factor: float = 1  # not used capacity contributes fully to spinning reserve


@dataclass
class DevicePowerSinkData(DeviceData):
    pass


@dataclass
class DeviceStorageElData(DeviceData):
    E_max: float = 0  # MWh storage capacity
    E_min: float = 0
    E_end: Optional[float] = None  # required storage level at end of horzion
    E_cost: Optional[float] = None  # cost for non-ful storage
    eta: float = 1  # efficiency
    target_profile: Optional[str] = None
    E_init: Optional[float] = 0


@dataclass
class EdgeData:
    id: str
    node_from: str
    node_to: str
    length_km: Optional[float] = None
    flow_max: float = None  # Maximum flow (MW or Sm3/s)
    bidirectional: Optional[bool] = True
    include: bool = True  # whether to include object in problem formulation
    carrier: str = field(init=False)

    def __post_init__(self):
        carrier_name = re.search("Edge(.+?)Data", self.__class__.__name__).group(1)
        self.carrier = carrier_name.lower()


@dataclass
class EdgeElData(EdgeData):
    resistance: float = 0  # ohm per km
    reactance: float = 0  # ohm per km
    # Voltage for line (single value) or transformer (tuple)
    voltage: Union[float, Tuple[float, float]] = None  # kV.
    # Power loss in MW as function of power transfer in MW:
    power_loss_function: Optional[Tuple[List[float], List[float]]] = None


@dataclass
class CarrierData:
    id: str


@dataclass
class CarrierElData(CarrierData):
    powerflow_method: str = "transport"  # "transport","dc_pf"
    reference_node: str = None  # reference node for dc-pf electrical voltage angles
    # required (globally) spinning reserve (MW), -1=no limit
    el_reserve_margin: float = -1
    # required backup margin (MW), -1=no limit
    el_backup_margin: Optional[float] = -1  # MW, -1=no limit
    # minutes, how long stored energy must be sustained to count as reserve:
    reserve_storage_minutes: Optional[int] = None


@dataclass
class OptimisationParametersData:
    # name of objective function to use:
    objective: str
    # minutes per timestep:
    time_delta_minutes: int
    # timesteps in each rolling optimisation:
    planning_horizon: int
    # timesteps between each optimisation:
    optimisation_timesteps: int
    # timesteps beyond which forecast (instead of nowcast) profile is used:
    forecast_timesteps: int
    # costs for co2 emissions (currency/kgCO2)
    co2_tax: Optional[float] = None
    # limit on allowable emission intensity (kgCO2/Sm3oe), -1=no limit
    emission_intensity_max: Optional[float] = -1
    # limit on allowable emission intensity (kgCO2/hour), -1= no limit
    emission_rate_max: Optional[float] = -1
    # how to represent piecewise linear constraints:
    piecewise_repn: str = "SOS2"
    # specify which data to return from simulation as a list. None=return all data
    optimisation_return_data: Optional[List[str]] = None


TimeSeries = List[float]


@dataclass
class TimeSeriesData:
    id: str
    data: TimeSeries
    data_nowcast: Optional[TimeSeries] = None


@dataclass
class EnergySystemData:
    parameters: OptimisationParametersData
    carriers: List[CarrierData]
    nodes: List[NodeData]
    edges: List[EdgeData]
    devices: List[DeviceData]
    profiles: List[TimeSeriesData]
