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
- time-series profiles (e.g. for variable energy demand)
"""


@dataclass
class NodeData:
    """Network node data

    Attributes:
        id (str): Identifier string
        maxdeviation_pressure (dict): Relative pressure deviation,
          specified as dictionary, example {"gas":0.1, "water":0.1}
    """

    id: str  # unique identifier
    maxdeviation_pressure: Optional[Dict] = field(default_factory=lambda: {})


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
    model: str = field(init=False)  # model name is derived from class name

    def __post_init__(self):
        # Model name is given by class name. Eg "gasturbine" <-> "DeviceGasturbineData"
        # This field is added to know (when serialized) which subclass the device belongs to
        modelname = re.search("Device(.+?)Data", self.__class__.__name__).group(1)
        self.model = modelname.lower()


@dataclass
class DevicePowersourceData(DeviceData):
    # Penalty may be fuel, emissions, cost and combinations of these
    penalty_function: Tuple[List[float], List[float]] = None
    reserve_factor: float = 1  # not used capacity contributes fully to spinning reserve


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
    pass


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
class DeviceStorage_elData(DeviceData):
    E_max: float = 0  # MWh storage capacity
    E_min: float = 0
    E_end: Optional[float] = None  # required storage level at end of horzion
    E_cost: Optional[float] = None  # cost for non-ful storage
    eta: float = 1  # efficiency
    target_profile: Optional[str] = None
    E_init: Optional[float] = 0


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
class CarrierData:
    id: str


@dataclass
class CarrierElData(CarrierData):
    powerflow_method: str = "dc_pf"  # "transport","dc_pf"
    reference_node: str = None  # reference node for dc-pf electrical voltage angles
    # required (globally) spinning reserve (MW), -1=no limit
    el_reserve_margin: float = -1
    # required backup margin (MW), -1=no limit
    el_backup_margin: Optional[float] = -1  # MW, -1=no limit


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
    # minutes, how long stored energy must be sustained to count as reserve:
    time_reserve_minutes: Optional[int] = None
    # costs for co2 emissions (currency/kgCO2)
    co2_tax: Optional[float] = None
    # global limit for allowable relative pressure deviation from nominal:
    max_pressure_deviation: float = -1
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
    # devices: Dict[str, DeviceData]
    devices: List[DeviceData]
    profiles: List[TimeSeriesData]
    # profiles: Dict[str, TimeSeries]
