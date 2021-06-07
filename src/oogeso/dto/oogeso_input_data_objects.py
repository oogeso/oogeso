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
- time-series profiles (e.g. for variable energy demand)
"""


def _default_pressures():
    carrier_with_pressure = ["gas", "oil", "water"]
    x = {c: {t: None for t in ["in", "out"]} for c in carrier_with_pressure}
    return x


@dataclass
class NodeData:
    id: str
    maxdeviation_pressure: Optional[Dict] = field(default_factory=lambda: {})


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
    is_on_init: bool = False
    startup_cost: float = None
    startup_delay: float = None  # Minutes from activation to power delivery
    shutdown_cost: float = None
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


@dataclass
class DeviceStorage_hydrogenData(DeviceData):
    E_max: float = 0  # MWh storage capacity (maximum stored energy)
    E_min: float = 0
    eta: float = 1  # efficiency
    target_profile: Optional[str] = None  # target profile for use of (seasonal) storage
    E_cost: float = 0  # cost for depleting storage


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


@dataclass
class CarrierHeatData(CarrierData):
    pass


@dataclass
class CarrierHydrogenData(CarrierData):
    pass


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


TimeSeries = Dict[str, float]  # sring key is timestamp in iso format


@dataclass
class EnergySystemData:
    parameters: OptimisationParametersData
    carriers: List[CarrierData]
    nodes: List[NodeData]
    edges: List[EdgeData]
    devices: Dict[str, DeviceData]
    profiles: Dict[str, TimeSeries]
    # "nowcast" is near real-time updated forecast:
    profiles_nowcast: Optional[Dict[str, TimeSeries]] = None


# Serialize (save to file)
class DataclassJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any):
        if is_dataclass(obj=obj):
            dct = asdict(obj=obj)
            return dct
        return super().default(obj)


# Deserialize (read from file)
class DataclassJSONDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def _newCarrier(self, dct):
        model = dct["id"]
        carrier_class_str = "Carrier{}Data".format(model.capitalize())
        carrier_class = globals()[carrier_class_str]
        return carrier_class(**dct)

    def _newNode(self, dct):
        return NodeData(**dct)

    def _newEdge(self, dct):
        model = dct.pop("model")  # gets and deletes model from dictionary
        edge_class_str = "Edge{}Data".format(model.capitalize())
        edge_class = globals()[edge_class_str]
        return edge_class(**dct)

    def _newDevice(self, dct):
        model = dct.pop("model")  # gets and deletes model from dictionary
        dev_class_str = "Device{}Data".format(model.capitalize())
        dev_class = globals()[dev_class_str]
        return dev_class(**dct)

    def _newProfile(self, dct: TimeSeries):
        return dct

    def object_hook(self, dct):
        carriers = []
        nodes = []
        edges = []
        devs = {}
        profiles = {}
        profiles_nowcast = {}
        if "nodes" in dct:
            # Top level
            d_params = dct["parameters"]
            params = OptimisationParametersData(**d_params)
            for n in dct["carriers"]:
                carriers.append(self._newCarrier(n))
            for n in dct["nodes"]:
                nodes.append(self._newNode(n))
            for n in dct["edges"]:
                edges.append(self._newEdge(n))
            for n in dct["devices"]:
                devs[n["id"]] = self._newDevice(n)
            if "profiles" in dct:
                for i, n in dct["profiles"].items():
                    profiles[i] = self._newProfile(n)
            if "profiles_nowcast" in dct:
                for i, n in dct["profiles_nowcast"].items():
                    profiles_nowcast[i] = self._newProfile(n)
            energy_system_data = EnergySystemData(
                carriers=carriers,
                nodes=nodes,
                edges=edges,
                devices=devs,
                parameters=params,
                profiles=profiles,
                profiles_nowcast=profiles_nowcast,
            )
            return energy_system_data
        return dct

    def default(self, obj: Any):
        if isinstance(obj, EnergySystemData):
            return EnergySystemData(obj=obj)
        return super().default(obj)


def serialize_oogeso_data(energy_system_data: EnergySystemData):
    json_string = json.dumps(energy_system_data, cls=DataclassJSONEncoder, indent=2)
    return json_string


def deserialize_oogeso_data(json_data):
    energy_system_data = json.loads(json_data, cls=DataclassJSONDecoder)
    return energy_system_data


# Example:
energy_system = EnergySystemData(
    carriers=[
        CarrierElData(id="el"),
        CarrierHeatData("heat"),
        CarrierGasData(
            "gas",
            co2_content=0.4,
            G_gravity=0.6,
            Pb_basepressure_MPa=100,
            R_individual_gas_constant=9,
            Tb_basetemp_K=300,
            Z_compressibility=0.9,
            energy_value=40,
            k_heat_capacity_ratio=0.7,
            rho_density=0.6,
        ),
    ],
    nodes=[NodeData("node1"), NodeData("node2")],
    edges=[
        EdgeElData(
            id="edge1",
            node_from="node1",
            node_to="node2",
            flow_max=500,
            reactance=1,
            resistance=1,
            voltage=33,
            length_km=10,
        ),
        EdgeElData(
            id="edge2",
            node_from="node2",
            node_to="node1",
            flow_max=50,
            voltage=33,
            length_km=10,
            # reactance=1,
            # resistance=1,
            power_loss_function=([0, 1], [0, 0.02]),
        ),
    ],
    devices={
        "elsource": DeviceSource_elData(id="elsource", node_id="node1", flow_max=12),
        "gt1": DevicePowersourceData(
            id="gt1", node_id="node2", flow_max=30, profile="profile1"
        ),
        "demand": DeviceSink_elData(
            id="demand", node_id="node2", flow_min=4, profile="profile1"
        ),
    },
    parameters=OptimisationParametersData(
        time_delta_minutes=30,
        planning_horizon=12,
        optimisation_timesteps=6,
        forecast_timesteps=6,
        time_reserve_minutes=30,
        el_reserve_margin=-1,
        max_pressure_deviation=-1,
        reference_node="node1",
        co2_tax=30,
        objective="exportRevenue",
    ),
    profiles={"profile1": {"2021-04-12": 12, "2021-05-13": 10, "2021-06-10": 21}},
)


if __name__ == "__main__":
    print("Serializing example data and saving to file (examples/energysystem.json)")
    # serialized = serialize_oogeso_data(energy_system)
    # print(serialized)

    with open("examples/energysystem.json", "w") as outfile:
        json.dump(energy_system, fp=outfile, cls=DataclassJSONEncoder, indent=2)
