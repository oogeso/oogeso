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


def _default_pressures():
    carrier_with_pressure = ["gas", "oil", "water"]
    x = {c: {t: None for t in ["in", "out"]} for c in carrier_with_pressure}
    return x


@dataclass
class NodeData:
    id: str
    maxdeviation_pressure: Optional[Dict] = field(
        default_factory=lambda: {}
    )  # field(default_factory=_default_pressures)


@dataclass
class DeviceData:  # Parent class - use subclasses instead
    id: str
    node_id: str
    include: Optional[bool] = True
    model: str = field(init=False)
    profile: Optional[str] = None  # reference to time-series

    def __post_init__(self):
        # Model name is given by class name. Eg "gasturbine" <-> "DeviceGasturbineData"
        # This field is added to know (when serialized) which subclass the device belongs to
        modelname = re.search("Device(.+?)Data", self.__class__.__name__).group(1)
        self.model = modelname.lower()


@dataclass
class DeviceGenericData(DeviceData):
    # TODO: Replace generic "params" dictionary with the actual parameters used by the various device types
    params: Dict = field(
        default_factory=lambda: {}
    )  # key:value pairs, depending on device model


@dataclass
class DevicePowersourceData(DeviceData):
    power_to_penalty_data: Tuple[
        List[float], List[float]
    ] = None  # Penalty may be fuel, emissions, cost and combinations of these
    p_max: float = None  # MW, lowr limit for el output
    p_min: float = None  # MW, lower limit for el output


@dataclass
class DeviceSource_elData(DeviceData):
    co2em: Optional[float] = None


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
    pass


@dataclass
class DeviceSink_oilData(DeviceData):
    pass


@dataclass
class DeviceSink_waterData(DeviceData):
    avg_flow: Optional[float] = None  # required average flow
    max_accumulated_deviation: Optional[
        float
    ] = None  # buffer size (max accumulated deviation from average)


@dataclass
class EdgeData:
    id: str
    node_from: str
    node_to: str
    length_km: float
    include: bool = True  # whether to include object in problem formulation
    model: str = field(init=False)

    def __post_init__(self):
        modelname = re.search("Edge(.+?)Data", self.__class__.__name__).group(1)
        self.model = modelname.lower()


@dataclass
class EdgeElData(EdgeData):
    resistance: float = 0  # ohm per km
    reactance: float = 0  # ohm per km
    voltage: float = Union[float, Tuple]
    p_max: Optional[float] = None
    power_loss_function: Optional[Tuple[List[float], List[float]]] = None
    # field(
    #    default_factory=lambda: ([0, 1], [0, 0])
    # )
    directional: Optional[bool] = False


@dataclass
class EdgeHeatData(EdgeData):
    p_max: Optional[float] = None
    power_loss_function: Optional[Tuple[List[float], List[float]]] = None


@dataclass
class EdgeHydrogenData(EdgeData):
    q_max: Optional[float] = None


@dataclass
class EdgeFluidData(EdgeData):
    pressure_from: float = None
    pressure_to: float = None
    diameter_mm: float = None
    temperature_K: float = None
    q_max: Optional[float] = None


@dataclass
class CarrierData:
    id: str


@dataclass
class CarrierElData(CarrierData):
    powerflow_method: str = "dc_pf"  # "transport","dc_pf","ac_pf"


@dataclass
class CarrierHeatData(CarrierData):
    pass


@dataclass
class CarrierHydrogenData(CarrierData):
    pass


@dataclass
class CarrierGasData(CarrierData):
    CO2content: float  # kg/Sm3 - see SSB 2016 report -> 2.34 kg/Sm3
    G_gravity: float  # 0.6
    Pb_basepressure_MPa: float  # MPa -> 0.101 # MPa
    R_individual_gas_constant: float  # J/(kg K) -> 500 J/kgK
    Tb_basetemp_K: float  # K -> 288 K = 15 degC
    Z_compressibility: float  # 0.9
    energy_value: float  # MJ/Sm3 (calorific value) -> 40 MJ/Sm3
    k_heat_capacity_ratio: float  # 1.27
    rho_density: float  # kg/m3 -> 0.84 kg/m3
    pressure_method: Optional[
        str
    ] = "weymouth"  # method used for pipe pressure drop calculation


@dataclass
class CarrierWellstreamData(CarrierData):
    darcy_friction: float  # 0.02
    rho_density: float  # kg/m3 -> 900 kg/m3
    viscosity: float  # kg/(m s) -> 0.0026 kg/(m s)
    pressure_method: Optional[
        str
    ] = "darcy-weissbach"  # method used for pipe pressure drop calculation


@dataclass
class CarrierOilData(CarrierData):
    darcy_friction: float  # 0.02
    rho_density: float  # kg/m3 -> 900 kg/m3
    viscosity: float  # kg/(m s) -> 0.0026 kg/(m s)
    pressure_method: Optional[
        str
    ] = "darcy-weissbach"  # method used for pipe pressure drop calculation


@dataclass
class CarrierWaterData(CarrierData):
    darcy_friction: float  # 0.02
    rho_density: float  # kg/m3 -> 900 kg/m3
    viscosity: float  # kg/(m s) -> 0.0026 kg/(m s)
    pressure_method: Optional[
        str
    ] = "darcy-weissbach"  # method used for pipe pressure drop calculation


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
    reference_node: str  # id of node used as reference for electrical voltage angle
    el_backup_margin: Optional[float] = -1  # MW, -1=no limit
    emission_intensity_max: Optional[float] = -1  # kgCO2/Sm3oe, -1=no limit
    emission_rate_max: Optional[float] = -1  # kgCO2/hour, -1=no limit


@dataclass
class EnergySystemData:
    parameters: OptimisationParametersData
    carriers: List[CarrierData]
    nodes: List[NodeData]
    edges: List[EdgeData]
    devices: List[DeviceData]


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

    def object_hook(self, dct):
        nodes = []
        edges = []
        devs = []
        carriers = []
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
                devs.append(self._newDevice(n))
            energy_system_data = EnergySystemData(
                carriers=carriers,
                nodes=nodes,
                edges=edges,
                devices=devs,
                parameters=params,
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
    energy_system_data = json.loads(json_data, cls=DataclassJSONEncoder)
    return energy_system_data


# Example:
energy_system = EnergySystemData(
    carriers=[
        CarrierElData(id="el"),
        CarrierHeatData("heat"),
        CarrierGasData(
            "gas",
            CO2content=0.4,
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
            p_max=500,
            reactance=1,
            resistance=1,
            voltage=33,
            length_km=10,
        ),
        EdgeElData(
            id="edge2",
            node_from="node2",
            node_to="node1",
            p_max=50,
            voltage=33,
            length_km=10,
            # reactance=1,
            # resistance=1,
            power_loss_function=([0, 1], [0, 0.02]),
        ),
    ],
    devices=[
        DeviceGenericData(
            id="elsource", node_id="node1", params={"model": "source_el", "Pmax": 500}
        ),
        DevicePowersourceData(
            id="gt1",
            node_id="node2",
        ),
    ],
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
)

print("Serializing example data and saving to file (examples/energysystem.json)")
# serialized = serialize_oogeso_data(energy_system)
# print(serialized)

with open("examples/energysystem.json", "w") as outfile:
    json.dump(energy_system, fp=outfile, cls=DataclassJSONEncoder, indent=2)
