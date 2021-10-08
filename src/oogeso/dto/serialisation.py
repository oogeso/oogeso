import json
from dataclasses import dataclass, is_dataclass, asdict, field
from typing import List, Optional, Tuple, Any, Dict, Union
import logging
from .oogeso_input_data_objects import *
import pandas as pd

logger = logging.getLogger(__name__)

# Serialize (save to file)
class DataclassJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any):
        if is_dataclass(obj=obj):
            dct = asdict(obj=obj)
            return dct
        elif isinstance(obj, pd.Series):
            dct = json.loads(obj.reset_index().to_json())
            return dct
        elif isinstance(obj, pd.DataFrame):
            dct = obj.reset_index().to_json()
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
        carrier = dct.pop("carrier")  # gets and deletes model from dictionary
        edge_class_str = "Edge{}Data".format(carrier.capitalize())
        edge_class = globals()[edge_class_str]
        return edge_class(**dct)

    def _newDevice(self, dct):
        logger.debug(dct)
        model = dct.pop("model")  # gets and deletes model from dictionary
        startstop = dct.pop("start_stop", None)
        if startstop is not None:
            startstop_obj = StartStopData(**startstop)
            dct["start_stop"] = startstop_obj
        dev_class_str = "Device{}Data".format(model.capitalize())
        dev_class = globals()[dev_class_str]
        return dev_class(**dct)

    def _newProfile(self, dct: TimeSeries):
        name = dct["id"]
        data = dct["data"]
        data_nowcast = None
        if "data_nowcast" in dct:
            data_nowcast = dct["data_nowcast"]
        return TimeSeriesData(name, data, data_nowcast)

    def object_hook(self, dct):
        carriers = []
        nodes = []
        edges = []
        # devs = {}
        devs = []
        profiles = []
        # profiles_nowcast = {}
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
                # devs[n["id"]] = self._newDevice(n)
                devs.append(self._newDevice(n))
            if "profiles" in dct:
                for n in dct["profiles"]:
                    profiles.append(self._newProfile(n))
                    # profiles[i] = self._newProfile(n)
            # if "profiles_nowcast" in dct:
            #    for i, n in dct["profiles_nowcast"].items():
            #        profiles_nowcast[i] = self._newProfile(n)
            energy_system_data = EnergySystemData(
                carriers=carriers,
                nodes=nodes,
                edges=edges,
                devices=devs,
                parameters=params,
                profiles=profiles,
                # profiles_nowcast=profiles_nowcast,
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
        CarrierElData(
            id="el",
            reference_node="node1",
            el_reserve_margin=-1,
        ),
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
    devices=[
        DeviceSource_elData(id="elsource", node_id="node1", flow_max=12),
        DevicePowersourceData(
            id="gt1", node_id="node2", flow_max=30, profile="profile1"
        ),
        DeviceSink_elData(id="demand", node_id="node2", flow_min=4, profile="profile1"),
    ],
    parameters=OptimisationParametersData(
        time_delta_minutes=30,
        planning_horizon=12,
        optimisation_timesteps=6,
        forecast_timesteps=6,
        time_reserve_minutes=30,
        max_pressure_deviation=-1,
        co2_tax=30,
        objective="exportRevenue",
    ),
    profiles=[TimeSeriesData(id="profile1", data=[12, 10, 21])],
)


if __name__ == "__main__":
    print("Serializing example data and saving to file (examples/energysystem.json)")
    # serialized = serialize_oogeso_data(energy_system)
    # print(serialized)

    with open("examples/energysystem.json", "w") as outfile:
        json.dump(energy_system, fp=outfile, cls=DataclassJSONEncoder, indent=2)
