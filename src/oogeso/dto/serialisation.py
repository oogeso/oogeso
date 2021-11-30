import json
import logging
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from oogeso import dto
from oogeso.utils.util import get_class_from_dto

logger = logging.getLogger(__name__)


class DataclassJSONEncoder(json.JSONEncoder):
    """Serialize (save to file)"""

    def default(self, obj: Any):
        if is_dataclass(obj=obj):
            dct = asdict(obj=obj)
            return dct
        if isinstance(obj, pd.Series):
            dct = obj.reset_index().to_json(orient="records")
            return dct
        if isinstance(obj, pd.DataFrame):
            dct = obj.reset_index().to_json(orient="records")
            return dct
        return super().default(obj)


class DataclassJSONDecoder(json.JSONDecoder):
    """Deserialize (read from file)"""

    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def _new_carrier(self, dct: Dict[str, str]):
        model = dct["id"]
        carrier_class_str = f"Carrier{model.capitalize()}Data"
        carrier_class = get_class_from_dto(class_str=carrier_class_str)
        return carrier_class(**dct)

    def _new_node(self, dct: Dict[str, str]) -> dto.NodeData:
        return dto.NodeData(**dct)

    def _new_edge(self, dct: Dict[str, str]) -> dto.EdgeData:
        carrier = dct.pop("carrier")  # gets and deletes model from dictionary
        edge_class_str = f"Edge{carrier.capitalize()}Data"
        edge_class = get_class_from_dto(class_str=edge_class_str)
        return edge_class(**dct)

    def _new_device(self, dct: Dict[str, Union[str, object]]) -> dto.DeviceData:
        logger.debug(dct)
        model = dct.pop("model")  # gets and deletes model from dictionary
        start_stop: Optional[Dict] = dct.pop("start_stop", None)
        if start_stop is not None:
            startstop_obj = dto.StartStopData(**start_stop)
            dct["start_stop"] = startstop_obj
        dev_class_str = f"Device{model.capitalize()}Data"
        dev_class = get_class_from_dto(class_str=dev_class_str)
        return dev_class(**dct)

    @staticmethod
    def _new_profile(dct: Dict[str, Union[str, List[float]]]) -> dto.TimeSeriesData:
        name = dct["id"]
        data = dct["data"]
        data_nowcast = None
        if "data_nowcast" in dct:
            data_nowcast = dct["data_nowcast"]
        return dto.TimeSeriesData(name, data, data_nowcast)

    def object_hook(self, dct):  # pylint: disable=E0202
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
            params = dto.OptimisationParametersData(**d_params)
            for n in dct["carriers"]:
                carriers.append(self._new_carrier(n))
            for n in dct["nodes"]:
                nodes.append(self._new_node(n))
            for n in dct["edges"]:
                edges.append(self._new_edge(n))
            for n in dct["devices"]:
                devs.append(self._new_device(n))
            if "profiles" in dct:
                for n in dct["profiles"]:
                    profiles.append(self._new_profile(n))
                    # profiles[i] = self._new_profile(n)
            # if "profiles_nowcast" in dct:
            #    for i, n in dct["profiles_nowcast"].items():
            #        profiles_nowcast[i] = self._new_profile(n)
            energy_system_data = dto.EnergySystemData(
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
        if isinstance(obj, dto.EnergySystemData):
            return dto.EnergySystemData(obj=obj)
        return json.JSONEncoder.default(self, obj)


def serialize_oogeso_data(energy_system_data: dto.EnergySystemData):
    json_string = json.dumps(energy_system_data, cls=DataclassJSONEncoder, indent=2)
    return json_string


def deserialize_oogeso_data(json_data):
    energy_system_data = json.loads(json_data, cls=DataclassJSONDecoder)
    return energy_system_data


# Deserialize result object
class OogesoResultJSONDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, dct):  # pylint: disable=E0202
        res_dfs = {}
        # profiles_nowcast = {}
        if "device_flow" in dct:
            # Top level
            for k, val in dct.items():
                print(k)
                if val is None:
                    res_dfs[k] = None
                else:
                    val2 = json.loads(val)
                    df = pd.DataFrame.from_dict(val2)
                    if df.empty:
                        df = None
                    elif k in ["profiles_forecast", "profiles_nowcast"]:
                        df = df.set_index("index")
                    else:
                        # change to multi-index pandas Series
                        index_names = ["device", "time", "carrier", "terminal", "edge", "node"]
                        multiind = [c for c in df.columns if c in index_names]
                        # print(df.columns)
                        # if multiind == []:
                        #    print(val, type(val))
                        df = df.set_index(multiind)
                        # print(df.columns)
                        df = df.iloc[:, 0]
                    res_dfs[k] = df
            result_data = dto.SimulationResult(**res_dfs)
            return result_data
        return dct

    def default(self, obj: Any):
        if isinstance(obj, dto.SimulationResult):
            return dto.SimulationResult(obj=obj)
        return json.JSONEncoder.default(self, obj)


def deserialize_oogeso_results(json_data):
    result_data = json.loads(json_data, cls=OogesoResultJSONDecoder)
    return result_data


if __name__ == "__main__":
    # Todo: Move to examples and/or tests.
    print("Serializing example data and saving to file (examples/energysystem.json)")

    # Example:
    energy_system = dto.EnergySystemData(
        carriers=[
            dto.CarrierElData(
                id="el",
                reference_node="node1",
                el_reserve_margin=-1,
            ),
            dto.CarrierHeatData("heat"),
            dto.CarrierGasData(
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
        nodes=[dto.NodeData("node1"), dto.NodeData("node2")],
        edges=[
            dto.EdgeElData(
                id="edge1",
                node_from="node1",
                node_to="node2",
                flow_max=500,
                reactance=1,
                resistance=1,
                voltage=33,
                length_km=10,
            ),
            dto.EdgeElData(
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
            dto.DeviceSourceElData(id="elsource", node_id="node1", flow_max=12),
            dto.DevicePowersourceData(id="gt1", node_id="node2", flow_max=30, profile="profile1"),
            dto.DeviceSinkElData(id="demand", node_id="node2", flow_min=4, profile="profile1"),
        ],
        parameters=dto.OptimisationParametersData(
            time_delta_minutes=30,
            planning_horizon=12,
            optimisation_timesteps=6,
            forecast_timesteps=6,
            time_reserve_minutes=30,
            max_pressure_deviation=-1,
            co2_tax=30,
            objective="exportRevenue",
        ),
        profiles=[dto.TimeSeriesData(id="profile1", data=[12, 10, 21])],
    )

    # serialized = serialize_oogeso_data(energy_system)
    # print(serialized)

    with open("examples/energysystem.json", "w") as outfile:
        json.dump(energy_system, fp=outfile, cls=DataclassJSONEncoder, indent=2)
