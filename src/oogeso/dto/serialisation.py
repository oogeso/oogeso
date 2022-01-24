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

    @staticmethod
    def _new_carrier(dct: Dict[str, str]):
        model = dct["id"]
        carrier_class_str = f"Carrier{model.capitalize()}Data"
        carrier_class = get_class_from_dto(class_str=carrier_class_str)
        return carrier_class(**dct)

    @staticmethod
    def _new_node(dct: Dict[str, str]) -> dto.NodeData:
        return dto.NodeData(**dct)

    @staticmethod
    def _new_edge(dct: Dict[str, str]) -> dto.EdgeData:
        carrier = dct.pop("carrier")  # gets and deletes model from dictionary
        edge_class_str = f"Edge{carrier.capitalize()}Data"
        edge_class = get_class_from_dto(class_str=edge_class_str)
        return edge_class(**dct)

    @staticmethod
    def _new_device(dct: Dict[str, Union[str, object]]) -> dto.DeviceData:
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

    def object_hook(self, dct):  # noqa
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
            return dto.EnergySystemData(obj=obj)  # noqa
        return json.JSONEncoder.default(self, obj)  # noqa


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

    def object_hook(self, dct):  # noqa
        res_dfs = {}
        # profiles_nowcast = {}
        if "device_flow" in dct:
            # Top level
            for k, val in dct.items():
                logger.debug(k)
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
                        df = df.set_index(multiind)
                        df = df.iloc[:, 0]
                    res_dfs[k] = df
            result_data = dto.SimulationResult(**res_dfs)
            return result_data
        return dct

    @staticmethod
    def default(obj: Any):
        if isinstance(obj, dto.SimulationResult):
            return dto.SimulationResult(obj=obj)  # Fixme: Failing static test. No test coverage.
        return json.JSONEncoder().default(obj)


def deserialize_oogeso_results(json_data):
    result_data = json.loads(json_data, cls=OogesoResultJSONDecoder)
    return result_data
