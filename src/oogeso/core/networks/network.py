from typing import Dict, Optional, Tuple, Union

import pyomo.environ as pyo

from oogeso import dto
from oogeso.core.devices.base import Device
from oogeso.core.networks.edge import Edge, HeatEdge


class Network:
    def __init__(
        self,
        carrier_data: dto.CarrierData,
        edges: Dict[str, Edge],
    ):
        """Energy carrier."""
        self.carrier_id = carrier_data.id
        self.carrier_data = carrier_data
        self.edges = edges

    def define_constraints(self, pyomo_model: pyo.Model):
        piecewise_repn = pyomo_model.optimisation_parameters.piecewise_repn
        for edge in self.edges.values():
            edge.define_constraints(pyomo_model=pyomo_model, piecewise_repn=piecewise_repn)


class HeatNetwork(Network):
    def __init__(
        self,
        carrier_data: dto.CarrierHeatData,
        edges: Dict[str, HeatEdge],
    ):
        super().__init__(carrier_data=carrier_data, edges=edges)
        self.carrier_data = carrier_data
        self.edges = edges

    def define_constraints(self, pyomo_model: pyo.Model) -> None:
        super().define_constraints(pyomo_model=pyomo_model)

    @staticmethod
    def compute_heat_reserve(
        pyomo_model,
        t,
        all_devices: Dict[str, Device],
        exclude_device: str = None,
    ):
        """compute non-used capacity (and available loadflex)
        This is reserve to cope with forecast errors

        all_devices : dict
            dictionary of all devices in the model {device_id:device_object}
        exclude_device : str (default None)
            compute reserve by devices excluding this one
        """
        alldevs = [d for d in pyomo_model.setDevice if d != exclude_device]
        # relevant devices are devices with heat output or input
        cap_avail = 0.0
        p_generating = 0.0
        loadreduction = 0.0
        for d in alldevs:
            dev_obj = all_devices[d]
            reserve = dev_obj.compute_heat_reserve(pyomo_model, t)
            cap_avail += reserve["capacity_available"]
            p_generating += reserve["capacity_used"]
            loadreduction += reserve["loadreduction_available"]

        res_dev = (cap_avail - p_generating) + loadreduction
        return res_dev


class HydrogenNetwork(Network):
    pass


class DieselNetwork(Network):
    pass