import logging
from typing import Dict, Optional, Tuple, Union

import pyomo.environ as pyo

from oogeso import dto
from oogeso.core.devices.base import Device
from oogeso.core.networks import electricalsystem as el_calc
from oogeso.core.networks.edge import ElEdge
from oogeso.core.networks.network import Network

logger = logging.getLogger(__name__)


class ElNetwork(Network):
    def __init__(
        self,
        carrier_data: dto.CarrierElData,
        edges: Dict[str, ElEdge],
    ):
        super().__init__(carrier_data=carrier_data, edges=edges)
        self.carrier_data = carrier_data
        self.edges = edges
        self.el_flow_coeff_B: Optional[Dict[Tuple[str, str], float]] = None
        self.el_flow_coeff_DA: Optional[Dict[Tuple[str, str], float]] = None

    def define_constraints(self, pyomo_model: pyo.Model) -> None:
        super().define_constraints(pyomo_model=pyomo_model)

        if self.carrier_data.powerflow_method == "dc-pf":
            raise NotImplementedError("Power flow equation not implemented yet.")

            logger.warning("Code for electric powerflow calculations need improvement (pu conversion)")
            nodelist = list(pyomo_model.setNode)  # self.all_nodes.keys()
            edgelist_el = {edge_id: edge.edge_data.dict() for edge_id, edge in self.edges.items()}
            coeff_B, coeff_DA = el_calc.compute_power_flow_matrices(nodelist, edgelist_el, base_Z=1)
            self.el_flow_coeff_B = coeff_B
            self.el_flow_coeff_DA = coeff_DA

            # Reference voltage node:
            constr_el_voltage_reference = pyo.Constraint(pyomo_model.setHorizon, rule=self._rule_el_voltage_reference)
            setattr(
                pyomo_model,
                "constrN_{}_{}".format(self.carrier_data.reference_node, "voltageref"),
                constr_el_voltage_reference,
            )
            # Linearised power flow equations:
            for edge in self.edges.keys():
                constr_flow = pyo.Constraint(
                    edge,
                    pyomo_model.setHorizon,
                    rule=self._rule_dc_power_flow_equations,
                )
                setattr(pyomo_model, "constrE_{}_{}".format(edge, "flow"), constr_flow)

    def _rule_el_voltage_reference(self, pyomo_model: pyo.Model, t: int) -> Union[pyo.Expression, pyo.Constraint.Skip]:
        el_carrier = self.carrier_data
        n = el_carrier.reference_node
        expr = pyomo_model.varElVoltageAngle[n, t] == 0
        return expr

    def _rule_dc_power_flow_equations(
        self, pyomo_model: pyo.Model, edge: str, t: int
    ) -> Union[pyo.Expression, pyo.Constraint.Skip]:
        """Flow vs voltage angle"""
        base_mva = el_calc.elbase["baseMVA"]
        base_angle = el_calc.elbase["baseAngle"]
        lhs = pyomo_model.varEdgeFlow[edge, t]
        lhs = lhs / base_mva

        n2s = [k[1] for k in self.el_flow_coeff_DA.keys() if k[0] == edge]

        rhs = sum([self.el_flow_coeff_DA[edge, n2] * (pyomo_model.varElVoltageAngle[n2, t] * base_angle)] for n2 in n2s)
        return lhs == rhs

    @staticmethod
    def compute_el_reserve(
        pyomo_model,
        t,
        all_devices: Dict[str, Device],
        exclude_device: str = None,
    ):
        """compute non-used generator capacity (and available loadflex)
        This is reserve to cope with forecast errors, e.g. because of wind
        variation or motor start-up

        all_devices : dict
            dictionary of all devices in the model {device_id:device_object}
        exclude_device : str (default None)
            compute reserve by devices excluding this one
        """
        alldevs = [d for d in pyomo_model.setDevice if d != exclude_device]
        # relevant devices are devices with el output or input
        cap_avail = 0.0
        p_generating = 0.0
        loadreduction = 0.0
        for d in alldevs:
            dev_obj = all_devices[d]
            reserve = dev_obj.compute_el_reserve(pyomo_model, t)
            cap_avail += reserve["capacity_available"]
            p_generating += reserve["capacity_used"]
            loadreduction += reserve["loadreduction_available"]

        res_dev = (cap_avail - p_generating) + loadreduction
        return res_dev
