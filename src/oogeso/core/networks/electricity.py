from .network import Network
from dataclasses import asdict
from . import electricalsystem as el_calc
import logging
import pyomo.environ as pyo
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from oogeso.core.devices import Device

logger = logging.getLogger(__name__)


class El(Network):
    # def __init__(
    #    self,
    #    carrier_data: CarrierElData,
    #    edges: Dict[str, EdgeElData],
    # ):
    #    super().__init__(carrier_data, edges)

    def defineConstraints(self, pyomo_model):
        super().defineConstraints(pyomo_model)

        if self.carrier_data.powerflow_method == "dc-pf":
            logger.warning(
                "TODO: code for electric powerflow calculations need improvement (pu conversion)"
            )
            nodelist = list(pyomo_model.setNode)  # self.all_nodes.keys()
            edgelist_el = {
                edge_id: asdict(edge.edge_data) for edge_id, edge in self.edges.items()
            }
            coeff_B, coeff_DA = el_calc.computePowerFlowMatrices(
                nodelist, edgelist_el, baseZ=1
            )
            self.elFlowCoeffB = coeff_B
            self.elFlowCoeffDA = coeff_DA

            # Reference voltage node:
            constr_ElVoltageReference = pyo.Constraint(
                pyomo_model.setHorizon, rule=self._rule_elVoltageReference
            )
            setattr(
                pyomo_model,
                "constrN_{}_{}".format(self.carrier_data.reference_node, "voltageref"),
                constr_ElVoltageReference,
            )
            # Linearised power flow equations:
            for edge in self.edges.keys():
                constr_flow = pyo.Constraint(
                    edge,
                    pyomo_model.setHorizon,
                    rule=self._ruleDcPowerFlowEquations,
                )
                setattr(pyomo_model, "constrE_{}_{}".format(edge, "flow"), constr_flow)

    def _rule_elVoltageReference(self, model, t):
        el_carrier = self.carrier_data
        n = el_carrier.reference_node
        expr = model.varElVoltageAngle[n, t] == 0
        return expr

    def _ruleDcPowerFlowEquations(self, model, edge, t):
        """Flow vs voltage angle"""
        base_mva = el_calc.elbase["baseMVA"]
        base_angle = el_calc.elbase["baseAngle"]
        lhs = model.varEdgeFlow[edge, t]
        lhs = lhs / base_mva
        rhs = 0
        # TODO speed up creatioin of constraints - remove for loop
        n2s = [k[1] for k in self.elFlowCoeffDA.keys() if k[0] == edge]
        for n2 in n2s:
            rhs += self.elFlowCoeffDA[edge, n2] * (
                model.varElVoltageAngle[n2, t] * base_angle
            )
        return lhs == rhs

    def compute_elReserve(
        self,
        pyomo_model,
        t,
        all_devices: Dict[str, "Device"],
        exclude_device: str = None,
    ):
        """compute non-used generator capacity (and available loadflex)
        This is reserve to cope with forecast errors, e.g. because of wind
        variation or motor start-up

        all_devices : dict
            dictionary of all devices in the model {device_id:device_object}
        exclue_device : str (default None)
            compute reserve by devices excluding this one
        """
        alldevs = [d for d in pyomo_model.setDevice if d != exclude_device]
        # relevant devices are devices with el output or input
        cap_avail = 0
        p_generating = 0
        loadreduction = 0
        for d in alldevs:
            dev_obj = all_devices[d]
            reserve = dev_obj.compute_elReserve(pyomo_model, t)
            cap_avail += reserve["capacity_available"]
            p_generating += reserve["capacity_used"]
            loadreduction += reserve["loadreduction_available"]

        res_dev = (cap_avail - p_generating) + loadreduction
        return res_dev
