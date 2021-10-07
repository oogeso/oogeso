from oogeso.core.networks.network_node import NetworkNode
from ...dto.oogeso_input_data_objects import (
    CarrierElData,
    EdgeElData,
    NodeData,
    DeviceData,
    OptimisationParametersData,
)
from .network import Network
from dataclasses import asdict
from . import electricalsystem as el_calc
import logging
import pyomo.environ as pyo
from typing import Dict


class El(Network):
    def __init__(
        self,
        carrier_data: CarrierElData,
        all_nodes: Dict[str, NodeData],
        all_devices: Dict[str, DeviceData],
        edges: Dict[str, EdgeElData],
        optimisation_parameters: OptimisationParametersData,
    ):

        super().__init__(
            carrier_data, all_nodes, all_devices, edges, optimisation_parameters
        )

        if self.carrier_data.powerflow_method == "dc-pf":
            logging.warning(
                "TODO: code for electric powerflow calculations need improvement (pu conversion)"
            )
            nodelist = self.all_nodes.keys()
            edgelist_el = {
                edge_id: asdict(edge.edge_data) for edge_id, edge in self.edges.items()
            }
            coeff_B, coeff_DA = el_calc.computePowerFlowMatrices(
                nodelist, edgelist_el, baseZ=1
            )
            self.elFlowCoeffB = coeff_B
            self.elFlowCoeffDA = coeff_DA

    def defineConstraints(self, pyomo_model):
        super().defineConstraints(pyomo_model)

        el_reserve_margin = self.carrier_data.el_reserve_margin
        el_backup_margin = self.carrier_data.el_backup_margin

        if (el_reserve_margin is not None) and (el_reserve_margin >= 0):
            pyomo_model.constrO_elReserveMargin = pyo.Constraint(
                pyomo_model.setHorizon, rule=self._rule_elReserveMargin
            )
        else:
            logging.info("No el_reserve_margin limit specified")
        if (el_backup_margin is not None) and (el_backup_margin >= 0):
            pyomo_model.constrO_elBackupMargin = pyo.Constraint(
                pyomo_model.setDevice,
                pyomo_model.setHorizon,
                rule=self._rule_elBackupMargin,
            )
        else:
            logging.info("No el_backup_margin limit specified")

        if self.carrier_data.powerflow_method == "dc-pf":
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

    def _rule_elReserveMargin(self, model, t):
        """Reserve margin constraint (electrical supply)
        Not used capacity by power suppliers/storage/load flexibility
        must be larger than some specified margin
        (to cope with unforeseen variations)
        """
        # exclude constraint for first timesteps since the point of the
        # dispatch margin is exactly to cope with forecast errors
        if t < self.optimisation_parameters.forecast_timesteps:
            return pyo.Constraint.Skip

        margin = self.carrier_data.el_reserve_margin
        capacity_unused = self.compute_elReserve(model, t, self.all_devices)
        expr = capacity_unused >= margin
        return expr

    def _rule_elBackupMargin(self, model, dev, t):
        """Backup capacity constraint (electrical supply)
        Not used capacity by other online power suppliers plus sheddable
        load must be larger than power output of this device
        (to take over in case of a fault)

        elBackupMargin is zero or positive (if loss of load is acceptable)
        """
        dev_obj = self.all_devices[dev]
        margin = self.carrier_data.el_backup_margin
        if "el" not in dev_obj.carrier_out:
            # this is not a power generator
            return pyo.Constraint.Skip
        res_otherdevs = self.compute_elReserve(model, t, exclude_device=dev)
        expr = res_otherdevs - model.varDeviceFlow[dev, "el", "out", t] >= -margin
        return expr

    def compute_elReserve(self, pyomo_model, t, exclude_device=None):
        """compute non-used generator capacity (and available loadflex)
        This is reserve to cope with forecast errors, e.g. because of wind
        variation or motor start-up
        (does not include load reduction yet)

        exclue_device : str (default None)
            compute reserve by devices excluding this one
        """
        alldevs = [d for d in pyomo_model.setDevice if d != exclude_device]
        # relevant devices are devices with el output or input
        cap_avail = 0
        p_generating = 0
        loadreduction = 0
        for d in alldevs:
            dev_obj = self.all_devices[d]
            reserve = dev_obj.compute_elReserve(t)
            cap_avail += reserve["capacity_available"]
            p_generating += reserve["capacity_used"]
            loadreduction += reserve["loadreduction_available"]

        res_dev = (cap_avail - p_generating) + loadreduction
        # logging.info("TODO: elReserve: Ignoring load reduction option")
        # res_dev = (cap_avail-p_generating)
        return res_dev
