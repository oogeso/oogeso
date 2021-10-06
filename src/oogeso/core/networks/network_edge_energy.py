from __future__ import annotations
import pyomo.environ as pyo
import logging
import numpy as np
import scipy
from . import electricalsystem as el_calc
from .network_edge import NetworkEdge
import typing

if typing.TYPE_CHECKING:
    from oogeso.core.networks.network_node import NetworkNode
    from ...dto.oogeso_input_data_objects import EdgeData, EdgeFluidData


class NetworkEdgeEnergy(NetworkEdge):
    """Network edge for energy flow (electrical power or heat)
    ["electricity", "heat", "hydrogen"]
    """

    def defineConstraints(self):
        """Returns the set of constraints for the node."""

        super().defineConstraints()

        constr_flow = pyo.Constraint(
            self.pyomo_model.setHorizon, rule=self._rulePowerFlowEquations
        )
        setattr(self.pyomo_model, "constrE_{}_{}".format(self.id, "flow"), constr_flow)

    def _rulePowerFlowEquations(self, model, t):
        """Flow as a function of node values (voltage/pressure)"""
        edge = self.id
        carrier = self.edge_data.carrier

        if carrier in ["heat", "hydrogen"]:
            return pyo.Constraint.Skip
        elif carrier == "el":
            """Depending on method, power flow depends on nodal voltage angles (dc-pf)
            or is unconstrained. DC-PF refers to the linearised power flow equations"""
            carrier_data = self.optimiser.all_carriers[carrier]
            flowmethod = carrier_data.powerflow_method
            if flowmethod is None:
                return pyo.Constraint.Skip
            elif flowmethod == "transport":
                return pyo.Constraint.Skip
            elif flowmethod == "dc-pf":
                base_mva = el_calc.elbase["baseMVA"]
                base_angle = el_calc.elbase["baseAngle"]
                lhs = model.varEdgeFlow[edge, t]
                lhs = lhs / base_mva
                rhs = 0
                # TODO speed up creatioin of constraints - remove for loop
                n2s = [
                    k[1] for k in self.optimiser.elFlowCoeffDA.keys() if k[0] == edge
                ]
                for n2 in n2s:
                    rhs += self.optimiser.elFlowCoeffDA[edge, n2] * (
                        model.varElVoltageAngle[n2, t] * base_angle
                    )
                return lhs == rhs
            else:
                raise Exception(
                    "Power flow method must be None, 'transport' or 'dc-pf'"
                )
