import logging
from typing import Union

import pyomo.environ as pyo

from oogeso.dto import NodeData

logger = logging.getLogger(__name__)


class NetworkNode:
    """Network node."""

    def __init__(self, node_data: NodeData):
        self.node_data = node_data
        self.id = node_data.id
        self.devices = {}
        self.devices_serial = {}  # devices with through-flow
        self.edges_from = {}
        self.edges_to = {}
        self.pressure_nominal = {}
        self.pressure_max_deviation = {}

    def get_pressure_nominal(self, carrier, term):
        if (carrier in self.pressure_nominal) and (term in self.pressure_nominal[carrier]):
            return self.pressure_nominal[carrier][term]
        return None

    def _set_pressure_parameter(self, carrier, term, parameter, value):
        if carrier in parameter:
            if term in parameter[carrier]:
                # pressure has already been set, so check consistency:
                p_exst = parameter[carrier][term]
                if p_exst != value:
                    raise Exception("Inconsistent pressure values for node {}:{} vs {}".format(self.id, p_exst, value))
            else:
                parameter[carrier][term] = value
        else:
            parameter[carrier] = {}
            parameter[carrier][term] = value

    def set_pressure_nominal(self, carrier, term, pressure):
        self._set_pressure_parameter(carrier, term, parameter=self.pressure_nominal, value=pressure)

    def set_pressure_maxdeviation(self, carrier, term, value):
        self._set_pressure_parameter(carrier, term, parameter=self.pressure_max_deviation, value=value)

    def add_device(self, device_id, device):
        # logger.debug("add_device: {},{}".format(self.id, device_id))
        self.devices[device_id] = device
        for carrier in device.serial:
            if carrier not in self.devices_serial:
                self.devices_serial[carrier] = {}
            self.devices_serial[carrier][device_id] = device

    def add_edge(self, edge, to_from: str):
        carrier = edge.edge_data.carrier
        edge_id = edge.id
        if to_from == "to":
            if carrier not in self.edges_to:
                self.edges_to[carrier] = {}
            self.edges_to[carrier][edge_id] = edge
        elif to_from == "from":
            if carrier not in self.edges_from:
                self.edges_from[carrier] = {}
            self.edges_from[carrier][edge_id] = edge

    def _rule_terminal_energy_balance(
        self, model: pyo.Model, carrier, terminal, t: int
    ) -> Union[pyo.Expression, pyo.Constraint.Skip]:
        r""" node energy balance (at in and out terminals)
        "in" terminal: flow into terminal is positive (P_inj>0)
        "out" terminal: flow out of terminal is positive (P_inj>0)

        distinguishing between the case (1) where node/carrier has a
        single terminal or (2) with an "in" and one "out" terminal
        (with device in series) that may have different properties
        (such as gas pressure)

        edge1 \                                     / edge3  # noqa
               \      devFlow_in    devFlow_out    /
                [term in]-->--device-->--[term out]
               /      \                       /    \
        edge2 /        \......termFlow......./      \ edge4


        """

        # P_inj = power injected into in terminal /out of out terminal
        P_inj = 0

        # Power in or out from connected devices:
        for dev_id, dev in self.devices.items():
            if (terminal == "in") and (carrier in dev.carrier_in):
                P_inj -= model.varDeviceFlow[dev_id, carrier, terminal, t]
            elif (terminal == "out") and (carrier in dev.carrier_out):
                P_inj -= model.varDeviceFlow[dev_id, carrier, terminal, t]

        # If no device is connected between in and out terminal for a given
        # energy carrier, connect the terminals (treat as one):
        if carrier not in self.devices_serial:
            P_inj -= model.varTerminalFlow[self.id, carrier, t]

        # edges:
        # losses are zero for all but "el" edges
        # "in": edgeFlow12 - loss12 - edgeFlow21 = edgeFlow - loss12
        # "out": edgeFlow12 - (edgeFlow21-loss21) = edgeFlow + loss21
        if (terminal == "in") and (carrier in self.edges_to):
            for edge_id, edge in self.edges_to[carrier].items():
                # power into node from edge
                if edge.has_loss():
                    P_inj += model.varEdgeFlow[edge_id, t] - model.varEdgeLoss12[edge_id, t]
                else:
                    P_inj += model.varEdgeFlow[edge_id, t]
        elif (terminal == "out") and (carrier in self.edges_from):
            for edge_id, edge in self.edges_from[carrier].items():
                # power out of node into edge
                if edge.has_loss():
                    P_inj += model.varEdgeFlow[edge_id, t] + model.varEdgeLoss21[edge_id, t]
                else:
                    P_inj += model.varEdgeFlow[edge_id, t]

        expr = P_inj == 0
        if (type(expr) is bool) and (expr is True):
            expr = pyo.Constraint.Skip
        return expr  # noqa

    def _rule_pressure_at_node(self, model: pyo.Model, carrier, t) -> Union[pyo.Expression, pyo.Constraint.Skip]:
        node = self.id
        if carrier in ["el", "heat"]:
            return pyo.Constraint.Skip  # noqa
        elif carrier in self.devices_serial:
            # pressure in and out are related via device equations for
            # device connected between in and out terminals. So no
            # extra constraint required
            return pyo.Constraint.Skip  # noqa
        else:
            # single terminal. (pressure out=pressure in)
            expr = model.varPressure[(node, carrier, "out", t)] == model.varPressure[(node, carrier, "in", t)]
            return expr  # noqa

    def _rule_pressure_bounds(
        self, model: pyo.Model, term, carrier, t: int
    ) -> Union[pyo.Expression, pyo.Constraint.Skip]:
        node = self.id
        nominal_pressure = self.pressure_nominal
        max_dev = None  # default is no constraint
        if carrier in nominal_pressure:
            if term in nominal_pressure[carrier]:
                nom_p = nominal_pressure[carrier][term]
                if nom_p is not None:
                    if (carrier in self.pressure_max_deviation) and (term in self.pressure_max_deviation[carrier]):
                        max_dev = self.pressure_max_deviation[carrier][term]
                        if t == 0:
                            logger.debug(f"Using individual pressure limit for: {node}, {carrier}, {term}, {max_dev}")
            else:
                nom_p = 0
        else:
            nom_p = 0
        if (max_dev is None) or (max_dev == -1):
            return pyo.Constraint.Skip  # noqa
        lower_bound = nom_p * (1 - max_dev)
        upper_bound = nom_p * (1 + max_dev)
        expr = pyo.inequality(lower_bound, model.varPressure[(node, carrier, term, t)], upper_bound)
        return expr

    def define_constraints(self, pyomo_model: pyo.Model):
        """Returns the set of constraints for the node."""
        model = pyomo_model

        constr_terminal_energy_balance = pyo.Constraint(
            model.setCarrier,
            ["in", "out"],
            model.setHorizon,
            rule=self._rule_terminal_energy_balance,
        )
        setattr(
            model,
            f"constrN_{self.id}_energybalance",
            constr_terminal_energy_balance,
        )

        constr_pressure_at_node = pyo.Constraint(model.setCarrier, model.setHorizon, rule=self._rule_pressure_at_node)
        setattr(
            model,
            f"constrN_{self.id}_nodepressure",
            constr_pressure_at_node,
        )

        constr_pressure_bounds = pyo.Constraint(
            model.setTerminal,
            model.setCarrier,
            model.setHorizon,
            rule=self._rule_pressure_bounds,
        )
        setattr(
            model,
            f"constrN_{self.id}_pressurebound",
            constr_pressure_bounds,
        )

    def is_non_trivial(self, carrier):
        """returns True if edges or devices are connected for given carrier"""
        # check if any edges are connected on this carrier
        if carrier in self.edges_from:
            return True
        if carrier in self.edges_to:
            return True
        for dev_obj in self.devices.values():
            # check if any devices are connected on this carrier
            if carrier in dev_obj.carrier_in:
                return True
            if carrier in dev_obj.carrier_out:
                return True
        # nothing connected:
        return False
