import logging
import typing

import pyomo.environ as pyo

if typing.TYPE_CHECKING:
    from oogeso.core.devices.device import Device
    from oogeso.core.networks.edge import Edge
    from oogeso.dto.oogeso_input_data_objects import NodeData

logger = logging.getLogger(__name__)


class NetworkNode:
    "Network node"

    def __init__(self, node_data: "NodeData"):
        self.node_data: "NodeData" = node_data
        self.id = node_data.id
        self.devices = {}
        self.devices_serial = {}  # devices with through-flow
        self.edges_from = {}
        self.edges_to = {}
        self.pressure_nominal = {}
        self.pressure_maxdeviation = {}

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
        self._set_pressure_parameter(carrier, term, parameter=self.pressure_maxdeviation, value=value)

    def addDevice(self, device_id, device: "Device"):
        # logger.debug("addDevice: {},{}".format(self.id, device_id))
        self.devices[device_id] = device
        for carrier in device.serial:
            if carrier not in self.devices_serial:
                self.devices_serial[carrier] = {}
            self.devices_serial[carrier][device_id] = device

    def addEdge(self, edge: "Edge", to_from: str):
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

    def _rule_terminalEnergyBalance(self, model, carrier, terminal, t):
        r""" node energy balance (at in and out terminals)
        "in" terminal: flow into terminal is positive (Pinj>0)
        "out" terminal: flow out of terminal is positive (Pinj>0)

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

        # Pinj = power injected into in terminal /out of out terminal
        Pinj = 0

        # Power in or out from connected devices:
        for dev_id, dev in self.devices.items():
            if (terminal == "in") and (carrier in dev.carrier_in):
                Pinj -= model.varDeviceFlow[dev_id, carrier, terminal, t]
            elif (terminal == "out") and (carrier in dev.carrier_out):
                Pinj -= model.varDeviceFlow[dev_id, carrier, terminal, t]

        # If no device is connected between in and out terminal for a given
        # energy carrier, connect the terminals (treat as one):
        if carrier not in self.devices_serial:
            Pinj -= model.varTerminalFlow[self.id, carrier, t]

        # edges:
        # losses are zero for all but "el" edges
        # "in": edgeFlow12 - loss12 - edgeFlow21 = edgeFlow - loss12
        # "out": edgeFlow12 - (edgeFlow21-loss21) = edgeFlow + loss21
        if (terminal == "in") and (carrier in self.edges_to):
            for edge_id, edge in self.edges_to[carrier].items():
                # power into node from edge
                if edge.has_loss():
                    Pinj += model.varEdgeFlow[edge_id, t] - model.varEdgeLoss12[edge_id, t]
                else:
                    Pinj += model.varEdgeFlow[edge_id, t]
        elif (terminal == "out") and (carrier in self.edges_from):
            for edge_id, edge in self.edges_from[carrier].items():
                # power out of node into edge
                if edge.has_loss():
                    Pinj += model.varEdgeFlow[edge_id, t] + model.varEdgeLoss21[edge_id, t]
                else:
                    Pinj += model.varEdgeFlow[edge_id, t]

        # if (carrier,node) in model.paramNodeEdgesTo and (terminal=='in'):
        #     for edg in model.paramNodeEdgesTo[(carrier,node)]:
        #         # power into node from edge
        #         Pinj += (model.varEdgeFlow[edg,t])
        # elif (carrier,node) in model.paramNodeEdgesFrom and (terminal=='out'):
        #     for edg in model.paramNodeEdgesFrom[(carrier,node)]:
        #         # power out of node into edge
        #         Pinj += (model.varEdgeFlow[edg,t])

        expr = Pinj == 0
        if (type(expr) is bool) and (expr is True):
            expr = pyo.Constraint.Skip
        return expr

    def _rule_pressureAtNode(self, model, carrier, t):
        node = self.id
        if carrier in ["el", "heat"]:
            return pyo.Constraint.Skip
        elif carrier in self.devices_serial:
            # pressure in and out are related via device equations for
            # device connected between in and out terminals. So no
            # extra constraint required
            return pyo.Constraint.Skip
        else:
            # single terminal. (pressure out=pressure in)
            expr = model.varPressure[(node, carrier, "out", t)] == model.varPressure[(node, carrier, "in", t)]
            return expr

    def _rule_pressureBounds(self, model, term, carrier, t):
        node = self.id
        nominal_pressure = self.pressure_nominal
        maxdev = None  # default is no constraint
        if carrier in nominal_pressure:
            if term in nominal_pressure[carrier]:
                nom_p = nominal_pressure[carrier][term]
                if nom_p is not None:
                    if (carrier in self.pressure_maxdeviation) and (term in self.pressure_maxdeviation[carrier]):
                        maxdev = self.pressure_maxdeviation[carrier][term]
                        if t == 0:
                            logger.debug(
                                "Using individual pressure limit for: {}, {}, {}, {}".format(
                                    node, carrier, term, maxdev
                                )
                            )
                    else:
                        # Using globally set pressure deviation limit
                        maxdev = model.paramMaxPressureDeviation
        if (maxdev is None) or (maxdev == -1):
            return pyo.Constraint.Skip
        lower_bound = nom_p * (1 - maxdev)
        upper_bound = nom_p * (1 + maxdev)
        expr = pyo.inequality(lower_bound, model.varPressure[(node, carrier, term, t)], upper_bound)
        return expr

    def defineConstraints(self, pyomo_model):
        """Returns the set of constraints for the node."""
        model = pyomo_model

        constrTerminalEnergyBalance = pyo.Constraint(
            model.setCarrier,
            ["in", "out"],
            model.setHorizon,
            rule=self._rule_terminalEnergyBalance,
        )
        setattr(
            model,
            "constrN_{}_{}".format(self.id, "energybalance"),
            constrTerminalEnergyBalance,
        )

        constrPressureAtNode = pyo.Constraint(model.setCarrier, model.setHorizon, rule=self._rule_pressureAtNode)
        setattr(
            model,
            "constrN_{}_{}".format(self.id, "nodepressure"),
            constrPressureAtNode,
        )

        constrPressureBounds = pyo.Constraint(
            model.setTerminal,
            model.setCarrier,
            model.setHorizon,
            rule=self._rule_pressureBounds,
        )
        setattr(
            model,
            "constrN_{}_{}".format(self.id, "pressurebound"),
            constrPressureBounds,
        )

    def isNontrivial(self, carrier):
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
