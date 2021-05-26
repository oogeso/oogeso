import pyomo.environ as pyo
import logging


class NetworkNode:
    "Network node"
    node_id = None
    pyomo_model = None
    params = {}

    def __init__(self, pyomo_model, node_id, node_data, optimiser):
        self.node_id = node_id
        self.pyomo_model = pyomo_model
        self.params = node_data
        self.optimiser = optimiser
        self.devices = {}
        self.devices_serial = {}  # devices with through-flow
        self.edges_from = {}
        self.edges_to = {}

    def addDevice(self, device_id, device):
        # logging.debug("addDevice: {},{}".format(self.node_id, device_id))
        self.devices[device_id] = device
        for carrier in device.serial:
            if carrier not in self.devices_serial:
                self.devices_serial[carrier] = {}
            self.devices_serial[carrier][device_id] = device

    def addEdge(self, edge_id, edge, to_from):
        carrier = edge.params["type"]
        if to_from == "to":
            if carrier not in self.edges_to:
                self.edges_to[carrier] = {}
            self.edges_to[carrier][edge_id] = edge
        elif to_from == "from":
            if carrier not in self.edges_from:
                self.edges_from[carrier] = {}
            self.edges_from[carrier][edge_id] = edge

    def _rule_terminalEnergyBalance(self, model, carrier, terminal, t):
        """ node energy balance (at in and out terminals)
        "in" terminal: flow into terminal is positive (Pinj>0)
        "out" terminal: flow out of terminal is positive (Pinj>0)

        distinguishing between the case (1) where node/carrier has a
        single terminal or (2) with an "in" and one "out" terminal
        (with device in series) that may have different properties
        (such as gas pressure)

        edge1 \                                     / edge3
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
            Pinj -= model.varTerminalFlow[self.node_id, carrier, t]

        # edges:
        if (terminal == "in") and (carrier in self.edges_to):
            for edge_id, edge in self.edges_to[carrier].items():
                # power into node from edge
                Pinj += model.varEdgeFlow[edge_id, t]
        elif (terminal == "out") and (carrier in self.edges_from):
            for edge_id, edge in self.edges_from[carrier].items():
                # power out of node into edge
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
        if (type(expr) is bool) and (expr == True):
            expr = pyo.Constraint.Skip
        return expr

    def _rule_elVoltageReference(self, model, t):
        n = self.optimiser.optimisation_parameters["reference_node"]
        expr = model.varElVoltageAngle[n, t] == 0
        return expr

    def _rule_pressureAtNode(self, model, carrier, t):
        node = self.node_id
        if carrier in ["el", "heat"]:
            return pyo.Constraint.Skip
        elif carrier in self.devices_serial:
            # pressure in and out are related via device equations for
            # device connected between in and out terminals. So no
            # extra constraint required
            return pyo.Constraint.Skip
        else:
            # single terminal. (pressure out=pressure in)
            expr = (
                model.varPressure[(node, carrier, "out", t)]
                == model.varPressure[(node, carrier, "in", t)]
            )
            return expr

    def _rule_pressureBounds(self, model, term, carrier, t):
        node = self.node_id
        params_node = self.params
        params_generic = self.optimiser.optimisation_parameters
        col = "pressure.{}.{}".format(carrier, term)
        if col not in params_node:
            # no pressure data relevant for this node/carrier
            return pyo.Constraint.Skip
        nom_p = params_node[col]
        if nom_p is None:
            # no pressure data specified for this node/carrier
            return pyo.Constraint.Skip
        cc = "maxdeviation_pressure.{}.{}".format(carrier, term)
        if (cc in params_node) and (params_node[cc] is not None):
            maxdev = params_node[cc]
            if t == 0:
                logging.debug(
                    "Using ind. pressure limit for: {}, {}, {}".format(node, cc, maxdev)
                )
        else:
            maxdev = params_generic["max_pressure_deviation"]
            if maxdev == -1:
                return pyo.Constraint.Skip
        lb = nom_p * (1 - maxdev)
        ub = nom_p * (1 + maxdev)
        return (lb, model.varPressure[(node, carrier, term, t)], ub)

    def defineConstraints(self):
        """Returns the set of constraints for the node."""
        model = self.pyomo_model

        constrTerminalEnergyBalance = pyo.Constraint(
            model.setCarrier,
            ["in", "out"],
            model.setHorizon,
            rule=self._rule_terminalEnergyBalance,
        )
        setattr(
            model,
            "constrN_{}_{}".format(self.node_id, "energybalance"),
            constrTerminalEnergyBalance,
        )

        constr_ElVoltageReference = pyo.Constraint(
            model.setHorizon, rule=self._rule_elVoltageReference
        )
        setattr(
            model,
            "constrN_{}_{}".format(self.node_id, "voltageref"),
            constr_ElVoltageReference,
        )

        constrPressureAtNode = pyo.Constraint(
            model.setCarrier, model.setHorizon, rule=self._rule_pressureAtNode
        )
        setattr(
            model,
            "constrN_{}_{}".format(self.node_id, "nodepressure"),
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
            "constrN_{}_{}".format(self.node_id, "pressurebound"),
            constrPressureBounds,
        )

    def isNontrivial(self, carrier):
        """returns True if edges or devices are connected for given carrier
        """
        # check if any edges are connected on this carrier
        if carrier in self.edges_from:
            return True
        if carrier in self.edges_to:
            return True
        for d, dev_obj in self.devices.items():
            # check if any devices are connected on this carrier
            if carrier in dev_obj.carrier_in:
                return True
            if carrier in dev_obj.carrier_out:
                return True
        # nothing connected:
        return False
