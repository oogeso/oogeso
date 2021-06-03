from pyomo.core.base.piecewise import PWRepn
import pyomo.environ as pyo
import logging
from . import Device
import numpy as np


class Source_el(Device):
    "Generic external source for electricity (e.g. cable or wind turbine)"
    carrier_in = []
    carrier_out = ["el"]
    serial = []

    def getFlowVar(self, t):
        return self.pyomo_model.varDeviceFlow[self.id, "el", "out", t]

    # overriding default
    def compute_CO2(self, timesteps):
        # co2 content in fuel combustion
        # co2em is kgCO2/MWh_el, deltaT is seconds, deviceFlow is MW
        # need to convert co2em to kgCO2/(MW*s)
        thisCO2 = 0
        if self.dev_data.co2em is not None:
            thisCO2 = (
                sum(
                    self.pyomo_model.varDeviceFlow[self.id, "el", "out", t]
                    * self.dev_data.co2em
                    for t in timesteps
                )
                * 1
                / 3600
            )
        return thisCO2


class Powersource(Device):
    "Generic external source for electricity (e.g. cable or wind turbine)"
    carrier_in = []
    carrier_out = ["el"]
    serial = []

    def defineConstraints(self):
        """Specifies the list of constraints for the device"""

        super().defineConstraints()

        constr_penalty = self._penaltyConstraint()
        setattr(
            self.pyomo_model,
            "constrPW_{}_{}".format(self.id, "penalty"),
            constr_penalty,
        )

    # TODO: Make piecewise constraint implementation more elegant
    # The varDevicePenalty seem to be required to have the same indices as the varDeviceFlow.
    # But only (device_id, time) is relevant
    # Could define a "varDevicePower" (or better name) representing the main variable
    # used with p_max / q_max / penalty_function
    def _penaltyConstraint(self):
        # Piecewise constraints require independent variable to be bounded:
        self.pyomo_model.varDeviceFlow[self.id, "el", "out", :].setlb(0)
        self.pyomo_model.varDeviceFlow[self.id, "el", "out", :].setub(
            self.dev_data.flow_max
        )
        lookup_table = self.dev_data.penalty_function
        pw_x = lookup_table[0]
        pw_y = lookup_table[1]
        var_x = self.pyomo_model.varDeviceFlow  # [self.dev_id, "el", "out", :]
        var_y = self.pyomo_model.varDevicePenalty  # [self.dev_id, "el", "out", :]
        pw_repn = "SOS2"
        if "piecewise_repn" in self.optimiser.optimisation_parameters:
            pw_repn = self.optimiser.optimisation_parameters["piecewise_repn"]
        else:
            logging.info("Using default SOS2 piecewise constraint implementation")
        constr_penalty = pyo.Piecewise(
            [self.id],
            ["el"],
            ["out"],
            self.pyomo_model.setHorizon,
            var_y,
            var_x,
            pw_repn=pw_repn,  # default "SOS2" does not work with CBC solver
            pw_constr_type="EQ",
            pw_pts=pw_x,
            f_rule=pw_y,  # self._penaltyfunction,
        )
        return constr_penalty

    def getFlowVar(self, t):
        return self.pyomo_model.varDeviceFlow[self.id, "el", "out", t]

    # overriding default
    def compute_CO2(self, timesteps):
        # co2 content in fuel combustion
        # co2em is kgCO2/MWh_el, deltaT is seconds, deviceFlow is MW
        # need to convert co2em to kgCO2/(MW*s)
        thisCO2 = 0
        if self.dev_data.co2em is not None:
            thisCO2 = (
                sum(
                    self.pyomo_model.varDeviceFlow[self.id, "el", "out", t]
                    * self.dev_data.co2em
                    for t in timesteps
                )
                * 1
                / 3600
            )
        return thisCO2


class Source_gas(Device):
    "Generic external source for gas"
    carrier_in = []
    carrier_out = ["gas"]
    serial = []

    def _rules(self, model, t):
        node = self.dev_data.node_id
        lhs = model.varPressure[(node, "gas", "out", t)]
        rhs = self.dev_data.naturalpressure
        return lhs == rhs

    def defineConstraints(self):
        """Specifies the list of constraints for the device"""
        # add generic device constraints:
        super().defineConstraints()

        constr_well = pyo.Constraint(self.pyomo_model.setHorizon, rule=self._rules)
        # add constraint to model:
        setattr(
            self.pyomo_model,
            "constr_{}_{}".format(self.id, "pressure"),
            constr_well,
        )

    def getFlowVar(self, t):
        return self.pyomo_model.varDeviceFlow[self.id, "gas", "out", t]


class Source_oil(Device):
    "Generic external source for oil"
    carrier_in = []
    carrier_out = ["oil"]
    serial = []

    def _rules(self, model, t):
        node = self.dev_data.node_id
        lhs = model.varPressure[(node, "oil", "out", t)]
        rhs = self.dev_data.naturalpressure
        return lhs == rhs

    def defineConstraints(self):
        """Specifies the list of constraints for the device"""
        # add generic device constraints:
        super().defineConstraints()

        constr_well = pyo.Constraint(self.pyomo_model.setHorizon, rule=self._rules)
        # add constraint to model:
        setattr(
            self.pyomo_model,
            "constr_{}_{}".format(self.id, "pressure"),
            constr_well,
        )

    def getFlowVar(self, t):
        return self.pyomo_model.varDeviceFlow[self.id, "oil", "out", t]


class Source_water(Device):
    "Generic external source for water"
    carrier_in = []
    carrier_out = ["water"]
    serial = []

    def _rules(self, model, t):
        node = self.dev_data.node_id
        lhs = model.varPressure[(node, "water", "out", t)]
        rhs = self.dev_data.naturalpressure
        return lhs == rhs

    def defineConstraints(self):
        """Specifies the list of constraints for the device"""
        # add generic device constraints:
        super().defineConstraints()

        constr_well = pyo.Constraint(self.pyomo_model.setHorizon, rule=self._rules)
        # add constraint to model:
        setattr(
            self.pyomo_model,
            "constr_{}_{}".format(self.id, "pressure"),
            constr_well,
        )

    def getFlowVar(self, t):
        return self.pyomo_model.varDeviceFlow[self.id, "water", "out", t]
