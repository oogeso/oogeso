from pyomo.core.base.piecewise import PWRepn
import pyomo.environ as pyo
from . import Device


class Source_el(Device):
    "Generic external source for electricity (e.g. cable or wind turbine)"
    carrier_in = []
    carrier_out = ["el"]
    serial = []

    def getFlowVar(self, pyomo_model, t):
        return pyomo_model.varDeviceFlow[self.id, "el", "out", t]

    # overriding default
    def compute_CO2(self, pyomo_model, timesteps):
        # co2 content in fuel combustion
        # co2em is kgCO2/MWh_el, deltaT is seconds, deviceFlow is MW
        # need to convert co2em to kgCO2/(MW*s)
        thisCO2 = 0
        if self.dev_data.co2em is not None:
            thisCO2 = (
                sum(
                    pyomo_model.varDeviceFlow[self.id, "el", "out", t]
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

    def defineConstraints(self, pyomo_model):
        """Specifies the list of constraints for the device"""

        list_to_reconstruct = super().defineConstraints(pyomo_model)

        if self.dev_data.penalty_function is not None:
            constr_penalty = self._penaltyConstraint(pyomo_model)
            setattr(
                pyomo_model,
                "constrPW_{}_{}".format(self.id, "penalty"),
                constr_penalty,
            )
        return list_to_reconstruct

    # TODO: Make piecewise constraint implementation more elegant
    # The varDevicePenalty seem to be required to have the same indices as the varDeviceFlow.
    # But only (device_id, time) is relevant
    # Could define a "varDevicePower" (or better name) representing the main variable
    # used with p_max / q_max / penalty_function
    def _penaltyConstraint(self, pyomo_model):
        # Piecewise constraints require independent variable to be bounded:
        # ub = self.dev_data.flow_max
        ub = self.getFlowUpperBound()
        pyomo_model.varDeviceFlow[self.id, "el", "out", :].setlb(0)
        pyomo_model.varDeviceFlow[self.id, "el", "out", :].setub(ub)
        lookup_table = self.dev_data.penalty_function
        pw_x = lookup_table[0]
        pw_y = lookup_table[1]
        var_x = pyomo_model.varDeviceFlow  # [self.dev_id, "el", "out", :]
        var_y = pyomo_model.varDevicePenalty  # [self.dev_id, "el", "out", :]
        pw_repn = pyo.value(pyomo_model.paramPiecewiseRepn)
        constr_penalty = pyo.Piecewise(
            [self.id],
            ["el"],
            ["out"],
            pyomo_model.setHorizon,
            var_y,
            var_x,
            pw_repn=pw_repn,  # default "SOS2" does not work with CBC solver
            pw_constr_type="EQ",
            pw_pts=pw_x,
            f_rule=pw_y,  # self._penaltyfunction,
        )
        return constr_penalty

    def getFlowVar(self, pyomo_model, t):
        return pyomo_model.varDeviceFlow[self.id, "el", "out", t]


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

    def defineConstraints(self, pyomo_model):
        """Specifies the list of constraints for the device"""
        # add generic device constraints:
        list_to_reconstruct = super().defineConstraints(pyomo_model)

        constr_well = pyo.Constraint(pyomo_model.setHorizon, rule=self._rules)
        # add constraint to model:
        setattr(
            pyomo_model,
            "constr_{}_{}".format(self.id, "pressure"),
            constr_well,
        )
        return list_to_reconstruct

    def getFlowVar(self, pyomo_model, t):
        return pyomo_model.varDeviceFlow[self.id, "gas", "out", t]


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

    def defineConstraints(self, pyomo_model):
        """Specifies the list of constraints for the device"""
        # add generic device constraints:
        list_to_reconstruct = super().defineConstraints(pyomo_model)

        constr_well = pyo.Constraint(pyomo_model.setHorizon, rule=self._rules)
        # add constraint to model:
        setattr(
            pyomo_model,
            "constr_{}_{}".format(self.id, "pressure"),
            constr_well,
        )
        return list_to_reconstruct

    def getFlowVar(self, pyomo_model, t):
        return pyomo_model.varDeviceFlow[self.id, "oil", "out", t]


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

    def defineConstraints(self, pyomo_model):
        """Specifies the list of constraints for the device"""
        # add generic device constraints:
        list_to_reconstruct = super().defineConstraints(pyomo_model)

        constr_well = pyo.Constraint(pyomo_model.setHorizon, rule=self._rules)
        # add constraint to model:
        setattr(
            pyomo_model,
            "constr_{}_{}".format(self.id, "pressure"),
            constr_well,
        )
        return list_to_reconstruct

    def getFlowVar(self, pyomo_model, t):
        return pyomo_model.varDeviceFlow[self.id, "water", "out", t]
