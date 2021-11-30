import pyomo.environ as pyo

from oogeso.core.devices.base import Device


class Heatpump(Device):
    "Heatpump or electric heater (el to heat)"
    carrier_in = ["el"]
    carrier_out = ["heat"]
    serial = []

    def _rules(self, model, t):
        dev = self.dev_id
        # heat out = el in * efficiency
        lhs = model.varDeviceFlow[dev, "heat", "out", t]
        rhs = model.varDeviceFlow[dev, "el", "in", t] * model.paramDevice[dev]["eta"]
        return lhs == rhs

    def define_constraints(self, pyomo_model):
        """Specifies the list of constraints for the device"""

        list_to_reconstruct = super().define_constraints(pyomo_model)

        constr = pyo.Constraint(pyomo_model.setHorizon, rule=self._rules)
        # add constraint to model:
        setattr(pyomo_model, "constr_{}_{}".format(self.dev_id, "misc"), constr)
        return list_to_reconstruct

    def get_flow_var(self, pyomo_model, t):
        return pyomo_model.varDeviceFlow[self.dev_id, "el", "in", t]
