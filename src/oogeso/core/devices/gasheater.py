import pyomo.environ as pyo
import logging
from . import Device


class Gasheater(Device):
    "Gas heater"
    carrier_in = ["gas"]
    carrier_out = ["heat"]
    serial = []

    def _rules(self, model, t):
        dev = self.dev_id
        param_dev = self.params
        param_gas = self.pyomo_model.all_carriers["gas"].params
        # heat out = gas input * energy content * efficiency
        gas_energy_content = param_gas["energy_value"]  # MJ/Sm3
        lhs = model.varDeviceFlow[dev, "heat", "out", t]
        rhs = (
            model.varDeviceFlow[dev, "gas", "in", t]
            * gas_energy_content
            * param_dev["eta"]
        )
        return lhs == rhs

    def defineConstraints(self):
        """Specifies the list of constraints for the device"""

        list_to_reconstruct = super().defineConstraints()

        constr = pyo.Constraint(self.pyomo_model.setHorizon, rule=self._rules)
        # add constraint to model:
        setattr(self.pyomo_model, "constr_{}_{}".format(self.dev_id, "misc"), constr)
        return list_to_reconstruct

    def getFlowVar(self, t):
        # using heat output as dimensioning variable
        # (alternative could be to use gas input)
        return self.pyomo_model.varDeviceFlow[self.dev_id, "heat", "out", t]

    # overriding default
    def compute_CO2(self, timesteps):
        param_gas = self.optimiser.all_carriers["gas"].params
        gasflow_co2 = param_gas["CO2content"]  # kg/m3
        thisCO2 = (
            sum(self.pyomo_model.varDeviceFlow[d, "gas", "in", t] for t in timesteps)
            * gasflow_co2
        )
        return thisCO2
