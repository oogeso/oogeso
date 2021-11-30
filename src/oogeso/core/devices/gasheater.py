import pyomo.environ as pyo

from oogeso.core.devices.base import Device


class Gasheater(Device):
    "Gas heater"
    carrier_in = ["gas"]
    carrier_out = ["heat"]
    serial = []

    def _rules(self, model, t):
        dev = self.dev_id
        param_dev = self.params
        param_gas = model.all_carriers["gas"].params
        # heat out = gas input * energy content * efficiency
        gas_energy_content = param_gas["energy_value"]  # MJ/Sm3
        lhs = model.varDeviceFlow[dev, "heat", "out", t]
        rhs = model.varDeviceFlow[dev, "gas", "in", t] * gas_energy_content * param_dev["eta"]
        return lhs == rhs

    def define_constraints(self, pyomo_model):
        """Specifies the list of constraints for the device"""

        list_to_reconstruct = super().define_constraints(pyomo_model)

        constr = pyo.Constraint(pyomo_model.setHorizon, rule=self._rules)
        # add constraint to model:
        setattr(pyomo_model, "constr_{}_{}".format(self.dev_id, "misc"), constr)
        return list_to_reconstruct

    def get_flow_var(self, pyomo_model, t):
        # using heat output as dimensioning variable
        # (alternative could be to use gas input)
        return pyomo_model.varDeviceFlow[self.dev_id, "heat", "out", t]

    # overriding default
    def compute_CO2(self, model, timesteps):
        """
        Fixme: The variable d and model_pyomo was not set. Changed to model and self.dev_data. Correct?
        """
        param_gas = self.carrier_data["gas"]
        gasflow_co2 = param_gas.co2_content  # kg/m3
        thisCO2 = sum(model.varDeviceFlow[self.dev_data, "gas", "in", t] for t in timesteps) * gasflow_co2
        return thisCO2
