import pyomo.environ as pyo
from . import Device


class Gasturbine(Device):
    "Gas turbine generator"

    carrier_in = ["gas"]
    carrier_out = ["el", "heat"]
    serial = []

    def _rules_misc(self, model, t, i):
        dev = self.id
        param_gas = self.carrier_data["gas"]
        # elpower = model.varDeviceFlow[dev, "el", "out", t]
        gas_energy_content = param_gas.energy_value  # MJ/Sm3
        if i == 1:
            """turbine el power out vs gas fuel in"""
            # fuel consumption (gas in) is a linear function of el power output
            # fuel = B + A*power
            # => efficiency = power/(A+B*power)
            A = self.dev_data.fuel_A
            B = self.dev_data.fuel_B
            Pmax = self.dev_data.flow_max
            lhs = model.varDeviceFlow[dev, "gas", "in", t] * gas_energy_content / Pmax
            rhs = (
                B * (model.varDeviceIsOn[dev, t] + model.varDeviceIsPrep[dev, t])
                + A * model.varDeviceFlow[dev, "el", "out", t] / Pmax
            )
            return lhs == rhs
        elif i == 2:
            """heat output = (gas energy in - el power out)* heat efficiency"""
            lhs = model.varDeviceFlow[dev, "heat", "out", t]
            rhs = (
                model.varDeviceFlow[dev, "gas", "in", t] * gas_energy_content
                - model.varDeviceFlow[dev, "el", "out", t]
            ) * self.dev_data.eta_heat
            return lhs == rhs

    def defineConstraints(self, pyomo_model):
        """Specifies the list of constraints for the device"""
        list_to_reconstruct = super().defineConstraints(pyomo_model)

        constr = pyo.Constraint(
            pyomo_model.setHorizon, pyo.RangeSet(1, 2), rule=self._rules_misc
        )
        setattr(pyomo_model, "constr_{}_{}".format(self.id, "misc"), constr)
        return list_to_reconstruct

    def getFlowVar(self, pyomo_model, t):
        return pyomo_model.varDeviceFlow[self.id, "el", "out", t]

    # overriding default
    def compute_CO2(self, pyomo_model, timesteps):
        param_gas = self.carrier_data["gas"]
        gasflow_co2 = param_gas.co2_content  # kg/m3
        thisCO2 = (
            sum(pyomo_model.varDeviceFlow[self.id, "gas", "in", t] for t in timesteps)
            * gasflow_co2
        )
        return thisCO2
