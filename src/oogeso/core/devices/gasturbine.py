from typing import Dict, List, Union

import pyomo.environ as pyo

from oogeso import dto
from oogeso.core.devices.base import Device


class GasTurbine(Device):
    """Gas turbine generator"""

    carrier_in = ["gas"]
    carrier_out = ["el", "heat"]
    serial = list()

    def __init__(
        self,
        dev_data: dto.DeviceGasTurbineData,
        carrier_data_dict: Dict[str, dto.CarrierGasData],
    ):
        super().__init__(dev_data=dev_data, carrier_data_dict=carrier_data_dict)
        self.dev_data = dev_data
        self.id = dev_data.id
        self.carrier_data = carrier_data_dict

    def _rules_misc(self, model: pyo.Model, t: int, i: int) -> Union[pyo.Expression, pyo.Constraint.Skip]:
        dev = self.id
        param_gas = self.carrier_data["gas"]
        # el_power = model.varDeviceFlow[dev, "el", "out", t]
        gas_energy_content = param_gas.energy_value  # MJ/Sm3
        if i == 1:
            """turbine el power out vs gas fuel in"""
            # fuel consumption (gas in) is a linear function of el power output
            # fuel = B + A*power
            # => efficiency = power/(A+B*power)
            A = self.dev_data.fuel_A
            B = self.dev_data.fuel_B
            P_max = self.dev_data.flow_max
            lhs = model.varDeviceFlow[dev, "gas", "in", t] * gas_energy_content / P_max
            rhs = (
                B * (model.varDeviceIsOn[dev, t] + model.varDeviceIsPrep[dev, t])
                + A * model.varDeviceFlow[dev, "el", "out", t] / P_max
            )
            return lhs == rhs
        elif i == 2:
            """heat output = (gas energy in - el power out)* heat efficiency"""
            lhs = model.varDeviceFlow[dev, "heat", "out", t]
            rhs = (
                model.varDeviceFlow[dev, "gas", "in", t] * gas_energy_content - model.varDeviceFlow[dev, "el", "out", t]
            ) * self.dev_data.eta_heat
            return lhs == rhs

    def define_constraints(self, pyomo_model: pyo.Model):
        """Specifies the list of constraints for the device"""
        list_to_reconstruct = super().define_constraints(pyomo_model)

        constr = pyo.Constraint(pyomo_model.setHorizon, pyo.RangeSet(1, 2), rule=self._rules_misc)
        setattr(pyomo_model, "constr_{}_{}".format(self.id, "misc"), constr)
        return list_to_reconstruct

    def get_flow_var(self, pyomo_model: pyo.Model, t: int):
        return pyomo_model.varDeviceFlow[self.id, "el", "out", t]

    # overriding default
    def compute_CO2(self, pyomo_model: pyo.Model, timesteps: List[int]) -> float:
        param_gas = self.carrier_data["gas"]
        gasflow_co2 = param_gas.co2_content  # kg/m3

        return sum(pyomo_model.varDeviceFlow[self.id, "gas", "in", t] for t in timesteps) * gasflow_co2
