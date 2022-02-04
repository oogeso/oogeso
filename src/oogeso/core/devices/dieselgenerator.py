from typing import Dict, List, Union

import pyomo.environ as pyo

from oogeso import dto
from oogeso.core.devices.base import Device


class DieselGenerator(Device):
    """Diesel engine generator"""

    carrier_in = ["diesel"]
    carrier_out = ["el", "heat"]
    serial = list()

    def __init__(
        self,
        dev_data: dto.DeviceDieselGeneratorData,
        carrier_data_dict: Dict[str, dto.CarrierDieselData],
    ):
        super().__init__(dev_data=dev_data, carrier_data_dict=carrier_data_dict)
        self.dev_data = dev_data
        self.id = dev_data.id
        self.carrier_data = carrier_data_dict

    def _rules_misc(self, model: pyo.Model, t: int, i: int) -> Union[pyo.Expression, pyo.Constraint.Skip]:
        dev = self.id
        param_diesel = self.carrier_data["diesel"]
        # el_power = model.varDeviceFlow[dev, "el", "out", t]
        diesel_energy_content = param_diesel.energy_value  # MJ/m3
        if i == 1:
            """generator el power out vs diesel fuel in"""
            # fuel consumption (diesel in) is a linear function of el power output
            # fuel = B + A*power
            # => efficiency = power/(A+B*power)
            A = self.dev_data.fuel_A
            B = self.dev_data.fuel_B
            P_max = self.dev_data.flow_max
            lhs = model.varDeviceFlow[dev, "diesel", "in", t] #* diesel_energy_content / P_max
            rhs = (
                B * (model.varDeviceIsOn[dev, t] + model.varDeviceIsPrep[dev, t])
                + A * model.varDeviceFlow[dev, "el", "out", t] #/ P_max
            )
            return lhs == rhs
        elif i == 2:
            """heat output = (diesel energy in - el power out)* heat efficiency"""
            lhs = model.varDeviceFlow[dev, "heat", "out", t]
            rhs = (
                model.varDeviceFlow[dev, "diesel", "in", t] * diesel_energy_content - model.varDeviceFlow[dev, "el", "out", t]
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
        param_diesel = self.carrier_data["diesel"]
        dieselflow_co2 = param_diesel.co2_content  # kg/m3

        return sum(pyomo_model.varDeviceFlow[self.id, "diesel", "in", t] for t in timesteps) * dieselflow_co2