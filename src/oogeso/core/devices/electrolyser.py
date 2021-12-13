from typing import Dict, Union

import pyomo.environ as pyo

from oogeso import dto
from oogeso.core.devices.base import Device


class Electrolyser(Device):

    carrier_in = ["el"]
    carrier_out = ["hydrogen", "heat"]
    serial = []

    def __init__(
        self,
        dev_data: dto.DeviceElectrolyserData,
        carrier_data_dict: Dict[str, dto.CarrierHydrogenData],
    ):
        super().__init__(dev_data=dev_data, carrier_data_dict=carrier_data_dict)
        self.dev_data = dev_data
        self.id = dev_data.id
        self.carrier_data = carrier_data_dict

    def _rules(self, pyomo_model: pyo.Model, t: int, i: int) -> Union[pyo.Expression, pyo.Constraint.Skip]:
        dev = self.id
        dev_data = self.dev_data
        param_hydrogen = self.carrier_data["hydrogen"]

        energy_value = param_hydrogen.energy_value  # MJ/Sm3
        efficiency = dev_data.eta
        if i == 1:
            lhs = pyomo_model.varDeviceFlow[dev, "hydrogen", "out", t] * energy_value
            rhs = pyomo_model.varDeviceFlow[dev, "el", "in", t] * efficiency
            return lhs == rhs
        elif i == 2:
            """heat output = waste energy * heat recovery factor"""
            lhs = pyomo_model.varDeviceFlow[dev, "heat", "out", t]
            eta_heat = dev_data.eta_heat
            rhs = pyomo_model.varDeviceFlow[dev, "el", "in", t] * (1 - efficiency) * eta_heat
            return lhs == rhs
        else:
            raise ValueError(f"Argument i needs to be either 1 or 2. {i} was given")

    def define_constraints(self, pyomo_model: pyo.Model):
        """Specifies the list of constraints for the device"""
        list_to_reconstruct = super().define_constraints(pyomo_model)
        constr = pyo.Constraint(pyomo_model.setHorizon, pyo.RangeSet(1, 2), rule=self._rules)
        # add constraint to model:
        setattr(pyomo_model, "constr_{}_{}".format(self.id, "misc"), constr)
        return list_to_reconstruct

    def get_flow_var(self, pyomo_model: pyo.Model, t: int):
        return pyomo_model.varDeviceFlow[self.id, "el", "in", t]
