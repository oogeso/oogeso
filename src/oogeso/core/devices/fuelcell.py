from typing import Dict, Union

import pyomo.environ as pyo

from oogeso import dto
from oogeso.core.devices.base import Device


class FuelCell(Device):
    """Fuel cell - hydrogen to electricity."""

    carrier_in = ["hydrogen"]
    carrier_out = ["el", "heat"]
    serial = []

    def __init__(
        self,
        dev_data: dto.DeviceFuelCellData,
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
        eta_heat = dev_data.eta_heat  # heat recovery efficiency
        if i == 1:
            """hydrogen to el"""
            lhs = pyomo_model.varDeviceFlow[dev, "el", "out", t]  # MW
            rhs = pyomo_model.varDeviceFlow[dev, "hydrogen", "in", t] * energy_value * efficiency
            return lhs == rhs
        elif i == 2:
            """heat output = waste energy * heat recovery factor"""
            lhs = pyomo_model.varDeviceFlow[dev, "heat", "out", t]
            rhs = pyomo_model.varDeviceFlow[dev, "hydrogen", "in", t] * energy_value * (1 - efficiency) * eta_heat
            return lhs == rhs
        else:
            raise ValueError(f"Argument i must be 1 or 2. {i} was given.")

    def define_constraints(self, pyomo_model: pyo.Model):
        """Specifies the list of constraints for the device"""
        list_to_reconstruct = super().define_constraints(pyomo_model)
        constr = pyo.Constraint(pyomo_model.setHorizon, pyo.RangeSet(1, 2), rule=self._rules)
        # add constraint to model:
        setattr(pyomo_model, "constr_{}_{}".format(self.id, "misc"), constr)
        return list_to_reconstruct

    def get_flow_var(self, pyomo_model: pyo.Model, t: int):
        return pyomo_model.varDeviceFlow[self.id, "el", "out", t]
