from typing import Dict, List, Union

import pyomo.environ as pyo

from oogeso import dto
from oogeso.core.devices.base import Device


class GasHeater(Device):
    carrier_in = ["gas"]
    carrier_out = ["heat"]
    serial = []

    def __init__(
        self,
        dev_data: dto.DeviceGasHeaterData,
        carrier_data_dict: Dict[str, dto.CarrierGasData],
    ):
        super().__init__(dev_data=dev_data, carrier_data_dict=carrier_data_dict)
        self.dev_data = dev_data
        self.id = dev_data.id
        self.carrier_data = carrier_data_dict

    def _rules(self, pyomo_model: pyo.Model, t: int) -> Union[pyo.Expression, pyo.Constraint.Skip]:
        dev = self.id
        gas_data = self.carrier_data["gas"]
        # heat out = gas input * energy content * efficiency
        gas_energy_content = gas_data.energy_value  # MJ/Sm3
        lhs = pyomo_model.varDeviceFlow[dev, "heat", "out", t]
        rhs = pyomo_model.varDeviceFlow[dev, "gas", "in", t] * gas_energy_content * self.dev_data.eta
        return lhs == rhs

    def define_constraints(self, pyomo_model: pyo.Model):
        """Specifies the list of constraints for the device"""

        list_to_reconstruct = super().define_constraints(pyomo_model)

        constr = pyo.Constraint(pyomo_model.setHorizon, rule=self._rules)
        # add constraint to model:
        setattr(pyomo_model, "constr_{}_{}".format(self.id, "misc"), constr)
        return list_to_reconstruct

    def get_flow_var(self, pyomo_model: pyo.Model, t: int):
        # using heat output as dimensioning variable
        # (alternative could be to use gas input)
        return pyomo_model.varDeviceFlow[self.id, "heat", "out", t]

    def compute_CO2(self, pyomo_model: pyo.Model, timesteps: List[int]) -> float:
        param_gas = self.carrier_data["gas"]
        gas_flow_co2 = param_gas.co2_content  # kg/m3
        return sum(pyomo_model.varDeviceFlow[self.id, "gas", "in", t] for t in timesteps) * gas_flow_co2
