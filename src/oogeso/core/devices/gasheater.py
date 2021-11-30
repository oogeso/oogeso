from typing import Dict, List, Union

import pyomo.environ as pyo

from oogeso import dto
from oogeso.core.devices.base import Device


class GasHeater(Device):
    "Gas heater"
    carrier_in = ["gas"]
    carrier_out = ["heat"]
    serial = []

    def __init__(
        self,
        dev_data: dto.DeviceGasHeaterData,  # Fixme: Correct?
        carrier_data_dict: Dict[str, dto.CarrierGasData],  # Fixme: Correct?
    ):
        super().__init__(dev_data=dev_data, carrier_data_dict=carrier_data_dict)
        self.dev_data = dev_data
        self.id = dev_data.id
        self.carrier_data = carrier_data_dict

    def _rules(self, pyomo_model: pyo.Model, t: int) -> Union[bool, pyo.Expression, pyo.Constraint.Skip]:
        dev = self.id
        param_dev = self.params  # Fixme: Attribute params is missing
        param_gas = pyomo_model.all_carriers["gas"].params
        # heat out = gas input * energy content * efficiency
        gas_energy_content = param_gas["energy_value"]  # MJ/Sm3
        lhs = pyomo_model.varDeviceFlow[dev, "heat", "out", t]
        rhs = pyomo_model.varDeviceFlow[dev, "gas", "in", t] * gas_energy_content * param_dev["eta"]
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

    # overriding default
    def compute_CO2(self, pyomo_model: pyo.Model, timesteps: List[int]) -> float:
        """
        Fixme: The variable d and model_pyomo was not set. Changed to model and self.dev_data. Correct?
        """
        param_gas = self.carrier_data["gas"]
        gasflow_co2 = param_gas.co2_content  # kg/m3
        thisCO2 = sum(pyomo_model.varDeviceFlow[self.dev_data, "gas", "in", t] for t in timesteps) * gasflow_co2
        return thisCO2
