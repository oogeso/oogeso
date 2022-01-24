from typing import Dict, Union

import pyomo.environ as pyo

from oogeso import dto
from oogeso.core.devices.base import Device


class HeatPump(Device):
    """
    Heat pump or electric heater (el to heat)
    """

    carrier_in = ["el"]
    carrier_out = ["heat"]
    serial = []

    def __init__(
        self,
        dev_data: dto.DeviceHeatPumpData,
        carrier_data_dict: Dict[str, dto.CarrierElData],
    ):
        super().__init__(dev_data=dev_data, carrier_data_dict=carrier_data_dict)
        self.dev_data = dev_data
        self.id = dev_data.id
        self.carrier_data = carrier_data_dict

    def _rules(self, pyomo_model: pyo.Model, t: int) -> Union[pyo.Expression, pyo.Constraint.Skip]:
        dev = self.id
        # heat out = el in * efficiency
        lhs = pyomo_model.varDeviceFlow[dev, "heat", "out", t]
        rhs = pyomo_model.varDeviceFlow[dev, "el", "in", t] * self.dev_data.eta
        return lhs == rhs

    def define_constraints(self, pyomo_model: pyo.Model):
        """Specifies the list of constraints for the device"""

        list_to_reconstruct = super().define_constraints(pyomo_model)

        constr = pyo.Constraint(pyomo_model.setHorizon, rule=self._rules)
        # add constraint to model:
        setattr(pyomo_model, "constr_{}_{}".format(self.id, "misc"), constr)
        return list_to_reconstruct

    def get_flow_var(self, pyomo_model: pyo.Model, t: int):
        return pyomo_model.varDeviceFlow[self.id, "el", "in", t]
