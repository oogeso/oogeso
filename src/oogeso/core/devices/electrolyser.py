import pyomo.environ as pyo
import logging

from oogeso.dto.oogeso_input_data_objects import DeviceElectrolyserData
from . import Device


class Electrolyser(Device):
    "Electrolyser"
    carrier_in = ["el"]
    carrier_out = ["hydrogen", "heat"]
    serial = []

    def _rules(self, model, t, i):
        dev = self.id
        dev_data: DeviceElectrolyserData = self.dev_data
        param_hydrogen = self.optimiser.all_carriers["hydrogen"]

        energy_value = param_hydrogen.energy_value  # MJ/Sm3
        efficiency = dev_data.eta
        if i == 1:
            lhs = model.varDeviceFlow[dev, "hydrogen", "out", t] * energy_value
            rhs = model.varDeviceFlow[dev, "el", "in", t] * efficiency
            return lhs == rhs
        elif i == 2:
            """heat output = waste energy * heat recovery factor"""
            lhs = model.varDeviceFlow[dev, "heat", "out", t]
            eta_heat = dev_data.eta_heat
            rhs = model.varDeviceFlow[dev, "el", "in", t] * (1 - efficiency) * eta_heat
            return lhs == rhs

    def defineConstraints(self):
        """Specifies the list of constraints for the device"""
        super().defineConstraints()
        model = self.pyomo_model
        constr = pyo.Constraint(model.setHorizon, pyo.RangeSet(1, 2), rule=self._rules)
        # add constraint to model:
        setattr(self.pyomo_model, "constr_{}_{}".format(self.id, "misc"), constr)

    def getFlowVar(self, t):
        return self.pyomo_model.varDeviceFlow[self.id, "el", "in", t]
