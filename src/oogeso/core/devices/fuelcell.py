import pyomo.environ as pyo
import logging

from oogeso.dto.oogeso_input_data_objects import DeviceFuelcellData
from . import Device


class Fuelcell(Device):
    "Fuel cell (hydrogen to el)"
    carrier_in = ["hydrogen"]
    carrier_out = ["el", "heat"]
    serial = []

    def _rules(self, model, t, i):
        dev = self.id
        dev_data: DeviceFuelcellData = self.dev_data
        param_hydrogen = self.optimiser.all_carriers["hydrogen"]
        energy_value = param_hydrogen.energy_value  # MJ/Sm3
        efficiency = dev_data.eta
        eta_heat = dev_data.eta_heat  # heat recovery efficiency
        if i == 1:
            """hydrogen to el"""
            lhs = model.varDeviceFlow[dev, "el", "out", t]  # MW
            rhs = (
                model.varDeviceFlow[dev, "hydrogen", "in", t]
                * energy_value
                * efficiency
            )
            return lhs == rhs
        elif i == 2:
            """heat output = waste energy * heat recovery factor"""
            lhs = model.varDeviceFlow[dev, "heat", "out", t]
            rhs = (
                model.varDeviceFlow[dev, "hydrogen", "in", t]
                * energy_value
                * (1 - efficiency)
                * eta_heat
            )
            return lhs == rhs

    def defineConstraints(self):
        """Specifies the list of constraints for the device"""
        super().defineConstraints()
        model = self.pyomo_model
        constr = pyo.Constraint(model.setHorizon, pyo.RangeSet(1, 2), rule=self._rules)
        # add constraint to model:
        setattr(self.pyomo_model, "constr_{}_{}".format(self.id, "misc"), constr)

    def getFlowVar(self, t):
        return self.pyomo_model.varDeviceFlow[self.id, "el", "out", t]
