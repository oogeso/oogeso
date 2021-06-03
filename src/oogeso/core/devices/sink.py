import pyomo.environ as pyo
import logging
from . import Device


class Sink_el(Device):
    "Generic electricity consumption"
    carrier_in = ["el"]
    carrier_out = []
    serial = []

    def getFlowVar(self, t):
        return self.pyomo_model.varDeviceFlow[self.id, "el", "in", t]


class Sink_heat(Device):
    "Generic heat consumption"
    carrier_in = ["heat"]
    carrier_out = []
    serial = []

    def getFlowVar(self, t):
        return self.pyomo_model.varDeviceFlow[self.id, "heat", "in", t]


class Sink_gas(Device):
    "Generic electricity consumption"
    carrier_in = ["gas"]
    carrier_out = []
    serial = []

    def getFlowVar(self, t):
        return self.pyomo_model.varDeviceFlow[self.id, "gas", "in", t]


class Sink_oil(Device):
    "Generic oil consumption"
    carrier_in = ["oil"]
    carrier_out = []
    serial = []

    def getFlowVar(self, t):
        return self.pyomo_model.varDeviceFlow[self.id, "oil", "in", t]


class Sink_water(Device):
    "Generic water consumption"
    carrier_in = ["water"]
    carrier_out = []
    serial = []

    def rule_devmodel_sink_water(self, model, t, i):
        dev = self.id
        dev_data = self.dev_data
        param_generic = self.optimiser.optimisation_parameters

        if dev_data.flow_avg is None:
            return pyo.Constraint.Skip
        if dev_data.max_accumulated_deviation is None:
            return pyo.Constraint.Skip
        if dev_data.max_accumulated_deviation == 0:
            return pyo.Constraint.Skip
        if i == 1:
            # FLEXIBILITY
            # (water_in-water_avg)*dt = delta buffer
            delta_t = param_generic.time_delta_minutes / 60  # hours
            lhs = (
                model.varDeviceFlow[dev, "water", "in", t] - dev_data.flow_avg
            ) * delta_t
            if t > 0:
                Eprev = model.varDeviceStorageEnergy[dev, t - 1]
            else:
                Eprev = model.paramDeviceEnergyInitially[dev]
            rhs = model.varDeviceStorageEnergy[dev, t] - Eprev
            return lhs == rhs
        elif i == 2:
            # energy buffer limit
            Emax = dev_data.max_accumulated_deviation
            return pyo.inequality(
                -Emax / 2, model.varDeviceStorageEnergy[dev, t], Emax / 2
            )
        else:
            raise Exception("impossible")

    def defineConstraints(self):
        """Specifies the list of constraints for the device"""
        super().defineConstraints()

        constr = pyo.Constraint(
            self.pyomo_model.setHorizon,
            pyo.RangeSet(1, 2),
            rule=self.rule_devmodel_sink_water,
        )
        # add constraints to model:
        setattr(self.pyomo_model, "constr_{}_{}".format(self.id, "flex"), constr)

    def getFlowVar(self, t):
        return self.pyomo_model.varDeviceFlow[self.id, "water", "in", t]
