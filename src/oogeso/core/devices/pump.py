import pyomo.environ as pyo
import logging
from . import Device


class _PumpDevice(Device):
    """Parent class for pumps. Don't use this class directly."""

    def compute_pump_demand(
        self, model, dev, carrier, linear=False, Q=None, p1=None, p2=None, t=None
    ):
        param_dev = self.params
        node = param_dev["node"]
        param_node = self.optimiser.all_nodes[node].params
        # power demand vs flow rate and pressure difference
        # see eg. doi:10.1016/S0262-1762(07)70434-0
        # P = Q*(p_out-p_in)/eta
        # units: m3/s*MPa = MW
        #
        # Assuming incompressible fluid, so flow rate m3/s=Sm3/s
        # (this approximation may not be very good for multiphase
        # wellstream)
        # assume nominal pressure and keep only flow rate dependence
        # TODO: Better linearisation?

        eta = param_dev["eta"]

        if t is None:
            t = 0
        if Q is None:
            Q = model.varDeviceFlow[dev, carrier, "in", t]
        if p1 is None:
            p1 = model.varPressure[node, carrier, "in", t]
        if p2 is None:
            p2 = model.varPressure[node, carrier, "out", t]
        if linear:
            # linearised equations around operating point
            # p1=p10, p2=p20, Q=Q0
            p10 = param_node["pressure.{}.in".format(carrier)]
            p20 = param_node["pressure.{}.out".format(carrier)]
            delta_p = p20 - p10
            #            Q0 = model.paramDevice[dev]['Q0']
            # P = (Q*(p20-p10)+Q0*(p10-p1))/eta
            P = Q * delta_p / eta
        #        elif self._quadraticConstraints:
        #            # Quadratic constraint...
        #            delta_p = (model.varPressure[(node,carrier,'out',t)]
        #                        -model.varPressure[(node,carrier,'in',t)])
        else:
            # non-linear equation - for computing outside optimisation
            P = Q * (p2 - p1) / eta
        return P

    def _rules_pump(self, model, t, i):
        dev = self.dev_id
        devmodel = self.params["model"]
        if devmodel == "pump_oil":
            carrier = "oil"
        elif devmodel == "pump_water":
            carrier = "water"
        if i == 1:
            # flow out = flow in
            lhs = model.varDeviceFlow[dev, carrier, "out", t]
            rhs = model.varDeviceFlow[dev, carrier, "in", t]
            return lhs == rhs
        elif i == 2:
            lhs = model.varDeviceFlow[dev, "el", "in", t]
            rhs = self.compute_pump_demand(
                model, dev, linear=True, t=t, carrier=carrier
            )
            return lhs == rhs

    def defineConstraints(self):
        """Specifies the list of constraints for the device"""

        super().defineConstraints()

        constr = pyo.Constraint(
            self.pyomo_model.setHorizon, pyo.RangeSet(1, 2), rule=self._rules_pump
        )
        # add constraint to model:
        setattr(self.pyomo_model, "constr_{}_{}".format(self.dev_id, "misc"), constr)

    def getPowerVar(self, t):
        return self.pyomo_model.varDeviceFlow[self.dev_id, "el", "in", t]


class Pump_oil(_PumpDevice):
    "Oil pump"
    carrier_in = ["oil", "el"]
    carrier_out = ["oil"]
    serial = ["oil"]


class Pump_wellstream(_PumpDevice):
    "Wellstream pump"
    carrier_in = ["wellstream", "el"]
    carrier_out = ["wellstream"]
    serial = ["wellstream"]


class Pump_water(_PumpDevice):
    "Water pump"
    carrier_in = ["water", "el"]
    carrier_out = ["water"]
    serial = ["water"]
