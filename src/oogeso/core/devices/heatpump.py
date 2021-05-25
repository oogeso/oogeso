import pyomo.environ as pyo
import logging
from . import Device

class Heatpump(Device):
    "Heatpump or electric heater (el to heat)"
    carrier_in = ['el']
    carrier_out = ['heat']
    serial = []

    def _rules(self,model,t):
        dev = self.dev_id
        # heat out = el in * efficiency
        lhs = model.varDeviceFlow[dev,'heat','out',t]
        rhs = (model.varDeviceFlow[dev,'el','in',t]
                *model.paramDevice[dev]['eta'])
        return (lhs==rhs)

    def defineConstraints(self):
        """Specifies the list of constraints for the device"""

        super().defineConstraints()

        constr = pyo.Constraint(model.setHorizon,rule=self._rules)
        # add constraint to model:
        setattr(self.pyomo_model,'constr_{}_{}'.format(self.dev_id,'misc'),
            constr)

    def getPowerVar(self,t):
        return self.pyomo_model.varDeviceFlow[self.dev_id,'el','in',t]
