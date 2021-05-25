import pyomo.environ as pyo
import logging
from . import Device

class Fuelcell(Device):
    "Fuel cell (hydrogen to el)"
    carrier_in = ['hydrogen']
    carrier_out = ['el','heat']
    serial = []

    def _rules(model,t,i):
        dev = self.dev_id
        param_dev = self.params
        param_hydrogen = self.optimiser.all_carriers['hydrogen'].params
        energy_value = param_hydrogen['energy_value'] # MJ/Sm3
        efficiency = param_dev['eta']
        eta_heat = param_dev['eta_heat'] # heat recovery efficiency
        if i==1:
            '''hydrogen to el'''
            lhs = model.varDeviceFlow[dev,'el','out',t] # MW
            rhs = (model.varDeviceFlow[dev,'hydrogen','in',t]
                    *energy_value*efficiency)
            return (lhs==rhs)
        elif i==2:
            '''heat output = waste energy * heat recovery factor'''
            lhs = model.varDeviceFlow[dev,'heat','out',t]
            rhs = (model.varDeviceFlow[dev,'hydrogen','in',t]*energy_value
                    *(1-efficiency)*eta_heat)
            return (lhs==rhs)

    def defineConstraints(self):
        """Specifies the list of constraints for the device"""
        super().defineConstraints()
        constr = pyo.Constraint(model.setHorizon,pyo.RangeSet(1,2),
            rule=self._rules)
        # add constraint to model:
        setattr(self.pyomo_model,'constr_{}_{}'.format(self.dev_id,'misc'),
            constr)

    def getPowerVar(self,t):
        return self.pyomo_model.varDeviceFlow[self.dev_id,'el','out',t]
