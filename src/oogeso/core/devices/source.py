import pyomo.environ as pyo
import logging
from . import Device


class Source_el(Device):
    "Generic external source for electricity (e.g. cable or wind turbine)"
    carrier_in = []
    carrier_out = ['el']
    serial = []

    def defineConstraints(self):
        """Specifies the list of constraints for the device"""

        super().defineConstraints()
        # No additional specific constraints

    def getPowerVar(self,t):
        return self.pyomo_model.varDeviceFlow[self.dev_id,'el','out',t]

    # overriding default
    def compute_CO2(self,timesteps):
        # co2 content in fuel combustion
        # co2em is kgCO2/MWh_el, deltaT is seconds, deviceFlow is MW
        # need to convert co2em to kgCO2/(MW*s)
        thisCO2 = 0
        if 'co2em' in self.params:
            thisCO2 = sum(
                self.pyomo_model.varDeviceFlow[self.dev_id,'el','out',t]
                *self.params['co2em']
                for t in timesteps)*1/3600
        return thisCO2

    #getFlowVar(...)
    #computeCO2(...)
    #computeStartupCosts(...)
    #computeOperatingCosts(...)


class Source_gas(Device):
    "Generic external source for gas"
    carrier_in = []
    carrier_out = ['gas']
    serial = []

    def _rules(self,model,t):
        node = self.params['node']
        lhs = model.varPressure[(node,'gas','out',t)]
        rhs = self.params['naturalpressure']
        return (lhs==rhs)

    def defineConstraints(self):
        """Specifies the list of constraints for the device"""
        # add generic device constraints:
        super().defineConstraints()

        constr_well = pyo.Constraint(
              self.pyomo_model.setHorizon,rule=self._rules)
        # add constraint to model:
        setattr(self.pyomo_model,'constr_{}_{}'.format(self.dev_id,'pressure'),
            constr_well)



    def getFlowVar(self,t):
        return self.pyomo_model.varDeviceFlow[self.dev_id,'gas','out',t]


class Source_oil(Device):
    "Generic external source for oil"
    carrier_in = []
    carrier_out = ['oil']
    serial = []

    def _rules(self,model,t):
        node = self.params['node']
        lhs = model.varPressure[(node,'oil','out',t)]
        rhs = self.params['naturalpressure']
        return (lhs==rhs)

    def defineConstraints(self):
        """Specifies the list of constraints for the device"""
        # add generic device constraints:
        super().defineConstraints()

        constr_well = pyo.Constraint(
              self.pyomo_model.setHorizon,rule=self._rules)
        # add constraint to model:
        setattr(self.pyomo_model,'constr_{}_{}'.format(self.dev_id,'pressure'),
            constr_well)

    def getFlowVar(self,t):
        return self.pyomo_model.varDeviceFlow[self.dev_id,'oil','out',t]


class Source_water(Device):
    "Generic external source for water"
    carrier_in = []
    carrier_out = ['water']
    serial = []

    def _rules(self,model,t):
        node = self.params['node']
        lhs = model.varPressure[(node,'water','out',t)]
        rhs = self.params['naturalpressure']
        return (lhs==rhs)

    def defineConstraints(self):
        """Specifies the list of constraints for the device"""
        # add generic device constraints:
        super().defineConstraints()

        constr_well = pyo.Constraint(
              self.pyomo_model.setHorizon,rule=self._rules)
        # add constraint to model:
        setattr(self.pyomo_model,'constr_{}_{}'.format(self.dev_id,'pressure'),
            constr_well)

    def getFlowVar(self,t):
        return self.pyomo_model.varDeviceFlow[self.dev_id,'water','out',t]
