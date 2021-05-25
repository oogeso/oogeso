import pyomo.environ as pyo
import logging
from . import Device


def compute_compressor_demand(model,param_dev,param_node,param_gas,linear=False,
                              Q=None,p1=None,p2=None,t=None):
    """Compute energy demand by compressor as function of pressure and flow"""
    # power demand depends on gas pressure ratio and flow
    # See LowEmission report DSP5_2020_04 for description

    k = param_gas['k_heat_capacity_ratio']
    Z = param_gas['Z_compressibility']
    # factor 1e-6 converts R units from J/kgK to MJ/kgK:
    R = param_gas['R_individual_gas_constant']*1e-6
    rho = param_gas['rho_density']
    T1 = param_dev['temp_in'] #inlet temperature
    eta = param_dev['eta'] #isentropic efficiency
    a = (k-1)/k
    c = rho/eta*1/(k-1)*Z*R*T1
    node = param_dev['node']
    if t is None:
        t=0
    if Q is None:
        Q = model.varDeviceFlow[dev,'gas','out',t]
    if p1 is None:
        p1 = model.varPressure[node,'gas','in',t]
    if p2 is None:
        p2 = model.varPressure[node,'gas','out',t]
    if linear:
        # linearised equations around operating point
        # p1=p10, p2=p20, Q=Q0
        p10 = param_node['pressure.gas.in']
        p20 = param_node['pressure.gas.out']
        Q0 = param_dev['Q0']
        P = c*(a*(p20/p10)**a * Q0*(p2/p20-p1/p10)
                 +((p20/p10)**a-1)*Q )
    else:
        P = c*((p2/p1)**a-1)*Q
    return P


class Compressor_el(Device):
    "Abstract parent class for all device types"
    params = {}
    carrier_in = []
    carrier_out = []
    serial = []

    def _rules(self,model,t,i):
        dev = self.dev_id
        param_dev = self.params
        param_node = self.pyomo_model.all_nodes[param_dev['node']].params
        param_gas = self.pyomo_model.all_carriers['gas'].params
        if i==1:
            '''gas flow in equals gas flow out (mass flow)'''
            lhs = model.varDeviceFlow[dev,'gas','in',t]
            rhs = model.varDeviceFlow[dev,'gas','out',t]
            return (lhs==rhs)
        elif i==2:
            '''Device el demand'''
            lhs = model.varDeviceFlow[dev,'el','in',t]
            rhs = compute_compressor_demand(model,
                param_dev,param_node,param_gas,linear=True,t=t)
            return (lhs==rhs)

    def defineConstraints(self):
        """Specifies the list of constraints for the device"""

        super().defineConstraints(pyomo_model)

        constr_compressor_el = pyo.Constraint(
              model.setHorizon,pyo.RangeSet(1,2),
              rule=self._rule_compressor_el)
        # add constraint to model:
        setattr(self.pyomo_model,'constr_{}_{}'.format(self.dev_id,'compr'),
            constr_well)

    def getPowerVar(self,t):
        return self.pyomo_model.varDeviceFlow[self.dev_id,'el','in',t]


class Compressor_gas(Device):
    "Abstract parent class for all device types"
    params = {}
    carrier_in = []
    carrier_out = []
    serial = []

    def _rules(model,t):
        dev = self.dev_id
        param_dev = self.params
        param_node = self.pyomo_model.all_nodes[param_dev['node']].params
        param_gas = self.pyomo_model.all_carriers['gas'].params
        gas_energy_content=self.all_carriers['gas']['energy_value'] #MJ/Sm3
        powerdemand = compute_compressor_demand(model,
            param_dev,param_node,param_gas,linear=True,t=t)
        # matter conservation:
        lhs = model.varDeviceFlow[dev,'gas','out',t]
        rhs = (model.varDeviceFlow[dev,'gas','in',t]
                - powerdemand/gas_energy_content)
        return lhs==rhs

    def defineConstraints(self):
        """Specifies the list of constraints for the device"""

        super().defineConstraints()

        constr = pyo.Constraint(model.setHorizon,rule=self._rules)
        # add constraint to model:
        setattr(self.pyomo_model,'constr_{}_{}'.format(self.dev_id,'compr'),
            constr)

    # overriding default
    def compute_CO2(self,timesteps):
        model = self.pyomo_model
        param_gas = self.optimiser.all_carriers['gas'].params
        gasflow_co2 = param_gas['CO2content'] #kg/m3
        thisCO2 = sum((model.varDeviceFlow[d,'gas','in',t]
                         -model.varDeviceFlow[d,'gas','out',t])
                        for t in timesteps)*gasflow_co2
        return thisCO2
