import pyomo.environ as pyo
import pyomo.opt as pyopt
import pandas as pd
import networkx as nx
import scipy
import pydot
import matplotlib.pyplot as plt
import logging
plt.close('all')

'''
Idea:
Multi-carrier network model optimal dispatch simulation

electric network
gas network
(heat network (hot water flow))
(oil network)
(crude oil/gas network)

TODO:
    def objective function(s)
    -min operating cost, min co2, max oil/gas export...
TODO:
    physical flow equations
        el: power flow equations

TODO: Rolling optimisation (time-step by time-step)
    with variable energy availability and demand (device Pmax,Pmin)
    
TODO: Rolling horizon optimisation
    optimize for 12 hour period, using forecasts that change with time
    -start/stop of gas turbines (startup cost and ramp rate limits)
    -water injection and other flexible demand
    -> expand problem with a time dimension (pyomo set)
    


fuel transport is converted to energy transport via energy value (calorific 
value) of the fuel, i.e. energy = volume * energy_value

Device Pmax/Pmin refer to energy values

'''


class Multicarrier:
    """Multicarrier energy system"""
    
    def __init__(self,loglevel=logging.DEBUG,logfile=None):
        logging.basicConfig(filename=logfile,level=loglevel,
                            format='%(asctime)s %(levelname)s: %(message)s', 
                            datefmt='%Y-%m-%d %H:%M:%S')
        logging.debug("Initialising Multicarrier")
        # Abstract model:
        self.model = self._createPyomoModel()
        self._check_constraints_complete()
        # Concrete model instance:
        self.instance = None
        self.elbase = {
                'baseMVA':100,
                'baseAngle':1}
        #self._df_profiles = None
        
        self._dfDeviceFlow = None
        self._dfDeviceIsOn = None
        self._dfDevicePower = None
        self._dfDeviceEnergy = None
        self._dfDeviceStarting = None
        self._dfDeviceStopping = None
        self._dfEdgePower = None
        self._dfElVoltageAngle = None
        self._dfGasPressure = None
        self._dfTerminalFlow = None
        
    def _createPyomoModel(self):
        model = pyo.AbstractModel()
        
        # Sets
        model.setCarrier = pyo.Set(doc="energy carrier")
        model.setNode = pyo.Set()
        #model.setEdge1= pyo.Set(within=model.setCarrier*model.setNode*model.setNode)
        model.setEdge= pyo.Set()
        model.setDevice = pyo.Set()
        model.setTerminal = pyo.Set(initialize=['in','out'])
        model.setDevicemodel = pyo.Set()
        # time for rolling horizon optimisation:
        model.setHorizon = pyo.Set(ordered=True)
        model.setParameters = pyo.Set()
        model.setProfile = pyo.Set()
        
        # Parameters (input data)
        model.paramNode = pyo.Param(model.setNode)
        model.paramEdge = pyo.Param(model.setEdge)
        model.paramDevice = pyo.Param(model.setDevice)
        model.paramDeviceIsOnInitially = pyo.Param(model.setDevice,
                                               mutable=True,within=pyo.Binary)
        model.paramDevicePowerInitially = pyo.Param(model.setDevice,
                                               mutable=True,within=pyo.Reals)
        model.paramDeviceEnergyInitially = pyo.Param(model.setDevice,
                                               mutable=True,within=pyo.NonNegativeReals)
        model.paramNodeCarrierHasSerialDevice = pyo.Param(model.setNode)
        model.paramNodeDevices = pyo.Param(model.setNode)
        model.paramNodeEdgesFrom = pyo.Param(model.setCarrier,model.setNode)
        model.paramNodeEdgesTo = pyo.Param(model.setCarrier,model.setNode)
        model.paramDevicemodel = pyo.Param(model.setDevicemodel)
        model.paramParameters = pyo.Param(model.setParameters)
        model.paramCarriers = pyo.Param(model.setCarrier)
        model.paramCoeffB = pyo.Param(model.setNode,model.setNode,within=pyo.Reals)
        model.paramCoeffDA = pyo.Param(model.setEdge,model.setNode,within=pyo.Reals)
        # Mutable parameters (will be modified between successive optimisations)
        model.paramProfiles = pyo.Param(model.setProfile,model.setHorizon,
                                        within=pyo.Reals, mutable=True)
        #model.paramPmaxScale = pyo.Param(model.setDevice,model.setHorizon,
        #                                 mutable=True)
        
        # Variables
        #model.varNodeVoltageAngle = pyo.Var(model.setNode,within=pyo.Reals)
        model.varEdgePower = pyo.Var(
                model.setEdge,model.setHorizon,within=pyo.Reals)
        model.varDevicePower = pyo.Var(
                model.setDevice,model.setHorizon,within=pyo.NonNegativeReals)
        model.varDeviceIsOn = pyo.Var(
                model.setDevice,model.setHorizon,within=pyo.Binary)
        model.varDeviceStarting = pyo.Var(
                model.setDevice,model.setHorizon,within=pyo.Binary)
        model.varDeviceStopping = pyo.Var(
                model.setDevice,model.setHorizon,within=pyo.Binary)
        model.varDeviceEnergy = pyo.Var(model.setDevice,model.setHorizon,
                                        within=pyo.NonNegativeReals)
        model.varGasPressure = pyo.Var(
                model.setNode,model.setTerminal,model.setHorizon, 
                within=pyo.NonNegativeReals)
        model.varElVoltageAngle = pyo.Var(
                model.setNode,model.setHorizon, 
                within=pyo.Reals)
        model.varDeviceFlow = pyo.Var(
                model.setDevice,model.setCarrier,model.setTerminal,
                model.setHorizon,within=pyo.Reals)
        model.varTerminalFlow = pyo.Var(
                model.setNode,model.setCarrier,model.setHorizon,
                within=pyo.Reals)
        

        
        # Objective
        #TODO update objective function
        logging.info("TODO: objective function definition")
        def rule_objective_P(model):
            sumE = sum(model.varDevicePower[k,t]
                       for k in model.setDevice for t in model.setHorizon)
            return sumE
        
        def rule_objective_co2(model):
            '''CO2 emissions'''
            sumE = self.compute_CO2(model) #*model.paramParameters['CO2_price']
            return sumE
        model.objObjective = pyo.Objective(rule=rule_objective_co2,
                                           sense=pyo.minimize)
        
        
        
        def rule_devmodel_compressor_el(model,dev,t,i):
            if model.paramDevice[dev]['model'] != 'compressor_el':
                return pyo.Constraint.Skip
            if i==1:
                ''' power demand depends on gas pressure difference'''
                node = model.paramDevice[dev]['node']
                lhs = model.varDevicePower[dev,t]
                k = model.paramDevice[dev]['eta']
                rhs = k*(model.varGasPressure[(node,'out',t)]
                                -model.varGasPressure[(node,'in',t)])
                return (lhs==rhs)
            elif i==2:
                '''gas flow in equals gas flow out'''
                lhs = model.varDeviceFlow[dev,'gas','in',t]
                rhs = model.varDeviceFlow[dev,'gas','out',t]
                return (lhs==rhs)
            elif i==3:
                '''el in equals power demand'''
                lhs = model.varDeviceFlow[dev,'el','in',t]
                rhs = model.varDevicePower[dev,t]
                return (lhs==rhs)
        model.constrDevice_compressor_el = pyo.Constraint(model.setDevice,
                  model.setHorizon,pyo.RangeSet(1,3),
                  rule=rule_devmodel_compressor_el)
        
        def rule_devmodel_compressor_gas(model,dev,t,i):
            if model.paramDevice[dev]['model'] != 'compressor_gas':
                return pyo.Constraint.Skip
            if i==1:
                '''compressor gas demand related to pressure difference'''
                node = model.paramDevice[dev]['node']
                k = model.paramDevice[dev]['eta']
                lhs = model.varDevicePower[dev,t]
                rhs = (k*(model.varGasPressure[(node,'out',t)]
                            -model.varGasPressure[(node,'in',t)]) )
                return lhs==rhs
            elif i==2:
                '''matter conservation'''
                node = model.paramDevice[dev]['node']
                lhs = model.varDeviceFlow[dev,'gas','out',t]
                rhs = (model.varDeviceFlow[dev,'gas','in',t] 
                        - model.varDevicePower[dev,t])
                return lhs==rhs
        model.constrDevice_compressor_gas = pyo.Constraint(model.setDevice,
                  model.setHorizon,pyo.RangeSet(1,2),
                  rule=rule_devmodel_compressor_gas)
                
        
        def rule_devmodel_sink_gas(model,dev,t):
            if model.paramDevice[dev]['model'] != 'sink_gas':
                return pyo.Constraint.Skip
            lhs = model.varDeviceFlow[dev,'gas','in',t]
            rhs = model.varDevicePower[dev,t]
            return (lhs==rhs)
        model.constrDevice_sink_gas = pyo.Constraint(model.setDevice,
                  model.setHorizon,
                  rule=rule_devmodel_sink_gas)

        def rule_devmodel_export_gas(model,dev,t):
            if model.paramDevice[dev]['model'] != 'export_gas':
                return pyo.Constraint.Skip
            lhs = model.varDeviceFlow[dev,'gas','in',t]
            rhs = model.varDevicePower[dev,t]
            return (lhs==rhs)
        model.constrDevice_export_gas = pyo.Constraint(model.setDevice,
                  model.setHorizon,
                  rule=rule_devmodel_export_gas)
        
        def rule_devmodel_source_gas(model,dev,t,i):
            if model.paramDevice[dev]['model'] != 'source_gas':
                return pyo.Constraint.Skip
            if i==1:
                lhs = model.varDeviceFlow[dev,'gas','out',t]
                rhs = model.varDevicePower[dev,t]
                return (lhs==rhs)
            elif i==2:
                node = model.paramDevice[dev]['node']
                lhs = model.varGasPressure[(node,'out',t)]
                rhs = model.paramDevice[dev]['naturalpressure']
                #return pyo.Constraint.Skip
                return (lhs==rhs)
        model.constrDevice_source_gas = pyo.Constraint(model.setDevice,
                  model.setHorizon,pyo.RangeSet(1,2),
                  rule=rule_devmodel_source_gas)
        
        def rule_devmodel_gasheater(model,dev,t,i):
            if model.paramDevice[dev]['model'] != 'gasheater':
                return pyo.Constraint.Skip
            if i==1:
                # gas in = device power * efficiency
                lhs = model.varDeviceFlow[dev,'gas','in',t]*model.paramDevice[dev]['eta']
                rhs = model.varDevicePower[dev,t]
                return (lhs==rhs)
            elif i==2:
                # heat out = device power
                lhs = model.varDeviceFlow[dev,'heat','out',t]
                rhs = model.varDevicePower[dev,t]
                return (lhs==rhs)
        model.constrDevice_gasheater = pyo.Constraint(model.setDevice,
                  model.setHorizon,pyo.RangeSet(1,2),
                  rule=rule_devmodel_gasheater)
        
        def rule_devmodel_heatpump(model,dev,t,i):
            if model.paramDevice[dev]['model'] != 'heatpump':
                return pyo.Constraint.Skip
            if i==1:
                # device power = el in * efficiency
                lhs = model.varDevicePower[dev,t]
                rhs = model.varDeviceFlow[dev,'el','in',t]*model.paramDevice[dev]['eta']
                return (lhs==rhs)
            elif i==2:
                # heat out = device power
                lhs = model.varDeviceFlow[dev,'heat','out',t]
                rhs = model.varDevicePower[dev,t]
                return (lhs==rhs)
        model.constrDevice_heatpump = pyo.Constraint(model.setDevice,
                  model.setHorizon,pyo.RangeSet(1,2),
                  rule=rule_devmodel_heatpump)
        
        
        logging.info("TODO: gas turbine power vs heat output")
        logging.info("TODO: startup cost")
        def rule_devmodel_gasturbine(model,dev,t,i):
            if model.paramDevice[dev]['model'] != 'gasturbine':
                return pyo.Constraint.Skip
            if i==1:
                '''turbine power = fuel usage (should be minimised)'''
                lhs = model.varDevicePower[dev,t]
                rhs = model.varDeviceFlow[dev,'el','out',t]
                return lhs==rhs
            if i==2:
                '''turbine el power out vs gas fuel in'''       
                # fuel consumption (gas in) is a linear function of el power output
                # fuel = A + B*power
                # => efficiency = power/(A+B*power)
                A = model.paramDevice[dev]['fuelA']
                B = model.paramDevice[dev]['fuelB']
                Pmax = model.paramDevice[dev]['Pmax']
                lhs = model.varDeviceFlow[dev,'gas','in',t]/Pmax
                rhs = (A*model.varDeviceIsOn[dev,t] 
                        + B*model.varDeviceFlow[dev,'el','out',t]/Pmax)
                return lhs==rhs
            elif i==3:
                '''power only if turbine is on'''
                lhs = model.varDevicePower[dev,t]
                rhs = model.varDeviceIsOn[dev,t]*model.paramDevice[dev]['Pmax']
                return lhs <= rhs
            elif i==4:
                '''turbine power = heat output * heat fraction'''
                lhs = model.varDeviceFlow[dev,'heat','out',t]
                rhs = model.varDevicePower[dev,t]*model.paramDevice[dev]['heat']
                return lhs==rhs
        model.constrDevice_gasturbine = pyo.Constraint(model.setDevice,
                  model.setHorizon,pyo.RangeSet(1,4),
                  rule=rule_devmodel_gasturbine)

        
        print("TODO: electrical storage objective")
        #TODO: electrical storage equations
        def rule_devmodel_storage_el(model,dev,t,i):
            if model.paramDevice[dev]['model'] != 'storage_el':
                return pyo.Constraint.Skip
            if i==1:
                #energy balance
                # (el out)*dt = -delta storage
                # eta = efficiency charging  (discharging assumed lossless)
                delta_t = model.paramParameters['time_delta_minutes']/60 #hours
                lhs = model.varDeviceFlow[dev,'el','out',t]*delta_t                        
                if t>0:
                    Eprev = model.varDeviceEnergy[dev,t-1]
                else:
                    Eprev = model.paramDeviceEnergyInitially[dev]
                rhs = -(model.varDeviceEnergy[dev,t]-Eprev)
                return (lhs==rhs)
            elif i==2:
                # energy storage limit
                ub = model.paramDevice[dev]['Emax']
                return pyo.inequality(0,model.varDeviceEnergy[dev,t],ub)
            elif i==3:
                #charging/discharging power limit
                ub = model.paramDevice[dev]['Pmax']                
                return pyo.inequality(
                        -ub,model.varDeviceFlow[dev,'el','out',t],ub)
            elif i==4:
                # device power = el out
                lhs = model.varDevicePower[dev,t]
                #rhs = model.varDeviceFlow[dev,'el','out',t]
                rhs = 0
                return (lhs==rhs)
        model.constrDevice_storage_el = pyo.Constraint(model.setDevice,
                  model.setHorizon,pyo.RangeSet(1,3),
                  rule=rule_devmodel_storage_el)


        def rule_startup_shutdown(model,dev,t):
            '''startup/shutdown constraint - for devices with startup costs'''
            # setHorizon is a rangeset [0,1,2,...,max]
            if (t>0):
                ison_prev = model.varDeviceIsOn[dev,t-1]
            else:
                ison_prev = model.paramDeviceIsOnInitially[dev]         
            rhs = (model.varDeviceIsOn[dev,t] - ison_prev)
            lhs = (model.varDeviceStarting[dev,t]
                    -model.varDeviceStopping[dev,t])
            return (lhs==rhs)
        model.constrDevice_startup_shutdown = pyo.Constraint(model.setDevice,
                  model.setHorizon,
                  rule=rule_startup_shutdown)
        
        def rule_ramprate(model,dev,t):
            '''ramp rate limit'''

            # If no ramp limits have been specified, skip constraint
            if pd.isna(model.paramDevice[dev]['maxRampUp']):
                return pyo.Constraint.Skip
            if (t>0):
                p_prev = model.varDevicePower[dev,t-1]
            else:
                p_prev = model.paramDevicePowerInitially[dev]         
            deltaP = (model.varDevicePower[dev,t]- p_prev)
            delta_t = model.paramParameters['time_delta_minutes']
            maxP = model.paramDevice[dev]['Pmax']
            max_neg = -model.paramDevice[dev]['maxRampDown']*maxP*delta_t
            max_pos = model.paramDevice[dev]['maxRampUp']*maxP*delta_t
            return pyo.inequality(max_neg, deltaP, max_pos)
        model.constrDevice_ramprate = pyo.Constraint(model.setDevice,
                 model.setHorizon, rule=rule_ramprate)
            
            
        logging.info("TODO: el source: dieselgen, fuel, on-off variables")
        #TODO: diesel gen fuel, onoff variables..
        def rule_devmodel_source_el(model,dev,t):
            if model.paramDevice[dev]['model'] != 'source_el':
                return pyo.Constraint.Skip
            '''turbine power = power infeed'''
            lhs = model.varDeviceFlow[dev,'el','out',t]
            rhs = model.varDevicePower[dev,t]
            return lhs==rhs
        model.constrDevice_source_el = pyo.Constraint(model.setDevice,
                  model.setHorizon,rule=rule_devmodel_source_el)
        
        def rule_devmodel_sink_el(model,dev,t):
            if model.paramDevice[dev]['model'] != 'sink_el':
                return pyo.Constraint.Skip
            '''sink power = power out'''
            lhs = model.varDeviceFlow[dev,'el','in',t]
            rhs = model.varDevicePower[dev,t]
            return lhs==rhs
        model.constrDevice_sink_el = pyo.Constraint(model.setDevice,
                  model.setHorizon,rule=rule_devmodel_sink_el)

        def rule_devmodel_export_el(model,dev,t):
            if model.paramDevice[dev]['model'] != 'export_el':
                return pyo.Constraint.Skip
            '''sink power = power out'''
            lhs = model.varDeviceFlow[dev,'el','in',t]
            rhs = model.varDevicePower[dev,t]
            return lhs==rhs
        model.constrDevice_export_el = pyo.Constraint(model.setDevice,
                  model.setHorizon,rule=rule_devmodel_export_el)
        
        def rule_devmodel_sink_heat(model,dev,t):
            if model.paramDevice[dev]['model'] != 'sink_heat':
                return pyo.Constraint.Skip
            '''sink heat = heat out'''
            lhs = model.varDeviceFlow[dev,'heat','in',t]
            rhs = model.varDevicePower[dev,t]
            return lhs==rhs
        model.constrDevice_sink_heat = pyo.Constraint(model.setDevice,
                  model.setHorizon,rule=rule_devmodel_sink_heat)
        
        def rule_terminalEnergyBalance(model,carrier,node,terminal,t):
            ''' energy balance at in and out terminals
            "in" terminal: flow into terminal is positive
            "out" terminal: flow out of terminal is positive
            '''
            
            #if carrier=='el':
            #    return pyo.Constraint.Skip
            
            # Pinj = power injected into terminal
            Pinj = 0
            # devices:
            if (node in model.paramNodeDevices):
                for dev in model.paramNodeDevices[node]:
                    # Power into terminal:
                    dev_model = model.paramDevice[dev]['model']
                    if pd.isna(dev_model):
                        logging.info("No device model specified - using dispatch factor")  
                    elif dev_model in model.paramDevicemodel:
                        #print("carrier:{},node:{},terminal:{},model:{}"
                        #      .format(carrier,node,terminal,dev_model))
                        if carrier in model.paramDevicemodel[dev_model][terminal]:
                            Pinj -= model.varDeviceFlow[dev,carrier,terminal,t]
                    else:
                        raise Exception("Undefined device model ({})".format(dev_model))
        
            # connect terminals:
            if not model.paramNodeCarrierHasSerialDevice[node][carrier]:
                Pinj -= model.varTerminalFlow[node,carrier,t]
        
                    
            # edges:
            if (carrier,node) in model.paramNodeEdgesTo and (terminal=='in'):
                for edg in model.paramNodeEdgesTo[(carrier,node)]:
                    # power into node from edge
                    Pinj += (model.varEdgePower[edg,t])
            elif (carrier,node) in model.paramNodeEdgesFrom and (terminal=='out'):
                for edg in model.paramNodeEdgesFrom[(carrier,node)]:
                    # power out of node into edge
                    Pinj += (model.varEdgePower[edg,t])
            
            
            expr = (Pinj==0)
            if ((type(expr) is bool) and (expr==True)):
                expr = pyo.Constraint.Skip
            return expr
        model.constrTerminalEnergyBalance = pyo.Constraint(model.setCarrier,
                      model.setNode, model.setTerminal,model.setHorizon,
                      rule=rule_terminalEnergyBalance)
        
        logging.info("TODO: el power balance constraint redundant?")
        def rule_terminalElPowerBalance(model,node,t):
            ''' electric power balance at in and out terminals
            '''
            # Pinj = power injected into terminal
            Pinj = 0
            rhs = 0
            carrier='el'
            # devices:
            if (node in model.paramNodeDevices):
                for dev in model.paramNodeDevices[node]:
                    # Power into terminal:
                    dev_model = model.paramDevice[dev]['model']
                    if dev_model in model.paramDevicemodel:
                        if carrier in model.paramDevicemodel[dev_model]['in']:
                            Pinj -= model.varDeviceFlow[dev,carrier,'in',t]
                        if carrier in model.paramDevicemodel[dev_model]['out']:
                            Pinj += model.varDeviceFlow[dev,carrier,'out',t]
                    else:
                        raise Exception("Undefined device model ({})".format(dev_model))
                   
            # edges:
            # El linearised power flow equations:
            # Pinj = B theta
            n2s = [k[1]  for k in model.paramCoeffB.keys() if k[0]==node]
            for n2 in n2s:
                rhs -= model.paramCoeffB[node,n2]*(
                        model.varElVoltageAngle[n2,t]*self.elbase['baseAngle'])
            rhs = rhs*self.elbase['baseMVA']
            
            expr = (Pinj==rhs)
            if ((type(expr) is bool) and (expr==True)):
                expr = pyo.Constraint.Skip
            return expr
        model.constrElPowerBalance = pyo.Constraint(model.setNode,
                    model.setHorizon, rule=rule_terminalElPowerBalance)
        
        
        #Electrical flow equations
        #logging.info("TODO: Electrical network power flow equations")
        def rule_elVoltageAndFlow(model,edge,t):
            '''power flow equations - power flow vs voltage angle difference
            
            Linearised power flow equations (DC power flow)'''
            if model.paramEdge[edge]['type'] !='el':
                return pyo.Constraint.Skip
            
            lhs = model.varEdgePower[edge,t]
            lhs = lhs/self.elbase['baseMVA']
            rhs = 0
            #TODO speed up creatioin of constraints - remove for loop
            n2s = [k[1]  for k in model.paramCoeffDA.keys() if k[0]==edge]
            for n2 in n2s:
                rhs += model.paramCoeffDA[edge,n2]*(
                        model.varElVoltageAngle[n2,t]*self.elbase['baseAngle'])
            expr = (lhs==rhs)
            return expr
        model.constrFlowAngle = pyo.Constraint(model.setEdge, model.setHorizon,
                                               rule=rule_elVoltageAndFlow)

        
        def rule_elVoltageReference(model,t):
            n = model.paramParameters['reference_node']
            expr = (model.varElVoltageAngle[n,t] == 0)
            return expr
        model.constrElVoltageReference = pyo.Constraint(model.setHorizon,
                                              rule=rule_elVoltageReference)
        
        
        
        def rule_gasPressureAndFlow(model,edge,t):
            '''Flow as a function of pressure difference and pipe properties
            
            Q = k*(Pin-Pout)
            
            Here, a linearised Weynmouth equation is implemented
            Q: m3/s, P: J/s=W
            P = Q*energy_value
            
            REFERENCES:     
            1) E Sashi Menon, Gas Pipeline Hydraulics, Taylor & Francis (2005), 
            https://doi.org/10.1201/9781420038224     
            2) A Tomasgard et al., Optimization  models  for  the  natural  gas  
            value  chain, in: Geometric Modelling, Numerical Simulation and 
            Optimization. Springer Verlag, New York (2007), 
            https://doi.org/10.1007/978-3-540-68783-2_16
            '''
            if model.paramEdge[edge]['type'] != 'gas':
                return pyo.Constraint.Skip
            n_from = model.paramEdge[edge]['nodeFrom']
            n_to = model.paramEdge[edge]['nodeTo']
            p_from = model.varGasPressure[(n_from,'out',t)]
            p_to = model.varGasPressure[(n_to,'in',t)]
            p0_from = model.paramNode[n_from]['gaspressure_out']
            p0_to = model.paramNode[n_to]['gaspressure_in']
            k = model.paramEdge[edge]['gasflow_k']
            exp_s = model.paramEdge[edge]['exp_s']
            coeff = k*(p0_from**2-exp_s*p0_to**2)**(-1/2)
            lhs = model.varEdgePower[edge,t]/model.paramCarriers['gas']['energy_value']
            rhs = coeff*(p0_from*p_from - exp_s*p0_to*p_to)
            logging.debug("constr gas pressure vs flow: {}-{},{},{},{}".format(
                              n_from,n_to,p0_from,p0_to,coeff))
            return (lhs==rhs)
        model.constrGasPressureAndFlow = pyo.Constraint(
                model.setEdge,model.setHorizon,rule=rule_gasPressureAndFlow)
        
        def rule_gasPressureAtNode(model,node,t):
            #if not model.paramNodeNontrivial[node]['gas']:
            if not model.paramNodeCarrierHasSerialDevice[node]['gas']:
                # trivial connection. pressure out=pressure in
                expr = (model.varGasPressure[(node,'out',t)]
                        == model.varGasPressure[(node,'in',t)] )
                return expr
            else:
                return pyo.Constraint.Skip    
        model.constrGasPressureAtNode = pyo.Constraint(
                model.setNode,model.setHorizon,rule=rule_gasPressureAtNode)
        
        # TODO: Set bounds from input data
        logging.info("TODO: gas pressure bounds set by input parameters")
        def rule_gasPressureBounds(model,node,term,t):
            col = 'gaspressure_{}'.format(term)
            nom_p = model.paramNode[node][col]
            lb = nom_p*0.9
            ub = nom_p*1.1
            return (lb,model.varGasPressure[(node,term,t)],ub)
        model.constrGasPressureBounds = pyo.Constraint(
                model.setNode,model.setTerminal,model.setHorizon,
                rule=rule_gasPressureBounds)
            
        
        def rule_devicePmax(model,dev,t):
            minValue = model.paramDevice[dev]['Pmin']
            maxValue = model.paramDevice[dev]['Pmax']
            extprofile = model.paramDevice[dev]['external']
            if (not pd.isna(extprofile)):
                #maxValue = maxValue*self._df_profiles.loc[t,extprofile]
                maxValue = maxValue*model.paramProfiles[extprofile,t]
                #print(dev,t,extprofile,maxValue)
            expr = pyo.inequality(minValue,model.varDevicePower[dev,t], maxValue)
            return expr                    
        model.constrDevicePmax = pyo.Constraint(
                model.setDevice,model.setHorizon,rule=rule_devicePmax)
        
        def rule_devicePmin(model,dev,t):
            return (model.varDevicePower[dev,t] >= model.paramDevice[dev]['Pmin'])
        model.constrDevicePmin = pyo.Constraint(
                model.setDevice,model.setHorizon,rule=rule_devicePmin)
        
        def rule_edgePmaxmin(model,edge,t):
            return pyo.inequality(-model.paramEdge[edge]['capacity'], 
                    model.varEdgePower[edge,t], 
                    model.paramEdge[edge]['capacity'])
        model.constrEdgeBounds = pyo.Constraint(
                model.setEdge,model.setHorizon,rule=rule_edgePmaxmin)
    
        return model
        # END class init.

    def _check_constraints_complete(self):
        """CHECK that constraints are defined for all device models"""
        logging.debug("Checking existance of constraints for all device types")
        devmodels = Multicarrier.devicemodel_inout()
        for i in devmodels.keys():
            isdefined = hasattr(self.model,"constrDevice_{}".format(i))
            if not isdefined:
                raise Exception("Device model constraints for '{}' have"
                                " not been defined".format(i))
            logging.debug("....{} -> OK".format(i))



    
    # Helper functions
    @staticmethod
    def compute_CO2(model,devices=None,timesteps=None):
        '''compute CO2 emission (kgCO2 per hour)
        
        model can be abstract model or model instance
        '''
        if devices is None:
            devices = model.setDevice
        if timesteps is None:
            timesteps = model.setHorizon
        deltaT = model.paramParameters['time_delta_minutes']/60
        sumHours = len(timesteps)*deltaT
        
        sumCO2 = 0
        # GAS: co2 emission from consumed gas (e.g. in gas heater)
        # EL: co2 emission from the generation of electricity
        # HEAT: co2 emission from the generation of heat
        for d in devices:
            devmodel = pyo.value(model.paramDevice[d]['model'])
            if devmodel in ['gasturbine','gasheater']:
                thisCO2 = sum(model.varDeviceFlow[d,'gas','in',t]
                            *model.paramCarriers['gas']['CO2content']
                            for t in timesteps)
            elif devmodel=='compressor_gas':
                thisCO2 = sum((model.varDeviceFlow[d,'gas','in',t]
                            -model.varDeviceFlow[d,'gas','out',t])
                            *model.paramCarriers['gas']['CO2content']
                            for t in timesteps)
            elif devmodel in ['source_el']:
                # co2 from co2 content in fuel usage
                thisCO2 = sum(model.varDeviceFlow[d,'el','out',t]
                            *model.paramDevice[d]['co2em']
                            for t in timesteps)
            elif devmodel in ['compressor_el','sink_heat','sink_el',
                              'heatpump','source_gas','export_gas',
                              'export_el','storage_el']:
                # no CO2 emission contribution
                thisCO2 = 0
            else:
                raise NotImplementedError(
                    "CO2 calculation for {} not implemented".format(devmodel))
            sumCO2 = sumCO2 + thisCO2*deltaT
            
        # Average per hour
        sumCO2 = sumCO2/sumHours
        
        return sumCO2

        

    def createModelInstance(self,data,profiles,filename=None):
        """Create concrete Pyomo model instance
        
        data : dict of data
        filename : name of file to write model to (optional)
        """
        self._df_profiles_actual = profiles['actual']
        self._df_profiles_forecast = profiles['forecast']
        #self._df_profiles = self._df_profiles_forecast.loc[
        #        data['setHorizon'][None]]
        
        instance = self.model.create_instance(data={None:data},name='MultiCarrier')
        instance.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

        if filename is not None:
            instance.pprint(filename=filename)
        self.instance = instance
        return instance
    
    def updateModelInstance(self,timestep,df_profiles,first=False,filename=None):
        """Update Pyomo model instance
        
        first : True if it is the first timestep
        """
        opt_timesteps = self.instance.paramParameters['optimisation_timesteps']
        horizon = self.instance.paramParameters['planning_horizon']

        # Update profile (using actual for first 4 timesteps, forecast for rest)
        for prof in self.instance.setProfile:
            for t in range(opt_timesteps):
                self.instance.paramProfiles[prof,t] = (
                        self._df_profiles_actual.loc[timestep+t,prof])
            for t in range(opt_timesteps+1,horizon):
                self.instance.paramProfiles[prof,t] = (
                        self._df_profiles_forecast.loc[timestep+t,prof])
                   
        # Update startup/shutdown info
        # pick the last value from previous optimistion prior to the present time
        if not first:
            t_prev = opt_timesteps-1
            for dev in self.instance.setDevice:
                self.instance.paramDeviceIsOnInitially[dev] = (
                        self.instance.varDeviceIsOn[dev,t_prev])
                self.instance.paramDevicePowerInitially[dev] = (
                        self.instance.varDevicePower[dev,t_prev])
                if self.instance.paramDevice[dev]['model'] in ['storage_el']:
                    self.instance.paramDeviceEnergyInitially[dev] = (
                            self.instance.varDeviceEnergy[dev,t_prev])
        return

    def getVarValues(self,variable,names):
        df = pd.DataFrame.from_dict(variable.get_values(),orient="index")
        df.index = pd.MultiIndex.from_tuples(df.index,names=names)
        return df[0].dropna()
#        if unstack is None:
#            df = df[0]
#        else:
#            df = df[0].unstack(level=unstack)            
#        df = df.dropna()
#        return df

    def _keep_decision(self,df,timelimit,timeshift):
        '''extract decision variables (first timesteps) from dataframe'''
        level = df.index.names.index('time')
        df = df[df.index.get_level_values(level)<timelimit]
        df.index.set_levels(df.index.levels[level]+timeshift,level=level,
                            inplace=True)
        return df
        

    def saveOptimisationResult(self,timestep):
        '''save results of optimisation for later analysis'''
        
        #TODO: Implement result storage
        
        timelimit = self.instance.paramParameters['optimisation_timesteps']
        timeshift = timestep
        
        # Retrieve variable values
        varDeviceFlow = self.getVarValues(self.instance.varDeviceFlow,
              names=('device','carrier','terminal','time'))
        varDeviceIsOn = self.getVarValues(self.instance.varDeviceIsOn,
              names=('device','time'))
        varDevicePower = self.getVarValues(self.instance.varDevicePower,
              names=('device','time'))
        varDeviceEnergy = self.getVarValues(self.instance.varDeviceEnergy,
              names=('device','time'))
        varDeviceStarting = self.getVarValues(self.instance.varDeviceStarting,
              names=('device','time'))
        varDeviceStopping = self.getVarValues(self.instance.varDeviceStopping,
              names=('device','time'))
        varEdgePower = self.getVarValues(self.instance.varEdgePower,
              names=('edge','time'))
        varElVoltageAngle = self.getVarValues(self.instance.varElVoltageAngle,
              names=('node','time'))
        varGasPressure = self.getVarValues(self.instance.varGasPressure,
              names=('node','terminal','time'))
        varTerminalFlow = self.getVarValues(self.instance.varTerminalFlow,
              names=('node','carrier','time'))
               
        # Add to dataframes storing results (only the decision variables)
        def _addToDf(df_prev,df_new):
            level = df_new.index.names.index('time')
            df_new = df_new[df_new.index.get_level_values(level)<timelimit]
            df_new.index.set_levels(df_new.index.levels[level]+timeshift,
                                    level=level, inplace=True)
            df = pd.concat([df_prev,df_new])
            df.sort_index(inplace=True)
            return df        
        self._dfDeviceFlow = _addToDf(self._dfDeviceFlow,varDeviceFlow)
        self._dfDeviceIsOn = _addToDf(self._dfDeviceIsOn,varDeviceIsOn)
        self._dfDevicePower = _addToDf(self._dfDevicePower,varDevicePower)
        self._dfDeviceEnergy = _addToDf(self._dfDeviceEnergy,varDeviceEnergy)
        self._dfDeviceStarting = _addToDf(self._dfDeviceStarting,varDeviceStarting)
        self._dfDeviceStopping = _addToDf(self._dfDeviceStopping,varDeviceStopping)
        self._dfEdgePower = _addToDf(self._dfEdgePower,varEdgePower)
        self._dfElVoltageAngle = _addToDf(self._dfElVoltageAngle,varElVoltageAngle)
        self._dfGasPressure = _addToDf(self._dfGasPressure,varGasPressure)
        self._dfTerminalFlow = _addToDf(self._dfTerminalFlow,varTerminalFlow)
        return    

    def devicemodel_inout():
        inout = {
                'compressor_el':    {'in':['el','gas'],'out':['gas']},
                'compressor_gas':   {'in':['gas'],'out':['gas']},
                #'separator':        {'in':['el'],'out':[]},
                'export_gas':         {'in':['gas'],'out':[]},
                'sink_gas':         {'in':['gas'],'out':[]},
                'source_gas':       {'in':[],'out':['gas']},
                'gasheater':        {'in':['gas'],'out':['heat']},
                'gasturbine':       {'in':['gas'],'out':['el','heat']},
                'source_el':           {'in':[],'out':['el']},
                'export_el':          {'in':['el'],'out':[]},
                'sink_el':          {'in':['el'],'out':[]},
                'sink_heat':        {'in':['heat'],'out':[]},
                'heatpump':         {'in':['el'],'out':['heat']},
                'storage_el':       {'in':[], 'out':['el']},
                }
        return inout
    
    def solve(self,solver="gurobi",write_yaml=False):
        """Solve problem for planning horizon at a single timestep"""
        
        opt = pyo.SolverFactory(solver)
        logging.debug("Solving...")
        sol = opt.solve(self.instance) 
        
        if write_yaml:
            sol.write_yaml()
        
        if ((sol.solver.status == pyopt.SolverStatus.ok) and 
           (sol.solver.termination_condition == pyopt.TerminationCondition.optimal)):
            logging.debug("Solved OK")
        elif (sol.solver.termination_condition == pyopt.TerminationCondition.infeasible):
            raise Exception("Infeasible solution")
        else:
            # Something else is wrong
            logging.info("Solver Status:{}".format(sol.solver.status))
        return sol
 
    def solveMany(self,solver="gurobi",time_end=None,write_yaml=False):
        """Solve problem over many timesteps - rolling horizon"""
        
        logging.info("TODO: solve many data missing")
        steps = self.instance.paramParameters['optimisation_timesteps']
        horizon = self.instance.paramParameters['planning_horizon']
        if time_end is None:
            time_end = self._df_profiles_forecast.index.max()+1-horizon
        #steps = range(0,5,2) # [0,2,4]
        df_profiles = pd.DataFrame()
        
        for step in range(0,time_end,steps):
            logging.info("Solving timestep={}".format(step))
            # 1. Save present  power output and on/off status (for gas turbines)
            #TODO: save state for later analysis
            # 2. Update problem formulation
            first = False
            if step==0:
                first=True
            self.updateModelInstance(step,df_profiles,first=first)
            # 3. Solve for planning horizon
            self.solve(solver=solver,write_yaml=write_yaml)
            # 4. Save results (for later analysis)
            self.saveOptimisationResult(step)
            # hdf5? https://stackoverflow.com/questions/47072859/how-to-append-data-to-one-specific-dataset-in-a-hdf5-file-with-h5py)
            
        
        
        
    def printSolution(self,instance):
        print("\nSOLUTION - devicePower:")
        for k in instance.varDevicePower.keys():
            print("  {}: {}".format(k,instance.varDevicePower[k].value))
        
        print("\nSOLUTION - edgePower:")
        for k in instance.varEdgePower.keys():
            power = instance.varEdgePower[k].value
            print("  {}: {}".format(k,power))
            #df_edge.loc[k,'edgePower'] = power
        
        print("\nSOLUTION - gasPressure:")
        for k,v in instance.varGasPressure.get_values().items():
            pressure = v #pyo.value(v)
            print("  {}: {}".format(k,pressure))
            #df_edge.loc[k,'gasPressure'] = pressure
        
        # display all duals
        print ("Duals")
        for c in instance.component_objects(pyo.Constraint, active=True):
            print ("   Constraint",c)
            for index in c:
                print ("      ", index, instance.dual[c[index]])

    def nodeIsNonTrivial(model,node,carrier):
        '''returns True if edges or devices are connected to node for this carrier
        '''
        isNontrivial = False
        if (((carrier,node) in model.paramNodeEdgesFrom) or
                ((carrier,node) in model.paramNodeEdgesTo)):
            isNontrivial = True
            return isNontrivial
        if node in model.paramNodeDevices:
            mydevs = model.paramNodeDevices[node]
            devmodels = [model.paramDevice[d]['model'] for d in mydevs]
            for dev_model in devmodels:
                carriers_used = [item for sublist in 
                     list(model.paramDevicemodel[dev_model].values()) 
                     for item in sublist]
                if(carrier in carriers_used):
                    isNontrivial = True
                    return isNontrivial
        return isNontrivial
    
    def getProfiles(self,names):
        if not type(names) is list:
            names = [names]
        #return self._df_profiles[names]
        df=pd.DataFrame.from_dict(
                self.instance.paramProfiles.extract_values(),orient='index')
        df.index=pd.MultiIndex.from_tuples(df.index)
        df = df[0].unstack(level=0)
        return df
    
class Plots:  
      
    def plotGasTurbineEfficiency(fuelA=1,fuelB=1, filename=None):
        #fuelA = model.paramDevice[dev]['fuelA']
        #fuelB = model.paramDevice[dev]['fuelB']
        x_pow = pd.np.linspace(0,1,50)
        y_fuel = fuelA + fuelB*x_pow
        plt.figure(figsize=(12,4))
        #plt.suptitle("Gas turbine fuel characteristics")
        
        plt.subplot(1,3,1)
        plt.title("Fuel usage ($P_{gas}$ vs $P_{el}$)")
        plt.xlabel("Electric power output ($P_{el}/P_{max}$)")
        plt.ylabel("Gas power input ($P_{gas}/P_{max}$)")
        plt.plot(x_pow,y_fuel)
        plt.ylim(bottom=0)
        
        plt.subplot(1,3,2)
        plt.title("Specific fuel usage ($P_{gas}/P_{el}$)")
        plt.xlabel("Electric power output ($P_{el}/P_{max}$)")
        plt.plot(x_pow,y_fuel/x_pow)
        plt.ylim(top=30)
        
        plt.subplot(1,3,3)
        plt.title("Efficiency ($P_{el}/P_{gas}$)")
        plt.xlabel("Electric power output ($P_{el}/P_{max}$)")
        plt.plot(x_pow,x_pow/y_fuel)
        if filename is not None:
            plt.savefig(filename,bbox_inches = 'tight')        

       
    def plotNetworkCombined(model,timestep=0,filename='pydotCombined.png',
                            only_carrier=None):
        """Plot energy network
        
        model : object
            Multicarrier Pyomo instance
        filename : string
            Name of file
        only_carrier : list
            Restrict energy carriers to these types (None=plot all)
        """
        cluster = {}
        col = {'t': {'el':'red','gas':'blue','heat':'darkgreen'},
               'e': {'el':'red','gas':'blue','heat':'darkgreen'},
               'd': 'orange',
               'cluster':'lightgray'
               }
        dotG = pydot.Dot(graph_type='digraph') #rankdir='LR',newrank='false')
        
        if only_carrier is None:
            carriers = model.setCarrier
        else:
            carriers = [only_carrier]
            fparts=filename.split('.')
            fparts.insert(-1,only_carrier)
            filename = '.'.join(fparts)
        
        
        # plot all node and terminals:
        for n_id in model.setNode:
            cluster[n_id] = pydot.Cluster(graph_name=n_id,label=n_id,
                   style='filled',color=col['cluster'])
            nodes_in=pydot.Subgraph(rank='same')
            nodes_out=pydot.Subgraph(rank='same')
            for carrier in carriers:
                #add only terminals that are connected to something
                if Multicarrier.nodeIsNonTrivial(model,n_id,carrier):
                    label_in = carrier+'_in'
                    label_out= carrier+'_out'
                    if carrier=='gas':
                        label_in +=':{:3.1f}'.format(pyo.value(model.varGasPressure[n_id,'in',timestep]))
                        label_out +=':{:3.1f}'.format(pyo.value(model.varGasPressure[n_id,'out',timestep]))
                    elif carrier=='el':
                        label_in +=':{:3.2g}'.format(pyo.value(model.varElVoltageAngle[n_id,timestep]))
                        label_out +=':{:3.2g}'.format(pyo.value(model.varElVoltageAngle[n_id,timestep]))
                    nodes_in.add_node(pydot.Node(name=n_id+'_'+carrier+'_in',
                           color=col['t'][carrier],label=label_in,shape='box'))
                    nodes_out.add_node(pydot.Node(name=n_id+'_'+carrier+'_out',
                           color=col['t'][carrier],label=label_out,shape='box'))
            cluster[n_id].add_subgraph(nodes_in)
            cluster[n_id].add_subgraph(nodes_out)
            dotG.add_subgraph(cluster[n_id])
        
        # plot all edges (per carrier):
        for carrier in carriers:
            for i,e in model.paramEdge.items():
                if e['type']==carrier:
                    edgelabel = '{:.2f}'.format(pyo.value(model.varEdgePower[i,timestep]))
                    dotG.add_edge(pydot.Edge(src=e['nodeFrom']+'_'+carrier+'_out',
                                             dst=e['nodeTo']+'_'+carrier+'_in',
                                             color='"{0}:invis:{0}"'.format(col['e'][carrier]),
                                             fontcolor=col['e'][carrier],
                                             label=edgelabel))
        
        # plot devices and device connections:
        for n,devs in model.paramNodeDevices.items():
            for d in devs:
                dev_model = model.paramDevice[d]['model']
                p_dev = pyo.value(model.varDevicePower[d,timestep])
                carriers_in = model.paramDevicemodel[dev_model]['in']
                carriers_out = model.paramDevicemodel[dev_model]['out']
                carriers_in_lim = list(set(carriers_in)&set(carriers))
                carriers_out_lim = list(set(carriers_out)&set(carriers))
                if (carriers_in_lim!=[]) or (carriers_out_lim!=[]):
                    cluster[n].add_node(pydot.Node(d,color=col['d'],style='filled',
                       label='"{}[{}]\n{:.2f}"'
                       .format(d,model.paramDevice[d]['name'],p_dev)))
                #print("carriers in/out:",d,carriers_in,carriers_out)
                for carrier in carriers_in_lim:
                    f_in = pyo.value(model.varDeviceFlow[d,carrier,'in',timestep])
                    dotG.add_edge(pydot.Edge(dst=d,src=n+'_'+carrier+'_in',
                         color=col['e'][carrier],
                         fontcolor=col['e'][carrier],
                         label="{:.2f}".format(f_in)))
                for carrier in carriers_out_lim:
                    f_out = pyo.value(model.varDeviceFlow[d,carrier,'out',timestep])
                    dotG.add_edge(pydot.Edge(dst=n+'_'+carrier+'_out',src=d,
                         color=col['e'][carrier],
                         fontcolor=col['e'][carrier],
                         label="{:.2f}".format(f_out)))
        
        # plot terminal in-out links:
        for n in model.setNode:
            for carrier in carriers:
                 if not model.paramNodeCarrierHasSerialDevice[n][carrier]:
                     if Multicarrier.nodeIsNonTrivial(model,n,carrier):    
                        flow = pyo.value(model.varTerminalFlow[n,carrier,timestep])
                        dotG.add_edge(pydot.Edge(dst=n+'_'+carrier+'_out',
                                                 src=n+'_'+carrier+'_in',
                             color=col['e'][carrier],
                             label='{:.2f}'.format(flow),fontcolor=col['e'][carrier]))
                             #arrowhead='none'))
    
        #dotG.write_dot('pydotCombinedNEW.dot',prog='dot')                
        dotG.write_png(filename,prog='dot')    

    def plotDevicePowerLastOptimisation1(mc,device,filename=None):
        model = mc.instance
        profile = model.paramDevice[device]['external']
        devname = model.paramDevice[device]['name']
        maxP = model.paramDevice[device]['Pmax']
        plt.figure(figsize=(12,4))
        ax=plt.gca()
        plt.xlabel("Timestep")
        plt.ylabel("Device power (MW)")
        label_profile=""
        if not pd.isna(profile):
            dfa = mc.getProfiles(profile)
            dfa = dfa*maxP
            dfa[profile].plot(ax=ax,label='available')
            label_profile="({})".format(profile)

#        df = pd.DataFrame.from_dict(model.varDevicePower.get_values(),
#                                    orient="index")
#        df.index = pd.MultiIndex.from_tuples(df.index,names=('device','time'))
#        df = df[0].unstack(level=0)
#        df[device].plot(ax=ax,label='actual')
        
        dfIn = pd.DataFrame.from_dict(model.varDeviceFlow.get_values(),
                                    orient="index")
        dfIn.index = pd.MultiIndex.from_tuples(dfIn.index,
                               names=('device','carrier','terminal','time'))
        dfIn = dfIn[0].dropna()
        for carr in dfIn.index.levels[1]:
            for term in dfIn.index.levels[2]:
                mask = ((dfIn.index.get_level_values(1)==carr) &
                        (dfIn.index.get_level_values(2)==term))
                df_this = dfIn[mask].unstack(0).reset_index()
                if device in df_this:
                    df_this[device].plot(ax=ax,linestyle='-',
                           label="actual ({} {})".format(carr,term))
        ax.legend(loc='upper left')#, bbox_to_anchor =(1.01,0),frameon=False)

        dfE = pd.DataFrame.from_dict(model.varDeviceEnergy.get_values(),
                                    orient="index")
        dfE.index = pd.MultiIndex.from_tuples(dfE.index,
                               names=('device','time'))
        dfE = dfE[0].dropna()
        df_this = dfE.unstack(0)
        if device in df_this:
            ax2=ax.twinx()
            ax2.set_ylabel("Energy (MWh)",color="red")
            ax2.tick_params(axis='y', labelcolor="red")
            df_this[device].plot(ax=ax2,linestyle='--',color='red',
                   label="storage".format(carr,term))
            ax2.legend(loc='upper right',)


        plt.title("{}:{} {}".format(device,devname,label_profile))
        if filename is not None:
            plt.savefig(filename,bbox_inches = 'tight')
        
    
    def plotDevicePowerLastOptimisation(model,devices='all',filename=None):
        """Plot power schedule over planning horizon (last optimisation)"""
        if devices=='all':
            devices = list(model.setDevice)
        df = pd.DataFrame.from_dict(model.varDevicePower.get_values(),
                                    orient="index")
        df.index = pd.MultiIndex.from_tuples(df.index,names=('device','time'))
        df = df[0].unstack(level=0)
        df_info = pd.DataFrame.from_dict(dict(model.paramDevice.items())).T

        plt.figure(figsize=(12,4))
        ax=plt.gca()
        df[devices].plot(ax=ax)
        labels = (df_info.loc[devices].index.astype(str)
                  +'_'+df_info.loc[devices,'name'])
        plt.legend(labels,loc='lower left', bbox_to_anchor =(1.01,0),
                   frameon=False)
        plt.xlabel("Timestep")
        plt.ylabel("Device power (MW)")
        if filename is not None:
            plt.savefig(filename,bbox_inches = 'tight')


    def plotDeviceSumPowerLastOptimisation(model,carrier='el',filename=None):
        """Plot power schedule over planning horizon (last optimisation)"""
        
        df = pd.DataFrame.from_dict(model.varDeviceFlow.get_values(),
                                    orient="index")
        df.index = pd.MultiIndex.from_tuples(df.index,
                     names=('device','carrier','inout','time'))
        
        # separate out in out
        df = df[0].unstack(level=2)
        df = df.fillna(0)
        df = df['out']-df['in']
        df = df.unstack(level=1)
        dfprod = df[df>0][carrier].dropna()
        dfcons = df[df<0][carrier].dropna()
#        dfprod = df[df>=0].unstack(level=1)[carrier]
#        dfprod = dfprod[dfprod>0]
#        dfcons = df[df<0].unstack(level=1)[carrier]
#        dfcons = dfcons[dfcons<0]
        
        df_info = pd.DataFrame.from_dict(dict(model.paramDevice.items())).T
        labels = (df_info.index.astype(str) +'_'+df_info['name'])

        #plt.figure(figsize=(12,4))
        fig,axes = plt.subplots(nrows=2,ncols=1,figsize=(12,8))
        dfprod.unstack(level=0).rename(columns=labels).plot.area(ax=axes[0],linewidth=0)
        # reverse axes to match stack order
        handles, lgnds = axes[0].get_legend_handles_labels()
        axes[0].legend(handles[::-1], lgnds[::-1],
                        loc='lower left', bbox_to_anchor =(1.01,0),
                   frameon=False)
        axes[0].set_ylabel("Produced power (MW)")
        axes[0].set_xlabel("")

        dfcons.unstack(level=0).rename(columns=labels).plot.area(ax=axes[1])      
        axes[1].legend(loc='lower left', bbox_to_anchor =(1.01,0),
                   frameon=False)
        axes[1].set_ylabel("Consumed power (MW)")
        
        axes[1].set_xlabel("Timestep")
        if filename is not None:
            plt.savefig(filename,bbox_inches = 'tight')
            
    def plotEmissionRateLastOptimisation(model,filename=None):
        devices = model.setDevice
        timesteps = model.setHorizon
        df_info = pd.DataFrame.from_dict(dict(model.paramDevice.items())).T
        labels = (df_info.index.astype(str) +'_'+df_info['name'])

        df = pd.DataFrame(index=timesteps,columns=devices)
        for d in devices:
            for t in timesteps:
                co2 = Multicarrier.compute_CO2(model,devices=[d],timesteps=[t])
                df.loc[t,d] = pyo.value(co2)
        plt.figure(figsize=(12,4))
        ax=plt.gca()
        df.loc[:,~(df==0).all()].rename(columns=labels).plot.area(ax=ax,linewidth=0)
        plt.xlabel("Timestep")
        plt.ylabel("Emission rate (kgCO2/hour)")
        ax.legend(loc='lower left', bbox_to_anchor =(1.01,0),
                  frameon=False)
        if filename is not None:
            plt.savefig(filename,bbox_inches = 'tight')
        
    
    def plotProfiles(profiles,curves=None,filename=None):
        '''Plot profiles (forecast and actual)'''
        
        plt.figure(figsize=(12,4))
        ax=plt.gca()
        if curves is None:
            curves = profiles['actual'].columns
        profiles['actual'][curves].plot(ax=ax)
        ax.set_prop_cycle(None)
        profiles['forecast'][curves].plot(ax=ax,linestyle=":")

        labels = curves
        ax.legend(labels=labels,loc='lower left', bbox_to_anchor =(1.01,0),
                  frameon=False)
        plt.xlabel("Timestep")
        plt.ylabel("Relative value")
        plt.title("Actual vs forecast profile (actual=sold line)")
        if filename is not None:
            plt.savefig(filename,bbox_inches = 'tight')
        
    

#########################################################################

def read_data_from_xlsx(filename,carrier_properties):
    """Read input data from spreadsheet.
    
    filename : str
        name of file
    carrier_properties: dict
        dictionary of all energy carriers and their specific properties
    """
        
    
    # Input data
    df_node = pd.read_excel(filename,sheet_name="node")
    df_edge = pd.read_excel(filename,sheet_name="edge")
    df_device = pd.read_excel(filename,sheet_name="device")
    df_parameters = pd.read_excel(filename,sheet_name="parameters",index_col=0)
    df_profiles = pd.read_excel(filename,sheet_name="profiles",index_col=0)
    df_profiles_forecast = pd.read_excel(filename,
                                         sheet_name="profiles_forecast",index_col=0)
    profiles = {'actual':df_profiles,'forecast':df_profiles_forecast}
    
    # discard edges and devices not to be included:    
    df_edge = df_edge[df_edge['include']==1]
    df_device = df_device[df_device['include']==1]
    
#    #into node:
#    cols_in = {col:col.split("in_")[1] 
#               for col in df_device.columns if "in_" in col }
#    dispatch_in = df_device[list(cols_in.keys())].rename(columns=cols_in).fillna(0)
#    #out of node:
#    cols_out = {col:col.split("out_")[1] 
#               for col in df_device.columns if "out_" in col }
#    dispatch_out = df_device[list(cols_out.keys())].rename(columns=cols_out).fillna(0)
#    
#    df_deviceR = df_device.drop(cols_in,axis=1).drop(cols_out,axis=1)
    
    allCarriers = list(carrier_properties.keys())

    ## find nodes where no devices connect in-out terminals:
    ## (node is non-trivial if any device connects both in and out)
    #dev_nontrivial = ((dispatch_in!=0) & (dispatch_out!=0))
    #node_nontrivial = pd.concat([df_device[['node']],
    #                             dev_nontrivial],axis=1).groupby('node').any()
    node_devices = df_device.groupby('node').groups
    devmodel_inout = Multicarrier.devicemodel_inout()
    node_carrier_has_serialdevice = {}
    for n in df_node['id']:
        devs=[]
        if n in node_devices:
            devs = node_devices[n]
        node_carrier_has_serialdevice[n] = {}
        for carrier in allCarriers:
            num_series = 0
            for dev_mod in df_device.loc[devs,'model']:
                has_series = ((carrier in devmodel_inout[dev_mod]['in'])
                                and (carrier in devmodel_inout[dev_mod]['out']))
                if has_series:
                    num_series +=1
                #print("series: {},{},{}".format(n,carrier,dev_mod))
            #print(n,carrier,num_series)
            node_carrier_has_serialdevice[n][carrier] = (num_series>0)
            
    # gas pipeline parameters - derive k and exp(s) parameters:
    ga=carrier_properties['gas']
    temp = df_edge['temperature_K']
    height_from = (df_edge.reset_index()
                    .merge(df_node,how='left',left_on='nodeFrom',right_on='id')
                    .set_index('index')['coord_z']*1000)
    height_to = (df_edge.reset_index()
                    .merge(df_node,how='left',left_on='nodeTo',right_on='id')
                    .set_index('index')['coord_z']*1000)
    s = 0.0684 * (ga['G_gravity']*(height_to-height_from)
                    /(temp*ga['Z_compressibility']))
    sfactor= (pd.np.exp(s)-1)/s
    sfactor.loc[s==0] = 1
    length = df_edge['length_km']*sfactor
    diameter = df_edge['diameter_mm']
    
    gas_edge_k = (4.3328e-8*ga['Tb_basetemp_K']/ga['Pb_basepressure_kPa']
        *(ga['G_gravity']*temp*length*ga['Z_compressibility'])**(-1/2)
        *diameter**(8/3))
    df_edge['gasflow_k'] = gas_edge_k
    df_edge['exp_s'] = pd.np.exp(s)
        
    coeffB,coeffDA = computePowerFlowMatrices(df_node,df_edge,baseZ=1)
    planning_horizon = df_parameters.loc['planning_horizon','value']
    data = {}
    data['setCarrier'] = {None:allCarriers}
    data['setNode'] = {None:df_node['id'].tolist()}
    data['setEdge'] = {None: df_edge.index.tolist()}
    data['setDevice'] = {None:df_device.index.tolist()}
    data['setDevicemodel'] = {None:devmodel_inout.keys()}
    data['setHorizon'] = {
            None:range(planning_horizon)}
    data['setParameters'] = {None:df_parameters.index.tolist()}
    data['setProfile'] = {None:df_device['external'].dropna().unique().tolist()}
#    data['paramDeviceDispatchIn'] = dispatch_in.to_dict(orient='index') 
#    data['paramDeviceDispatchOut'] = dispatch_out.to_dict(orient='index') 
    data['paramNode'] = df_node.set_index('id').to_dict(orient='index')
    data['paramNodeCarrierHasSerialDevice'] = node_carrier_has_serialdevice
    data['paramNodeDevices'] = df_device.groupby('node').groups
    data['paramDevice'] = df_device.to_dict(orient='index')
    data['paramDeviceIsOnInitially'] = {k:0 for k in df_device.index.tolist()}
    data['paramDevicePowerInitially'] = {k:0 for k in df_device.index.tolist()}
    data['paramDeviceEnergyInitially'] = {k:0 for k in df_device.index.tolist()}
    data['paramEdge'] = df_edge.to_dict(orient='index')
    data['paramNodeEdgesFrom'] = df_edge.groupby(['type','nodeFrom']).groups
    data['paramNodeEdgesTo'] = df_edge.groupby(['type','nodeTo']).groups
    data['paramDevicemodel'] = devmodel_inout
    data['paramParameters'] = df_parameters['value'].to_dict()#orient='index')
    #unordered set error - but is this needed - better use dataframe diretly instead?
    data['paramProfiles'] = df_profiles_forecast.loc[
            range(planning_horizon),data['setProfile'][None]
            ].T.stack().to_dict()
    data['paramCarriers'] = carrier_properties
    data['paramCoeffB'] = coeffB
    data['paramCoeffDA'] = coeffDA
    
    return data,profiles


#def _susceptancePu(df_edge,baseOhm=1):
#    '''If impedance is already given in pu, baseOhm should be 1
#    If not, well... baseOhm depends on the voltage level, so need to know
#    the nominal voltage at the bus to convert from ohm to pu.
#    '''
#    #return [-1/self.branch['reactance'][i]*baseOhm 
#    #        for i in self.branch.index.tolist()]
#    return 1/df_edge['reactance']*baseOhm

def computePowerFlowMatrices(df_node,df_edge,baseZ=1):
    """
    Compute and return dc power flow matrices B' and DA
    
    Parameters
    ==========
    baseZ : float (impedance should already be in pu.)
            base value for impedance        
    
    Returns 
    =======
    (coeff_B, coeff_DA) : dictionary of matrix values
          
    """
    
    df_branch = df_edge[df_edge['type']=='el']
    #el_edges = df_edge[df_edge['type']=='el'].index
    
    b = (1/df_branch['reactance']*baseZ)
    #b0 = pd.np.asarray(_susceptancePu(baseZ))

    # MultiDiGraph to allow parallel lines
    G = nx.MultiDiGraph()
    edges = [(df_branch['nodeFrom'][i],
              df_branch['nodeTo'][i],
              i,{'i':i,'b':b[i]})
              for i in df_branch.index]
    G.add_nodes_from(df_node['id'])
    G.add_edges_from(edges)   
    A_incidence_matrix = -nx.incidence_matrix(G,oriented=True,
                                             nodelist=df_node['id'],
                                             edgelist=edges).T
    # Diagonal matrix
    D = scipy.sparse.diags(-b,offsets=0)
    DA = D*A_incidence_matrix
    
    # Bf constructed from incidence matrix with branch susceptance
    # used as weight (this is quite fast)
    Bf = -nx.incidence_matrix(G,oriented=True,
                             nodelist=df_node['id'],
                             edgelist=edges,
                             weight='b').T
    Bbus = A_incidence_matrix.T * Bf

    n_i = df_node['id']
    b_i = df_edge[df_edge['type']=='el'].index
    coeff_B = dict()
    coeff_DA = dict()

    #logging.info("Creating B and DA coefficients...")
    cx = scipy.sparse.coo_matrix(Bbus)
    for i,j,v in zip(cx.row, cx.col, cx.data):
        coeff_B[(n_i[i],n_i[j])] = v

    cx = scipy.sparse.coo_matrix(DA)
    for i,j,v in zip(cx.row, cx.col, cx.data):
        coeff_DA[(b_i[i],n_i[j])] = v
    
    return coeff_B,coeff_DA




