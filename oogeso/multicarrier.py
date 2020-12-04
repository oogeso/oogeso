import pyomo.environ as pyo
import pyomo.opt as pyopt
import numpy as np
import pandas as pd
import networkx as nx
import scipy
import matplotlib.pyplot as plt
import logging
import yaml

# get progress bar:
try:
    from tqdm import trange
    has_tqdm = True
except:
    logging.info("Consider installing tqdm to get progress bar")
    trange = range
    has_tqdm = False

'''
TODO:well start/stop? (or continuous regulation of wellstream Pmax/Pmin)

TODO: flexible demand (water injection)

TODO: weymouth equation linearisation improvement:
-> closing well -> flow rate much less -> pressure drop much less ->
linearisation points no longer good -> error

TODO: It will probably be more straightforward to directly define the problem
as a concretemodel (with some mutable parameters)
'''

ZERO_WARNING_THRESHOLD = 1e-6
class Multicarrier:
    """Multicarrier energy system"""

    @staticmethod
    def devicemodel_inout():
        inout = {
                'compressor_el':    {'in':['el','gas'],'out':['gas'],
                                     'serial':['gas']},
                'compressor_gas':   {'in':['gas'],'out':['gas'],
                                     'serial':['gas']},
                'separator':        {'in':['wellstream','el','heat'],
                                     'out':['oil','gas','water']},
                'separator2':        {'in':['oil','gas','water','el','heat'],
                                     'out':['oil','gas','water'],
                                     'serial':['oil','gas','water']},
                'well_production':  {'in':[],'out':['wellstream']},
                'well_gaslift':     {'in':['gas'],'out':['oil','gas','water'],
                                     'serial':['gas']},
                'sink_gas':         {'in':['gas'],'out':[]},
                'sink_oil':         {'in':['oil'], 'out':[]},
                'sink_el':          {'in':['el'],'out':[]},
                'sink_heat':        {'in':['heat'],'out':[]},
                'sink_water':        {'in':['water'],'out':[]},
                'source_gas':       {'in':[],'out':['gas']},
                'source_water':       {'in':[],'out':['water']},
                'source_oil':       {'in':[],'out':['oil']},
                'gasheater':        {'in':['gas'],'out':['heat']},
                'gasturbine':       {'in':['gas'],'out':['el','heat']},
                'source_el':           {'in':[],'out':['el']},
                'heatpump':         {'in':['el'],'out':['heat']},
                'storage_el':       {'in':['el'], 'out':['el']},
                'pump_oil':         {'in':['oil','el'], 'out':['oil'],
                                     'serial':['oil']},
                'pump_water':       {'in':['water','el'],'out':['water'],
                                     'serial':['water']},
                'pump_wellstream':  {'in':['wellstream','el'],'out':['wellstream'],
                                     'serial':['wellstream']},
                }
        return inout

    models_with_storage = ['storage_el','well_injection']

    def __init__(self,loglevel=logging.INFO,logfile=None,
                 quadraticConstraints=True):
        #logging.basicConfig(filename=logfile,level=loglevel,
        #                    format='%(asctime)s %(levelname)s: %(message)s',
        #                    datefmt='%Y-%m-%d %H:%M:%S')
        logging.basicConfig()
        logger = logging.getLogger()
        logger.setLevel(loglevel)
        logging.debug("Initialising Multicarrier")
        # Abstract model:
        self._quadraticConstraints = quadraticConstraints
        self.model = self._createPyomoModel()
        self._check_constraints_complete()
        # Concrete model instance:
        self.instance = None
        self.elbase = {
                'baseMVA':100,
                'baseAngle':1}
        #self._df_profiles = None
        self._devmodels = Multicarrier.devicemodel_inout()

        self._dfDeviceFlow = None
        self._dfDeviceIsPrep = None
        self._dfDeviceIsOn = None
#        self._dfDevicePower = None
        self._dfDeviceStorageEnergy = None
        self._dfDeviceStoragePmax = None
        self._dfDeviceStarting = None
        self._dfDeviceStopping = None
        self._dfEdgeFlow = None
        self._dfElVoltageAngle = None
        self._dfTerminalPressure = None
        self._dfTerminalFlow = None
        self._dfCO2rate = None #co2 emission sum per timestep
        self._dfCO2rate_per_dev = None # co2 emission per device per timestep
        self._dfExportRevenue = None #revenue from exported energy
        self._dfCO2intensity = None
        self._dfElReserve = None #Reserve capacity
        self._dfElBackup = None #Backup capacity (handling faults)
        self._dfDuals = None


    def _createPyomoModel(self):
        model = pyo.AbstractModel()

        # Sets
        model.setCarrier = pyo.Set(doc="energy carrier")
        model.setNode = pyo.Set()
        model.setEdge= pyo.Set()
        model.setDevice = pyo.Set()
        model.setTerminal = pyo.Set(initialize=['in','out'])
#        model.setEdgeWithFlowComponents= pyo.Set()
        model.setFlowComponent= pyo.Set(initialize=['oil','gas','water'])
        #model.setDevicemodel = pyo.Set()
        # time for rolling horizon optimisation:
        model.setHorizon = pyo.Set(ordered=True)
        model.setParameters = pyo.Set()
        model.setProfile = pyo.Set()

        # Parameters (input data)
        model.paramNode = pyo.Param(model.setNode,within=pyo.Any)
        model.paramEdge = pyo.Param(model.setEdge,within=pyo.Any)
        model.paramDevice = pyo.Param(model.setDevice,within=pyo.Any)
        model.paramNodeCarrierHasSerialDevice = pyo.Param(
            model.setNode,within=pyo.Any)
        model.paramNodeDevices = pyo.Param(
            model.setNode,within=pyo.Any)
        model.paramNodeEdgesFrom = pyo.Param(
            model.setCarrier,model.setNode,within=pyo.Any)
        model.paramNodeEdgesTo = pyo.Param(
            model.setCarrier,model.setNode,within=pyo.Any)
        model.paramParameters = pyo.Param(
            model.setParameters,within=pyo.Any)
        model.paramCarriers = pyo.Param(
            model.setCarrier,within=pyo.Any)

        model.paramCoeffB = pyo.Param(
            model.setNode,model.setNode,within=pyo.Reals)
        model.paramCoeffDA = pyo.Param(
            model.setEdge,model.setNode,within=pyo.Reals)
        # Mutable parameters (will be modified between successive optimisations)
        model.paramProfiles = pyo.Param(model.setProfile,model.setHorizon,
                                        within=pyo.Reals, mutable=True,
                                        initialize=0)
        model.paramDeviceIsOnInitially = pyo.Param(model.setDevice,
                                               mutable=True,within=pyo.Binary,
                                               initialize=0)
        model.paramDevicePrepTimestepsInitially = pyo.Param(model.setDevice,
            mutable=True, within=pyo.Integers, initialize=0)
        # needed for ramp rate limits:
        model.paramDevicePowerInitially = pyo.Param(model.setDevice,
                                               mutable=True,within=pyo.Reals,
                                               initialize=0)
        # needed for energy storage:
        model.paramDeviceEnergyInitially = pyo.Param(model.setDevice,
                                               mutable=True,within=pyo.Reals,
                                               initialize=0)

        # Variables
        #model.varNodeVoltageAngle = pyo.Var(model.setNode,within=pyo.Reals)
        model.varEdgeFlow = pyo.Var(
                model.setEdge,model.setHorizon,within=pyo.Reals)
#        model.varEdgeFlowComponent = pyo.Var(
#                model.setEdgeWithFlowComponents,model.setFlowComponent,
#                model.setHorizon,within=pyo.Reals)
        model.varDeviceIsPrep = pyo.Var(
                model.setDevice,model.setHorizon,within=pyo.Binary,initialize=0)
        model.varDeviceIsOn = pyo.Var(
                model.setDevice,model.setHorizon,within=pyo.Binary,initialize=0)
        model.varDeviceStarting = pyo.Var(
                model.setDevice,model.setHorizon,within=pyo.Binary)
        model.varDeviceStopping = pyo.Var(
                model.setDevice,model.setHorizon,within=pyo.Binary)
        model.varDeviceStorageEnergy = pyo.Var(model.setDevice,model.setHorizon,
                                        within=pyo.Reals)
        # available reserve power from storage (linked to power rating and storage level):
        model.varDeviceStoragePmax = pyo.Var(model.setDevice,model.setHorizon,
            within=pyo.NonNegativeReals,initialize=0)
        # binary variable related to available powr from storage:
        model.varStorY = pyo.Var(model.setDevice,model.setHorizon,
            within=pyo.Binary)
        model.varPressure = pyo.Var(
                model.setNode,model.setCarrier,model.setTerminal,
                model.setHorizon, within=pyo.NonNegativeReals,initialize=0)
        model.varElVoltageAngle = pyo.Var(
                model.setNode,model.setHorizon,
                within=pyo.Reals)
        model.varDeviceFlow = pyo.Var(
                model.setDevice,model.setCarrier,model.setTerminal,
                model.setHorizon,within=pyo.NonNegativeReals)
        model.varTerminalFlow = pyo.Var(
                model.setNode,model.setCarrier,model.setHorizon,
                within=pyo.Reals)
#        model.varTerminalFlowComponent = pyo.Var(
#                model.setNode,model.setFlowComponent,model.setHorizon,
#                within=pyo.Reals)



        # Objective

#        def storPmaxPushup(model):
#            '''term in objective function to push varDeviceStoragePmax up
#            to its maximum value (to get correct calculation of reserve)'''
#            sumStorPmax=0
#            for dev in model.setDevice:
#                if model.paramDevice[dev]['model'] == 'storage_el':
#                    for t in model.setHorizon:
#                        sumStorPmax += model.varDeviceStoragePmax[dev,t]
#            return sumStorPmax



        def rule_objective_co2(model):
            '''CO2 emissions per sec'''
            sumE = self.compute_CO2(model) #*model.paramParameters['CO2_price']
            return sumE


        def rule_objective_co2intensity(model):
            '''CO2 emission intensity (CO2 per exported oil/gas)
            DOES NOT WORK - NONLINEAR (ratio)'''
            sumE = self.compute_CO2_intensity(model)
            return sumE

        def rule_objective_exportRevenue(model):
            '''revenue from exported oil and gas minus costs (co2 price and
             operating costs) per second'''
            sumRevenue = self.compute_exportRevenue(model) # kr/s
            startupCosts = self.compute_startup_costs(model) # kr/s
            co2 = self.compute_CO2(model) # kgCO2/s
            co2_tax = model.paramParameters['co2_tax'] # kr/kgCO2
            co2Cost = co2*co2_tax # kr/s
            storageDepletionCosts = self.compute_costForDepletedStorage(model)
            opCosts = self.compute_operatingCosts(model) # kr/s
            sumCost = (-sumRevenue
                        + co2Cost
                        + startupCosts
                        + storageDepletionCosts
                        + opCosts
                        )
            return sumCost

        def rule_objective(model):
            obj = model.paramParameters['objective']
            if obj=='co2':
                rule = rule_objective_co2(model)
            elif obj=='exportRevenue':
                rule = rule_objective_exportRevenue(model)
            elif obj=='co2intensity':
                rule = rule_objective_co2intensity(model)
            else:
                raise Exception("Objective '{}' has not been implemented"
                                .format(obj))
            return rule


        model.objObjective = pyo.Objective(rule=rule_objective,
                                           sense=pyo.minimize)
#        model.objObjective = pyo.Objective(rule=rule_objective_exportRevenue,
#                                           sense=pyo.minimize)





        def rule_devmodel_well_production(model,dev,t,i):
            if model.paramDevice[dev]['model'] != 'well_production':
                return pyo.Constraint.Skip
            if i==1:
                node = model.paramDevice[dev]['node']
                lhs = model.varPressure[(node,'wellstream','out',t)]
                rhs = model.paramDevice[dev]['naturalpressure']
                return (lhs==rhs)
        model.constrDevice_well_production = pyo.Constraint(model.setDevice,
                  model.setHorizon,pyo.RangeSet(1,1),
                  rule=rule_devmodel_well_production)

        def rule_devmodel_well_gaslift(model,dev,carrier,t,i):
            if model.paramDevice[dev]['model'] != 'well_gaslift':
                return pyo.Constraint.Skip

            # flow from reservoir (equals flow out minus gas injection)
            Q_reservoir = (sum(model.varDeviceFlow[dev,c,'out',t]
                            for c in model.setFlowComponent)
                            -model.varDeviceFlow[dev,'gas','in',t])
            node = model.paramDevice[dev]['node']
            if i==1:
                # output pressure is fixed
                lhs = model.varPressure[(node,carrier,'out',t)]
                rhs = model.paramDevice[dev]['separatorpressure']
                return (lhs==rhs)
            elif i==2:
                # output flow per comonent determined by GOR and WC
                GOR = model.paramDevice[dev]['gas_oil_ratio']
                WC = model.paramDevice[dev]['water_cut']
                comp_oil = (1-WC)/(1+GOR-GOR*WC)
                comp_water = WC/(1+GOR-GOR*WC)
                comp_gas = GOR*(1-WC)/(1+GOR*(1-WC))
                comp = {'oil':comp_oil,'gas':comp_gas,'water':comp_water}
                lhs = model.varDeviceFlow[dev,carrier,'out',t]
                if carrier=='gas':
                    lhs -= model.varDeviceFlow[dev,carrier,'in',t]
                rhs = comp[carrier]*Q_reservoir
                return (lhs==rhs)
            elif i==3:
                # gas injection rate vs proudction rate (determines input gas)
                # gas injection rate vs OIL proudction rate (determines input gas)
                if carrier=='gas':
                    lhs = model.varDeviceFlow[dev,'gas','in',t]
                    #rhs = model.paramDevice[dev]['f_inj']*Q_reservoir
                    rhs = model.paramDevice[dev]['f_inj']* model.varDeviceFlow[dev,'oil','out',t]
                    return (lhs==rhs)
                else:
                    return pyo.Constraint.Skip
            elif i==4:
                # gas injection pressure is fixed
                if carrier=='gas':
                    lhs = model.varPressure[(node,carrier,'in',t)]
                    rhs = model.paramDevice[dev]['injectionpressure']
                    return (lhs==rhs)
                else:
                    return pyo.Constraint.Skip
        model.constrDevice_well_gaslift = pyo.Constraint(model.setDevice,
                  model.setFlowComponent,model.setHorizon,pyo.RangeSet(1,4),
                  rule=rule_devmodel_well_gaslift)



        #TODO: separator equations
        logging.info("TODO: separator power (eta) and heat (eta2) demand")
        def rule_devmodel_separator(model,dev,t,i):
            if model.paramDevice[dev]['model'] != 'separator':
                return pyo.Constraint.Skip
            wellstream_prop=model.paramCarriers['wellstream']
            GOR = wellstream_prop['gas_oil_ratio']
            WC = wellstream_prop['water_cut']
            comp_oil = (1-WC)/(1+GOR-GOR*WC)
            comp_water = WC/(1+GOR-GOR*WC)
            comp_gas = GOR*(1-WC)/(1+GOR*(1-WC))
            flow_in = model.varDeviceFlow[dev,'wellstream','in',t]
            if i==1:
                lhs = model.varDeviceFlow[dev,'gas','out',t]
                rhs = flow_in*comp_gas
                return lhs==rhs
            elif i==2:
                lhs = model.varDeviceFlow[dev,'oil','out',t]
                rhs = flow_in*comp_oil
                return lhs==rhs
            elif i==3:
                #return pyo.Constraint.Skip
                lhs = model.varDeviceFlow[dev,'water','out',t]
                rhs = flow_in*comp_water
                return lhs==rhs
            elif i==4:
                # electricity demand
                lhs = model.varDeviceFlow[dev,'el','in',t]
                rhs = flow_in*model.paramDevice[dev]['eta_el']
                return lhs==rhs
            elif i==5:
                lhs = model.varDeviceFlow[dev,'heat','in',t]
                rhs = flow_in*model.paramDevice[dev]['eta_heat']
                return lhs==rhs
            elif i==6:
                '''gas pressure out = nominal'''
                node = model.paramDevice[dev]['node']
                lhs = model.varPressure[(node,'gas','out',t)]
                rhs = model.paramNode[node]['pressure.gas.out']
                return lhs==rhs
            elif i==7:
                '''oil pressure out = nominal'''
                node = model.paramDevice[dev]['node']
                lhs = model.varPressure[(node,'oil','out',t)]
                rhs = model.paramNode[node]['pressure.oil.out']
                return lhs==rhs
            elif i==8:
                '''water pressure out = nominal'''
                node = model.paramDevice[dev]['node']
                lhs = model.varPressure[(node,'water','out',t)]
                rhs = model.paramNode[node]['pressure.water.out']
                return lhs==rhs

        model.constrDevice_separator = pyo.Constraint(model.setDevice,
                  model.setHorizon,pyo.RangeSet(1,8),
                  rule=rule_devmodel_separator)

        #Alternative separator model - using oil/gas/water input instead of
        # wellstream
        def rule_devmodel_separator2(model,dev,fc,t,i):
            if model.paramDevice[dev]['model'] != 'separator2':
                return pyo.Constraint.Skip
            #composition = model.paramCarriers['wellstream']['composition']
            wellstream_prop=model.paramCarriers['wellstream']
            flow_in = sum(model.varDeviceFlow[dev,f,'in',t]
                            for f in model.setFlowComponent)
            if i==1:
                # component flow in = flow out
                lhs = model.varDeviceFlow[dev,fc,'out',t]
                rhs =  model.varDeviceFlow[dev,fc,'in',t]
                return (lhs==rhs)
            elif i==2:
                # pressure out is nominal
                node = model.paramDevice[dev]['node']
                lhs = model.varPressure[(node,fc,'out',t)]
                rhs = model.paramNode[node]['pressure.{}.out'.format(fc)]
                return (lhs==rhs)
            elif i==3:
                # electricity demand
                if fc != model.setFlowComponent.first():
                    return pyo.Constraint.Skip
                lhs = model.varDeviceFlow[dev,'el','in',t]
                rhs = flow_in*model.paramDevice[dev]['eta_el']
                return (lhs==rhs)
            elif i==4:
                # heat demand
                if fc != model.setFlowComponent.first():
                    return pyo.Constraint.Skip
                lhs = model.varDeviceFlow[dev,'heat','in',t]
                rhs = flow_in*model.paramDevice[dev]['eta_heat']
                return (lhs==rhs)

        model.constrDevice_separator2 = pyo.Constraint(model.setDevice,
                  model.setFlowComponent,model.setHorizon,pyo.RangeSet(1,4),
                  rule=rule_devmodel_separator2)


        def rule_devmodel_compressor_el(model,dev,t,i):
            if model.paramDevice[dev]['model'] != 'compressor_el':
                return pyo.Constraint.Skip
            if i==1:
                '''gas flow in equals gas flow out (mass flow)'''
                lhs = model.varDeviceFlow[dev,'gas','in',t]
                rhs = model.varDeviceFlow[dev,'gas','out',t]
                return (lhs==rhs)
            elif i==2:
                '''Device el demand'''
                lhs = model.varDeviceFlow[dev,'el','in',t]
                rhs = self.compute_compressor_demand(model,dev,linear=True,t=t)
                return (lhs==rhs)
        model.constrDevice_compressor_el = pyo.Constraint(model.setDevice,
                  model.setHorizon,pyo.RangeSet(1,2),
                  rule=rule_devmodel_compressor_el)

        def rule_devmodel_compressor_gas(model,dev,t,i):
            if model.paramDevice[dev]['model'] != 'compressor_gas':
                return pyo.Constraint.Skip
            # device power is defined as gas demand in MW (rather than Sm3)
            powerdemand = self.compute_compressor_demand(
                model,dev,linear=True,t=t)
            if i==1:
                # matter conservation
                gas_energy_content=model.paramCarriers['gas']['energy_value'] #MJ/Sm3
                lhs = model.varDeviceFlow[dev,'gas','out',t]
                rhs = (model.varDeviceFlow[dev,'gas','in',t]
                        - powerdemand/gas_energy_content)
                return lhs==rhs
        model.constrDevice_compressor_gas = pyo.Constraint(model.setDevice,
                  model.setHorizon,pyo.RangeSet(1,1),
                  rule=rule_devmodel_compressor_gas)


        def rule_devmodel_gasheater(model,dev,t,i):
            if model.paramDevice[dev]['model'] != 'gasheater':
                return pyo.Constraint.Skip
            if i==1:
                # heat out = gas input * energy content * efficiency
                gas_energy_content=model.paramCarriers['gas']['energy_value'] #MJ/Sm3
                lhs = model.varDeviceFlow[dev,'heat','out',t]
                rhs = (model.varDeviceFlow[dev,'gas','in',t]
                        * gas_energy_content * model.paramDevice[dev]['eta'])
                return (lhs==rhs)
        model.constrDevice_gasheater = pyo.Constraint(model.setDevice,
                  model.setHorizon,pyo.RangeSet(1,1),
                  rule=rule_devmodel_gasheater)

        def rule_devmodel_heatpump(model,dev,t,i):
            if model.paramDevice[dev]['model'] != 'heatpump':
                return pyo.Constraint.Skip
            if i==1:
                # heat out = el in * efficiency
                lhs = model.varDeviceFlow[dev,'heat','out',t]
                rhs = (model.varDeviceFlow[dev,'el','in',t]
                        *model.paramDevice[dev]['eta'])
                return (lhs==rhs)
        model.constrDevice_heatpump = pyo.Constraint(model.setDevice,
                  model.setHorizon,pyo.RangeSet(1,1),
                  rule=rule_devmodel_heatpump)


        logging.info("TODO: gas turbine power vs heat output")
        logging.info("TODO: startup cost")
        def rule_devmodel_gasturbine(model,dev,t,i):
            if model.paramDevice[dev]['model'] != 'gasturbine':
                return pyo.Constraint.Skip
            elpower = model.varDeviceFlow[dev,'el','out',t]
            gas_energy_content=model.paramCarriers['gas']['energy_value'] #MJ/Sm3
            if i==1:
                '''turbine el power out vs gas fuel in'''
                # fuel consumption (gas in) is a linear function of el power output
                # fuel = B + A*power
                # => efficiency = power/(A+B*power)
                A = model.paramDevice[dev]['fuelA']
                B = model.paramDevice[dev]['fuelB']
                Pmax = model.paramDevice[dev]['Pmax']
                lhs = model.varDeviceFlow[dev,'gas','in',t]*gas_energy_content/Pmax
                rhs = (B*(model.varDeviceIsOn[dev,t]
                            +model.varDeviceIsPrep[dev,t])
                        + A*model.varDeviceFlow[dev,'el','out',t]/Pmax)
                return lhs==rhs
            elif i==2:
                '''heat output = (gas energy in - el power out)* heat efficiency'''
                lhs = model.varDeviceFlow[dev,'heat','out',t]
                rhs = (model.varDeviceFlow[dev,'gas','in',t]*gas_energy_content
                        -model.varDeviceFlow[dev,'el','out',t]
                       )*model.paramDevice[dev]['eta_heat']
                return lhs==rhs
        model.constrDevice_gasturbine = pyo.Constraint(model.setDevice,
                  model.setHorizon,pyo.RangeSet(1,2),
                  rule=rule_devmodel_gasturbine)

        def rule_devmodel_source(model,dev,t,i,carrier):
            if model.paramDevice[dev]['model'] != 'source_{}'.format(carrier):
                return pyo.Constraint.Skip
            if i==1:
                node = model.paramDevice[dev]['node']
                lhs = model.varPressure[(node,carrier,'out',t)]
                rhs = model.paramDevice[dev]['naturalpressure']
                return (lhs==rhs)

        def rule_devmodel_source_gas(model,dev,t,i):
            return rule_devmodel_source(model,dev,t,i,'gas')
        model.constrDevice_source_gas = pyo.Constraint(model.setDevice,
                  model.setHorizon,pyo.RangeSet(1,1),
                  rule=rule_devmodel_source_gas)

        def rule_devmodel_source_oil(model,dev,t,i):
            return rule_devmodel_source(model,dev,t,i,'oil')
        model.constrDevice_source_oil = pyo.Constraint(model.setDevice,
                  model.setHorizon,pyo.RangeSet(1,1),
                  rule=rule_devmodel_source_oil)

        def rule_devmodel_source_water(model,dev,t,i):
            return rule_devmodel_source(model,dev,t,i,'water')
        model.constrDevice_source_water = pyo.Constraint(model.setDevice,
                  model.setHorizon,pyo.RangeSet(1,1),
                  rule=rule_devmodel_source_water)


        logging.info("TODO: el source: dieselgen, fuel, on-off variables")
        #TODO: diesel gen fuel, onoff variables..
        def rule_devmodel_source_el(model,dev,t):
            if model.paramDevice[dev]['model'] != 'source_el':
                return pyo.Constraint.Skip
            expr = pyo.Constraint.Skip
            return expr
        model.constrDevice_source_el = pyo.Constraint(model.setDevice,
                  model.setHorizon,rule=rule_devmodel_source_el)


        def rule_devmodel_storage_el(model,dev,t,i):
            if model.paramDevice[dev]['model'] != 'storage_el':
                return pyo.Constraint.Skip
            if i==1:
                #energy balance
                # (el_in*eta - el_out/eta)*dt = delta storage
                # eta = efficiency charging and discharging
                delta_t = model.paramParameters['time_delta_minutes']/60 #hours
                lhs = (model.varDeviceFlow[dev,'el','in',t]
                        *model.paramDevice[dev]['eta']
                       -model.varDeviceFlow[dev,'el','out',t]
                        /model.paramDevice[dev]['eta'] )*delta_t
                if t>0:
                    Eprev = model.varDeviceStorageEnergy[dev,t-1]
                else:
                    Eprev = model.paramDeviceEnergyInitially[dev]
                rhs = (model.varDeviceStorageEnergy[dev,t]-Eprev)
                return (lhs==rhs)
            elif i==2:
                # energy storage limit
                ub = model.paramDevice[dev]['Emax']
                lb = 0
                # Specifying Emin may be useful e.g. to impose that battery
                # should only be used for reserve
                # BUT - probably better to specify an energy depletion cost
                # instead (Ecost) - to allow using battery (at a cost) if required
                if 'Emin' in model.paramDevice[dev]:
                    lb = model.paramDevice[dev]['Emin']
                return pyo.inequality(lb,model.varDeviceStorageEnergy[dev,t],ub)
            elif i==3:
                #return pyo.Constraint.Skip # unnecessary -> generic Pmax/min constraints
                #charging power limit
                ub = model.paramDevice[dev]['Pmax']
                return (model.varDeviceFlow[dev,'el','out',t]<=ub)
            elif i==4:
                #return pyo.Constraint.Skip # unnecessary -> generic Pmax/min constraints
                #discharging power limit
                #ub = model.paramDevice[dev]['Pmax']
                ub = -model.paramDevice[dev]['Pmin'] # <- see generic Pmax/min constr
                return (model.varDeviceFlow[dev,'el','in',t]<=ub)
            elif i==5:
                # Constraint 5-8: varDeviceStoragePmax = min{Pmax,E/dt}
                # ref: https://or.stackexchange.com/a/1174
                lhs = model.varDeviceStoragePmax[dev,t]
                rhs = model.paramDevice[dev]['Pmax']
                return (lhs<=rhs)
            elif i==6:
                lhs = model.varDeviceStoragePmax[dev,t]
                # Parameter specifying for how long the power needs to be
                # sustained to count as reserve (e.g. similar to GT startup time)
                dt_hours = model.paramParameters['time_reserve_minutes']/60
                rhs = model.varDeviceStorageEnergy[dev,t]/dt_hours
                return (lhs<=rhs)
            elif i==7:
                bigM = 10*model.paramDevice[dev]['Pmax']
                lhs = model.varDeviceStoragePmax[dev,t]
                rhs = model.paramDevice[dev]['Pmax'] - bigM*(1-model.varStorY[dev,t])
                return (lhs>=rhs)
            elif i==8:
                dt_hours = model.paramParameters['time_reserve_minutes']/60
                bigM = 10*model.paramDevice[dev]['Emax']/dt_hours
                lhs = model.varDeviceStoragePmax[dev,t]
                rhs = (model.varDeviceStorageEnergy[dev,t]/dt_hours
                        - bigM*model.varStorY[dev,t])
                return (lhs>=rhs)
            elif i==9:
                # constraint on storage end vs start
                # Adding this does not seem to improve result (not lower CO2)
                if (('E_end' in model.paramDevice[dev])
                        and (t==model.setHorizon[-1])):
                    lhs = model.varDeviceStorageEnergy[dev,t]
                    #rhs = model.varDeviceStorageEnergy[dev,0] # end=start
                    rhs = model.paramDevice[dev]['E_end']
                    return (lhs==rhs)
                else:
                    return pyo.Constraint.Skip
        model.constrDevice_storage_el = pyo.Constraint(model.setDevice,
                  model.setHorizon,pyo.RangeSet(1,9),
                  rule=rule_devmodel_storage_el)


        logging.info("TODO: liquid pump approximation ok?")
        def rule_devmodel_pump(model,dev,t,i,carrier):
            if model.paramDevice[dev]['model'] != 'pump_{}'.format(carrier):
                return pyo.Constraint.Skip
            if i==1:
                # flow out = flow in
                lhs = model.varDeviceFlow[dev,carrier,'out',t]
                rhs = model.varDeviceFlow[dev,carrier,'in',t]
                return lhs==rhs
            elif i==2:
                # # power demand vs flow rate and pressure difference
                # # see eg. doi:10.1016/S0262-1762(07)70434-0
                # # P = Q*(p_out-p_in)/eta
                # # units: m3/s*MPa = MW
                # #
                # # Assuming incompressible fluid, so flow rate m3/s=Sm3/s
                # # (this approximation may not be very good for multiphase
                # # wellstream)
                # node = model.paramDevice[dev]['node']
                # eta = model.paramDevice[dev]['eta']
                # flowrate = model.varDeviceFlow[dev,carrier,'in',t]
                # # assume nominal pressure and keep only flow rate dependence
                # #TODO: Better linearisation?
                # p_in = model.paramNode[node]['pressure.{}.in'.format(carrier)]
                # p_out = model.paramNode[node]['pressure.{}.out'.format(carrier)]
                # delta_p = p_out - p_in
                # if self._quadraticConstraints:
                #     # Quadratic constraint...
                #     delta_p = (model.varPressure[(node,carrier,'out',t)]
                #                 -model.varPressure[(node,carrier,'in',t)])
                lhs = model.varDeviceFlow[dev,'el','in',t]
                # rhs = flowrate*delta_p/eta
                '''Device el demand'''
                rhs = self.compute_pump_demand(model,dev,linear=True,t=t,
                    carrier=carrier)
                return (lhs==rhs)

        def rule_devmodel_pump_oil(model,dev,t,i):
            return rule_devmodel_pump(model,dev,t,i,carrier='oil')
        model.constrDevice_pump_oil = pyo.Constraint(model.setDevice,
                  model.setHorizon,pyo.RangeSet(1,2),
                  rule=rule_devmodel_pump_oil)

        def rule_devmodel_pump_water(model,dev,t,i):
            return rule_devmodel_pump(model,dev,t,i,carrier='water')
        model.constrDevice_pump_water = pyo.Constraint(model.setDevice,
                  model.setHorizon,pyo.RangeSet(1,2),
                  rule=rule_devmodel_pump_water)

        def rule_devmodel_pump_wellstream(model,dev,t,i):
            return rule_devmodel_pump(model,dev,t,i,carrier='wellstream')
        model.constrDevice_pump_wellstream = pyo.Constraint(model.setDevice,
                  model.setHorizon,pyo.RangeSet(1,2),
                  rule=rule_devmodel_pump_wellstream)


        def rule_devmodel_sink_gas(model,dev,t):
            if model.paramDevice[dev]['model'] != 'sink_gas':
                return pyo.Constraint.Skip
            expr = pyo.Constraint.Skip
            return expr
        model.constrDevice_sink_gas = pyo.Constraint(model.setDevice,
                  model.setHorizon,
                  rule=rule_devmodel_sink_gas)

        def rule_devmodel_sink_oil(model,dev,t):
            if model.paramDevice[dev]['model'] != 'sink_oil':
                return pyo.Constraint.Skip
            expr = pyo.Constraint.Skip
            return expr
        model.constrDevice_sink_oil = pyo.Constraint(model.setDevice,
                  model.setHorizon,
                  rule=rule_devmodel_sink_oil)

        def rule_devmodel_sink_el(model,dev,t):
            if model.paramDevice[dev]['model'] != 'sink_el':
                return pyo.Constraint.Skip
            expr = pyo.Constraint.Skip
            return expr
        model.constrDevice_sink_el = pyo.Constraint(model.setDevice,
                  model.setHorizon,rule=rule_devmodel_sink_el)

        def rule_devmodel_sink_heat(model,dev,t):
            if model.paramDevice[dev]['model'] != 'sink_heat':
                return pyo.Constraint.Skip
            expr = pyo.Constraint.Skip
            return expr
        model.constrDevice_sink_heat = pyo.Constraint(model.setDevice,
                  model.setHorizon,rule=rule_devmodel_sink_heat)

        def rule_devmodel_sink_water(model,dev,t,i):
            if model.paramDevice[dev]['model'] != 'sink_water':
                return pyo.Constraint.Skip

            if ('Qavg' not in model.paramDevice[dev]):
                    return pyo.Constraint.Skip
            if ('Vmax' not in model.paramDevice[dev]):
                    return pyo.Constraint.Skip
            if model.paramDevice[dev]['Vmax']==0:
                    return pyo.Constraint.Skip
            if i==1:
                # FLEXIBILITY
                # (water_in-water_avg)*dt = delta buffer
                delta_t = model.paramParameters['time_delta_minutes']/60 #hours
                lhs = (model.varDeviceFlow[dev,'water','in',t]
                       -model.paramDevice[dev]['Qavg'])*delta_t
                if t>0:
                    Eprev = model.varDeviceStorageEnergy[dev,t-1]
                else:
                    Eprev = model.paramDeviceEnergyInitially[dev]
                rhs = (model.varDeviceStorageEnergy[dev,t]-Eprev)
                return (lhs==rhs)
            elif i==2:
                # energy buffer limit
                Emax = model.paramDevice[dev]['Vmax']
                return pyo.inequality(
                        -Emax/2,model.varDeviceStorageEnergy[dev,t],Emax/2)
            return expr
        model.constrDevice_sink_water = pyo.Constraint(model.setDevice,
            model.setHorizon,pyo.RangeSet(1,2),
            rule=rule_devmodel_sink_water)


        def rule_elReserveMargin(model,t):
            '''Reserve margin constraint (electrical supply)
            Not used capacity by power suppliers/storage/load flexibility
            must be larger than some specified margin
            (to cope with unforeseen variations)
            '''
            if (('elReserveMargin' not in model.paramParameters)
                    or (model.paramParameters['elReserveMargin']<0)):
                if ( t==model.setHorizon.first()):
                    logging.info("Valid elReserveMargin not defined -> no constraint")
                return pyo.Constraint.Skip
            # exclude constraint for first timesteps since the point of the
            # dispatch margin is exactly to cope with forecast errors
            # *2 to make sure there is time to start up gt
            if t < model.paramParameters['forecast_timesteps']:
                return pyo.Constraint.Skip

            margin = model.paramParameters['elReserveMargin']
            capacity_unused = self.compute_elReserve(model,t)
            expr = (capacity_unused >= margin)
            return expr
        model.constrDevice_elReserveMargin = pyo.Constraint(
                  model.setHorizon,rule=rule_elReserveMargin)

        def rule_elBackupMargin(model,dev,t):
            '''Backup capacity constraint (electrical supply)
            Not used capacity by other online power suppliers plus sheddable
            load must be larger than power output of this device
            (to take over in case of a fault)
            '''
            if (('elBackupMargin' not in model.paramParameters)
                    or (model.paramParameters['elBackupMargin']<0)):
                if ((dev==model.setDevice.first())
                    &( t==model.setHorizon.first())):
                    logging.info("Valid elBackupMargin not defined -> no constraint")
                return pyo.Constraint.Skip
            margin = model.paramParameters['elBackupMargin']
            devmodel = model.paramDevice[dev]['model']
            if 'el' not in Multicarrier.devicemodel_inout()[devmodel]['out']:
                # this is not a power generator
                return pyo.Constraint.Skip
            res_otherdevs = self.compute_elBackup(model,t,exclude_device=dev)
            # elBackupMargin is zero or positive (if loss of load is acceptable)
            expr = (res_otherdevs -model.varDeviceFlow[dev,'el','out',t]
                    >= -margin)
            return expr
        model.constrDevice_elBackupMargin = pyo.Constraint(model.setDevice,
                  model.setHorizon,rule=rule_elBackupMargin)



        def rule_startup_shutdown(model,dev,t):
            '''startup/shutdown constraint
            connecting starting, stopping, preparation, online stages of GTs'''

            Tdelay=0
            if ('startupDelay' in model.paramDevice[dev]):
                Tdelay_min = model.paramDevice[dev]['startupDelay']
                # Delay in timesteps, rounding down.
                # example: time_delta = 5 min, startupDelay= 8 min => Tdelay=1
                Tdelay = int(Tdelay_min/model.paramParameters['time_delta_minutes'])
            prevPart=0
            if (t >= Tdelay):
            	#prevPart = sum( model.varDeviceStarting[dev,t-tau]
                #    for tau in range(0,Tdelay) )
               prevPart = model.varDeviceStarting[dev,t-Tdelay]
            else:
                # OBS: for this to work as intended, need to reconstruct constraint
                # pyo.value(...) not needed
                #prepInit = pyo.value(model.paramDevicePrepTimestepsInitially[dev])
                prepInit = model.paramDevicePrepTimestepsInitially[dev]
                if (prepInit + t == Tdelay):
                    prevPart = 1
            if (t>0):
                ison_prev = model.varDeviceIsOn[dev,t-1]
            else:
                ison_prev = model.paramDeviceIsOnInitially[dev]
            lhs = model.varDeviceIsOn[dev,t] - ison_prev
            rhs = prevPart - model.varDeviceStopping[dev,t]
            return (lhs==rhs)
        model.constrDevice_startup_shutdown = pyo.Constraint(model.setDevice,
                  model.setHorizon,
                  rule=rule_startup_shutdown)

        def rule_startup_delay(model,dev,t):
            '''startup delay/preparation for GTs'''
            Tdelay=0
            if ('startupDelay' in model.paramDevice[dev]):
                Tdelay_min = model.paramDevice[dev]['startupDelay']
                # Delay in timesteps, rounding down.
                # example: time_delta = 5 min, startupDelay= 8 min => Tdelay=1
                Tdelay = int(Tdelay_min/model.paramParameters['time_delta_minutes'])
            else:
                return pyo.Constraint.Skip
            # determine if was in preparation previously
            # dependend on value - so must reconstruct constraint each time
            stepsPrevPrep = pyo.value(
                model.paramDevicePrepTimestepsInitially[dev])
            if (stepsPrevPrep>0):
                prevIsPrep = 1
            else:
                prevIsPrep = 0

            prevPart=0
            if (t < Tdelay-stepsPrevPrep):
            	prevPart = prevIsPrep
            tau_range = range(0,min(t+1,Tdelay))
            lhs = model.varDeviceIsPrep[dev,t]
            rhs = sum(model.varDeviceStarting[dev,t-tau]
                        for tau in tau_range) + prevPart
#            if (t==0) & (dev=="GT2"):
#                print("rule startup delay (t=0)")
#                print("prev steps=",stepsPrevPrep,"prevIsPrep=",prevIsPrep,
#                    'prevpart=',prevPart,'Tdelay=',Tdelay)
#                print("lhs=",lhs,"rhs=",rhs)
            return (lhs==rhs)
        model.constrDevice_startup_delay = pyo.Constraint(model.setDevice,
                  model.setHorizon,rule=rule_startup_delay)


        def rule_ramprate(model,dev,t):
            '''power ramp rate limit'''

            # If no ramp limits have been specified, skip constraint
            if (('maxRampUp' not in model.paramDevice[dev])
                or pd.isna(model.paramDevice[dev]['maxRampUp'])):
                return pyo.Constraint.Skip
            if (t>0):
                p_prev = self.getDevicePower(model,dev,t-1)
            else:
                p_prev = model.paramDevicePowerInitially[dev]
            p_this = self.getDevicePower(model,dev,t)
            deltaP = (self.getDevicePower(model,dev,t) - p_prev)
            delta_t = model.paramParameters['time_delta_minutes']
            maxP = model.paramDevice[dev]['Pmax']
            max_neg = -model.paramDevice[dev]['maxRampDown']*maxP*delta_t
            max_pos = model.paramDevice[dev]['maxRampUp']*maxP*delta_t
            return pyo.inequality(max_neg, deltaP, max_pos)
        model.constrDevice_ramprate = pyo.Constraint(model.setDevice,
                 model.setHorizon, rule=rule_ramprate)

        # Not so easy to formulate equal load sharing as a linear equation
        #
        # def rule_gasturbine_loadsharing(model,dev,t):
        #     '''ensure load is shared evently between online gas turbines'''
        #     if model.paramDevice[dev]['model'] != 'gasturbine':
        #         return pyo.Constraint.Skip
        #     for otherdev in model.setDevice:
        #         if (model.paramDevice[otherdev]['model'] == 'gasturbine'):
        #             rhs =
        #
        #     lhs = model.varDeviceFlow[dev,t,'el','out']
        #





        def rule_terminalEnergyBalance(model,carrier,node,terminal,t):
            ''' node energy balance (at in and out terminals)
            "in" terminal: flow into terminal is positive (Pinj>0)
            "out" terminal: flow out of terminal is positive (Pinj>0)

            distinguishing between the case (1) where node/carrier has a
            single terminal or (2) with an "in" and one "out" terminal
            (with device in series) that may have different properties
            (such as gas pressure)

            edge1 \                                     / edge3
                   \      devFlow_in    devFlow_out    /
                    [term in]-->--device-->--[term out]
                   /      \                       /    \
            edge2 /        \......termFlow......./      \ edge4


            '''

            # Pinj = power injected into in terminal /out of out terminal
            Pinj = 0
            # devices:
            if (node in model.paramNodeDevices):
                for dev in model.paramNodeDevices[node]:
                    # Power into terminal:
                    dev_model = model.paramDevice[dev]['model']
                    if dev_model in self._devmodels:#model.paramDevicemodel:
                        #print("carrier:{},node:{},terminal:{},model:{}"
                        #      .format(carrier,node,terminal,dev_model))
                        if carrier in self._devmodels[dev_model][terminal]:
                            Pinj -= model.varDeviceFlow[dev,carrier,terminal,t]
                    else:
                        raise Exception("Undefined device model ({})".format(dev_model))

            # connect terminals (i.e. treat as one):
            if not model.paramNodeCarrierHasSerialDevice[node][carrier]:
                Pinj -= model.varTerminalFlow[node,carrier,t]

            # edges:
            if (carrier,node) in model.paramNodeEdgesTo and (terminal=='in'):
                for edg in model.paramNodeEdgesTo[(carrier,node)]:
                    # power into node from edge
                    Pinj += (model.varEdgeFlow[edg,t])
            elif (carrier,node) in model.paramNodeEdgesFrom and (terminal=='out'):
                for edg in model.paramNodeEdgesFrom[(carrier,node)]:
                    # power out of node into edge
                    Pinj += (model.varEdgeFlow[edg,t])


            expr = (Pinj==0)
            if ((type(expr) is bool) and (expr==True)):
                expr = pyo.Constraint.Skip
            return expr
        model.constrTerminalEnergyBalance = pyo.Constraint(model.setCarrier,
                      model.setNode, model.setTerminal,model.setHorizon,
                      rule=rule_terminalEnergyBalance)

        # # additional terminal energy/mass balance for mixed flow (wellstream)
        # def rule_terminalEnergyBalanceWithFlowComponents(
        #         model,node,terminal,fc,t):
        #     carrier="wellstream"
        #     Pinj = 0
        #     # devices:
        #     if (node in model.paramNodeDevices):
        #         for dev in model.paramNodeDevices[node]:
        #             # Power into terminal:
        #             dev_model = model.paramDevice[dev]['model']
        #             if carrier in self._devmodels[dev_model][terminal]:
        #                 GOR = model.paramDevice[dev]['gas_oil_ratio']
        #                 WC = model.paramDevice[dev]['water_cut']
        #                 comp_oil = (1-WC)/(1+GOR-GOR*WC)
        #                 comp_water = WC/(1+GOR-GOR*WC)
        #                 comp_gas = GOR*(1-WC)/(1+GOR*(1-WC))
        #                 comp={'oil':comp_oil,'gas':comp_gas,'water':comp_water}
        #                 Pinj -= comp[fc]*model.varDeviceFlow[dev,carrier,terminal,t]
        #
        #     # connect terminals (i.e. treat as one):
        #     if not model.paramNodeCarrierHasSerialDevice[node][carrier]:
        #         Pinj -= model.varTerminalFlowComponent[node,fc,t]
        #
        #     # edges:
        #     if (carrier,node) in model.paramNodeEdgesTo and (terminal=='in'):
        #         for edg in model.paramNodeEdgesTo[(carrier,node)]:
        #             # power into node from edge
        #             Pinj += (model.varEdgeFlowComponent[edg,fc,t])
        #     elif (carrier,node) in model.paramNodeEdgesFrom and (terminal=='out'):
        #         for edg in model.paramNodeEdgesFrom[(carrier,node)]:
        #             # power out of node into edge
        #             Pinj += (model.varEdgeFlowComponent[edg,fc,t])
        #
        #     expr = (Pinj==0)
        #     if ((type(expr) is bool) and (expr==True)):
        #         expr = pyo.Constraint.Skip
        #     return expr
        # model.constrTerminalEnergyBalanceWithFlowComponents = pyo.Constraint(
        #     model.setNode, model.setTerminal,model.setFlowComponent,
        #     model.setHorizon, rule=rule_terminalEnergyBalanceWithFlowComponents)


#        logging.info("TODO: el power balance constraint redundant?")
#        def rule_terminalElPowerBalance(model,node,t):
#            ''' electric power balance at in and out terminals
#            '''
#            # Pinj = power injected into terminal
#            Pinj = 0
#            rhs = 0
#            carrier='el'
#            # devices:
#            if (node in model.paramNodeDevices):
#                for dev in model.paramNodeDevices[node]:
#                    # Power into terminal:
#                    dev_model = model.paramDevice[dev]['model']
#                    if dev_model in self._devmodels:#model.paramDevicemodel:
#                        if carrier in self._devmodels[dev_model]['in']:
#                            Pinj -= model.varDeviceFlow[dev,carrier,'in',t]
#                        if carrier in self._devmodels[dev_model]['out']:
#                            Pinj += model.varDeviceFlow[dev,carrier,'out',t]
#                    else:
#                        raise Exception("Undefined device model ({})".format(dev_model))
#
#            # edges:
#            # El linearised power flow equations:
#            # Pinj = B theta
#            n2s = [k[1]  for k in model.paramCoeffB.keys() if k[0]==node]
#            for n2 in n2s:
#                rhs -= model.paramCoeffB[node,n2]*(
#                        model.varElVoltageAngle[n2,t]*self.elbase['baseAngle'])
#            rhs = rhs*self.elbase['baseMVA']
#
#            expr = (Pinj==rhs)
#            if ((type(expr) is bool) and (expr==True)):
#                expr = pyo.Constraint.Skip
#            return expr
#        model.constrElPowerBalance = pyo.Constraint(model.setNode,
#                    model.setHorizon, rule=rule_terminalElPowerBalance)


        def rule_elVoltageReference(model,t):
            n = model.paramParameters['reference_node']
            expr = (model.varElVoltageAngle[n,t] == 0)
            return expr
        model.constrElVoltageReference = pyo.Constraint(model.setHorizon,
                                              rule=rule_elVoltageReference)



        logging.info("TODO: flow vs pressure equations for liquid flows")
        def rule_edgeFlowEquations(model,edge,t):
            '''Flow as a function of node values (voltage/pressure)'''
            carrier = model.paramEdge[edge]['type']
            n_from = model.paramEdge[edge]['nodeFrom']
            n_to = model.paramEdge[edge]['nodeTo']

            if carrier == 'el':
                '''power flow vs voltage angle difference
                Linearised power flow equations (DC power flow)'''

                lhs = model.varEdgeFlow[edge,t]
                lhs = lhs/self.elbase['baseMVA']
                rhs = 0
                #TODO speed up creatioin of constraints - remove for loop
                n2s = [k[1]  for k in model.paramCoeffDA.keys() if k[0]==edge]
                for n2 in n2s:
                    rhs += model.paramCoeffDA[edge,n2]*(
                            model.varElVoltageAngle[n2,t]*self.elbase['baseAngle'])
                return (lhs==rhs)

            elif carrier in ['gas','wellstream','oil','water']:
                p1 = model.varPressure[(n_from,carrier,'out',t)]
                p2 = model.varPressure[(n_to,carrier,'in',t)]
                Q = model.varEdgeFlow[edge,t]
                if 'num_pipes' in model.paramEdge[edge]:
                    num_pipes = model.paramEdge[edge]['num_pipes']
                    logging.debug("{},{}: {} parallel pipes"
                        .format(edge,t,num_pipes))
                    Q = Q/num_pipes
                p2_computed = self.compute_edge_pressuredrop(
                    model,edge,p1=p1,Q=Q,linear=True)
                return (p2==p2_computed)
#            elif carrier in ['wellstream','water']:
#                p_from = model.varPressure[(n_from,carrier,'out',t)]
#                p_to = model.varPressure[(n_to,carrier,'in',t)]
#                return (p_from==p_to)
            else:
                #Other types of edges - no constraints other than max/min flow
                return pyo.Constraint.Skip
        model.constrEdgeFlowEquations = pyo.Constraint(
                model.setEdge,model.setHorizon,rule=rule_edgeFlowEquations)

        def rule_pressureAtNode(model,node,carrier,t):
            if carrier in ['el','heat']:
                return pyo.Constraint.Skip
            elif model.paramNodeCarrierHasSerialDevice[node][carrier]:
                # pressure in and out are related via device equations for
                # device connected between in and out terminals. So no
                # extra constraint required
                return pyo.Constraint.Skip
            else:
                # single terminal. (pressure out=pressure in)
                expr = (model.varPressure[(node,carrier,'out',t)]
                        == model.varPressure[(node,carrier,'in',t)] )
                return expr
        model.constrPressureAtNode = pyo.Constraint(
                model.setNode,model.setCarrier,model.setHorizon,
                rule=rule_pressureAtNode)

        def rule_pressureBounds(model,node,term,carrier,t):
            col = 'pressure.{}.{}'.format(carrier,term)
            if not col in model.paramNode[node]:
                # no pressure data relevant for this node/carrier
                return pyo.Constraint.Skip
            nom_p = model.paramNode[node][col]
            if pd.isna(nom_p):
                # no pressure data specified for this node/carrier
                return pyo.Constraint.Skip
            cc = 'maxdeviation_pressure.{}.{}'.format(carrier,term)
            if ((cc in model.paramNode[node]) and
                (not np.isnan(model.paramNode[node][cc]))):
                maxdev = model.paramNode[node][cc]
                if t==0:
                    logging.debug("Using ind. pressure limit for: {}, {}, {}"
                        .format(node,cc, maxdev))
            else:
                maxdev = model.paramParameters['max_pressure_deviation']
                if maxdev==-1:
                    return pyo.Constraint.Skip
            lb = nom_p*(1 - maxdev)
            ub = nom_p*(1 + maxdev)
            return (lb,model.varPressure[(node,carrier,term,t)],ub)
        model.constrPressureBounds = pyo.Constraint(
                model.setNode,model.setTerminal,model.setCarrier,
                model.setHorizon,rule=rule_pressureBounds)


        def rule_devicePmax(model,dev,t):
            # max/min power (zero if device is not on)
            if 'Pmax' not in model.paramDevice[dev]:
                return pyo.Constraint.Skip
            maxValue = model.paramDevice[dev]['Pmax']
            if 'profile' in model.paramDevice[dev]:
                # use an availability profile if provided
                extprofile = model.paramDevice[dev]['profile']
                maxValue = maxValue*model.paramProfiles[extprofile,t]
            ison = 1
            if model.paramDevice[dev]['model'] in ['gasturbine']:
                ison = model.varDeviceIsOn[dev,t]
            power = self.getDevicePower(model,dev,t)
            expr = ( power <= ison*maxValue)
            return expr
        model.constrDevicePmax = pyo.Constraint(
                model.setDevice,model.setHorizon,rule=rule_devicePmax)

        def rule_devicePmin(model,dev,t):
            if 'Pmin' not in model.paramDevice[dev]:
                return pyo.Constraint.Skip
            minValue = model.paramDevice[dev]['Pmin']
            if 'profile' in model.paramDevice[dev]:
                # use an availability profile if provided
                extprofile = model.paramDevice[dev]['profile']
                minValue = minValue*model.paramProfiles[extprofile,t]
            ison = 1
            if model.paramDevice[dev]['model'] in ['gasturbine']:
                ison = model.varDeviceIsOn[dev,t]
            power = self.getDevicePower(model,dev,t)
            expr = (power >= ison*minValue)
            return expr
        model.constrDevicePmin = pyo.Constraint(
                model.setDevice,model.setHorizon,rule=rule_devicePmin)

        def rule_deviceQmax(model,dev,t):
            # max/min power (zero if device is not on)
            if 'Qmax' not in model.paramDevice[dev]:
                return pyo.Constraint.Skip
            maxValue = model.paramDevice[dev]['Qmax']
            if 'profile' in model.paramDevice[dev]:
                # use an availability profile if provided
                extprofile = model.paramDevice[dev]['profile']
                maxValue = maxValue*model.paramProfiles[extprofile,t]
            flow = self.getDeviceFlow(model,dev,t)
            expr = ( flow <= maxValue)
            return expr
        model.constrDeviceQmax = pyo.Constraint(
                model.setDevice,model.setHorizon,rule=rule_deviceQmax)

        def rule_deviceQmin(model,dev,t):
            if 'Qmin' not in model.paramDevice[dev]:
                return pyo.Constraint.Skip
            minValue = model.paramDevice[dev]['Qmin']
            if 'profile' in model.paramDevice[dev]:
                # use an availability profile if provided
                extprofile = model.paramDevice[dev]['profile']
                minValue = minValue*model.paramProfiles[extprofile,t]
            flow = self.getDeviceFlow(model,dev,t)
            expr = (flow >= minValue)
            return expr
        model.constrDeviceQmin = pyo.Constraint(
                model.setDevice,model.setHorizon,rule=rule_deviceQmin)


        def rule_edgePmaxmin(model,edge,t):
            if model.paramEdge[edge]['type'] in ['el','heat']:
                vname='Pmax'
                expr = pyo.inequality(-model.paramEdge[edge][vname],
                        model.varEdgeFlow[edge,t],
                        model.paramEdge[edge][vname])
            else:
                vname='Qmax'
                expr = pyo.Constraint.Skip
                if vname in model.paramEdge[edge]:
                    expr = (model.varEdgeFlow[edge,t] <=
                            model.paramEdge[edge][vname])

            return expr
        model.constrEdgeBounds = pyo.Constraint(
                model.setEdge,model.setHorizon,rule=rule_edgePmaxmin)

        def rule_emissionRateLimit(model,t):
            '''Upper limit on CO2 emission rate'''
            emissionRateMax = model.paramParameters['emissionRateMax']
            if emissionRateMax < 0:
                # don't include constraint if parameter is set to negative value
                return pyo.Constraint.Skip
            else:
                lhs = self.compute_CO2(model,timesteps=[t])
                rhs = emissionRateMax
                return (lhs<=rhs)
        model.constrEmissionRateLimit = pyo.Constraint(model.setHorizon,
                                               rule=rule_emissionRateLimit)

        def rule_emissionIntensityLimit(model,t):
            '''Upper limit on CO2 emission intensity'''
            emissionIntensityMax = model.paramParameters['emissionIntensityMax']
            if emissionIntensityMax < 0:
                # don't include constraint if parameter is set to negative value
                return pyo.Constraint.Skip
            else:
                lhs = self.compute_CO2(model,timesteps=[t])
                rhs = emissionIntensityMax * self.compute_oilgas_export(
                        model,timesteps=[t])
                return (lhs<=rhs)
        model.constrEmissionIntensityLimit = pyo.Constraint(model.setHorizon,
                                               rule=rule_emissionIntensityLimit)

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

    @staticmethod
    def getDevicePower(model,dev,t):
        '''returns the variable that defines device power (depends on model)
        used for ramp rate limits, and print/plot'''
        devmodel = model.paramDevice[dev]['model']
        if devmodel in ['gasturbine','source_el']:
            devpower = model.varDeviceFlow[dev,'el','out',t]
        elif devmodel in ['sink_el','pump_water','pump_oil','compressor_el']:
            devpower = model.varDeviceFlow[dev,'el','in',t]
        elif devmodel in ['gasheater','source_heat']:
            devpower = model.varDeviceFlow[dev,'heat','out',t]
        elif devmodel in ['sink_heat']:
            devpower = model.varDeviceFlow[dev,'heat','in',t]
        elif devmodel in ['storage_el']:
            # TODO: summing is OK for specifying Pmax/Pmin limits,
            # But Pmax/Pmin is also specified in storage_el constraints


#            devpower = (model.varDeviceFlow[dev,'el','out',t]
#                - model.varDeviceFlow[dev,'el','in',t])

            # Include energy loss so that charging and discharging is not
            # happeing at the same time (cf expresstion used in constraints
            # for storage_el)
            eta = model.paramDevice[dev]['eta']
            devpower = -(model.varDeviceFlow[dev,'el','in',t]*eta
                        - model.varDeviceFlow[dev,'el','out',t]/eta)
        else:
            # TODO: Complete this
            # no need to define devpower for other devices
            devpower = 0
        return devpower

    @staticmethod
    def getDeviceFlow(model,dev,t):
        '''returns the variable that defines device flow'''
        devmodel = model.paramDevice[dev]['model']
        if devmodel in ['sink_water','well_injection']:
            flow = model.varDeviceFlow[dev,'water','in',t]
        elif devmodel in ['source_water','pump_water']:
            flow = model.varDeviceFlow[dev,'water','out',t]
        elif devmodel in ['well_production']:
            flow = model.varDeviceFlow[dev,'wellstream','out',t]
        elif devmodel in ['well_gaslift']:
            # flow into well from reservoir
            flow = (model.varDeviceFlow[dev,'oil','out',t]
                    + model.varDeviceFlow[dev,'gas','out',t]
                    - model.varDeviceFlow[dev,'gas','in',t]
                    + model.varDeviceFlow[dev,'water','out',t])
        elif devmodel in ['source_gas']:
            flow = model.varDeviceFlow[dev,'gas','out',t]
        else:
            raise Exception("Undefined flow variable for {}".format(devmodel))
        return flow

    # Helper functions
    @staticmethod
    def compute_CO2(model,devices=None,timesteps=None):
        '''compute CO2 emissions - average per sec (kgCO2/s)

        model can be abstract model or model instance
        '''
        if devices is None:
            devices = model.setDevice
        if timesteps is None:
            timesteps = model.setHorizon
#        deltaT = model.paramParameters['time_delta_minutes']*60
#        sumTime = len(timesteps)*deltaT

        sumCO2 = 0
        # GAS: co2 emission from consumed gas (e.g. in gas heater)
        # EL: co2 emission from the generation of electricity
        # HEAT: co2 emission from the generation of heat

#        # MJ = MW*s=MWh*1/3600
#        # co2 = flow [m3/s] * CV [MJ/m3] * co2content [kg/MWh]
#        #     = flow*CV*co2content [m3/s * MWs/m3 * kg/MWh= kg/h]
#        gasflow_co2 = (model.paramCarriers['gas']['CO2content'] #kg/MWh
#                        *model.paramCarriers['gas']['energy_value'] #MJ/m3
#                        ) #kg/m3*s/h

        gasflow_co2 = model.paramCarriers['gas']['CO2content'] #kg/m3

        for d in devices:
            devmodel = pyo.value(model.paramDevice[d]['model'])
            if devmodel in ['gasturbine','gasheater']:
                thisCO2 = sum(model.varDeviceFlow[d,'gas','in',t]
                              for t in timesteps)*gasflow_co2
            elif devmodel=='compressor_gas':
                thisCO2 = sum((model.varDeviceFlow[d,'gas','in',t]
                            -model.varDeviceFlow[d,'gas','out',t])
                            for t in timesteps)*gasflow_co2
            elif devmodel in ['source_el']:
                # co2 content in fuel combustion
                # co2em is kgCO2/MWh_el, deltaT is seconds, deviceFlow is MW
                # need to convert co2em to kgCO2/(MW*s)

                thisCO2 = sum(model.varDeviceFlow[d,'el','out',t]
                            *model.paramDevice[d]['co2em']
                            for t in timesteps)*1/3600
            elif devmodel in ['compressor_el','sink_heat','sink_el',
                              'heatpump','source_gas','sink_gas',
                              'sink_oil','sink_water',
                              'storage_el','separator','separator2',
                              'well_production','well_injection','well_gaslift',
                              'pump_oil','pump_wellstream','pump_water',
                              'source_water','source_oil']:
                # no CO2 emission contribution
                thisCO2 = 0
            else:
                raise NotImplementedError(
                    "CO2 calculation for {} not implemented".format(devmodel))
            sumCO2 = sumCO2 + thisCO2

        # Average per s
        sumCO2 = sumCO2/len(timesteps)
        return sumCO2

    @staticmethod
    def compute_startup_costs(model,devices=None, timesteps=None):
        '''startup costs (average per sec)'''
        if timesteps is None:
            timesteps = model.setHorizon
        if devices is None:
            devices = model.setDevice
        start_stop_costs = 0
        for d in devices:
            #devmodel = pyo.value(model.paramDevice[d]['model'])
            if 'startupCost' in model.paramDevice[d]:
                startupcost = pyo.value(model.paramDevice[d]['startupCost'])
                thisCost = sum(model.varDeviceStarting[d,t]
                              for t in timesteps)*startupcost
                start_stop_costs += thisCost
            if 'shutdownCost' in model.paramDevice[d]:
                shutdowncost = pyo.value(model.paramDevice[d]['shutdownCost'])
                thisCost = sum(model.varDeviceStopping[d,t]
                              for t in timesteps)*shutdowncost
                start_stop_costs += thisCost
        # get average per sec:
        deltaT = model.paramParameters['time_delta_minutes']*60
        sumTime = len(timesteps)*deltaT
        start_stop_costs = start_stop_costs/sumTime
        return start_stop_costs

    logging.info("TODO: operating cost for el storage - needs improvement")
    def compute_operatingCosts(self,model):
        '''term in objective function to represent fuel costs or similar
        as average per sec ($/s)

        opCost = energy costs (NOK/MJ, or NOK/Sm3)
        Note: el costs per MJ not per MWh
        '''
        sumCost = 0
        timesteps = model.setHorizon
        for dev in model.setDevice:
            if 'opCost' in model.paramDevice[dev]:
                opcost = model.paramDevice[dev]['opCost']
                for t in model.setHorizon:
                    varP = self.getDevicePower(model,dev,t)
                    #delta_t = model.paramParameters['time_delta_minutes']*60
                    #energy_in_dt = varP*delta_t
                    #sumCost += opcost*energy_in_dt
                    sumCost += opcost*varP
                    #logging.info('opcost={}, e={}, scost={}'.format(opcost,energy_in_dt,opcost*energy_in_dt))
        # average per sec
        sumCost = sumCost/len(timesteps)
        return sumCost

    @staticmethod
    def compute_costForDepletedStorage(model):
        '''term in objective function to discourage depleting battery,
        making sure it is used only when required'''
        storCost = 0
        for dev in model.setDevice:
            if model.paramDevice[dev]['model'] == 'storage_el':
                stor_cost=0
                if 'Ecost' in model.paramDevice[dev]:
                    stor_cost = model.paramDevice[dev]['Ecost']
                Emax = model.paramDevice[dev]['Emax']
                for t in model.setHorizon:
                    varE = model.varDeviceStorageEnergy[dev,t]
                    storCost += stor_cost*(Emax-varE)
        return storCost



    @staticmethod
    def compute_exportRevenue(model,carriers=None,timesteps=None):
        '''revenue from exported oil and gas - average per sec ($/s)'''
        if carriers is None:
            carriers = model.setCarrier
        if timesteps is None:
            timesteps = model.setHorizon

        inouts = Multicarrier.devicemodel_inout()
        sumRevenue = 0
        for dev in model.setDevice:
            devmodel = model.paramDevice[dev]['model']
            carriers_in = inouts[devmodel]['in']
            carriers_incl = [v for v in carriers if v in carriers_in]
            for c in carriers_incl:
                # flow in m3/s, price in $/m3
                price_param = 'price.{}'.format(c)
                if price_param in model.paramDevice[dev]:
                    inflow = sum(model.varDeviceFlow[dev,c,'in',t]
                                    for t in timesteps)
#                    sumRevenue += inflow*model.paramCarriers[c]['export_price']
                    sumRevenue += inflow*model.paramDevice[dev][price_param]
        # average per second (timedelta is not required)
        sumRevenue = sumRevenue/len(timesteps)
        return sumRevenue


    @staticmethod
    def compute_oilgas_export(model,timesteps=None):
        '''Export volume (Sm3oe/s)'''
        if timesteps is None:
            timesteps = model.setHorizon

        carriers = model.setCarrier
        export_node = model.paramParameters['export_node']
        export_devs = model.paramNodeDevices[export_node]
        inouts = Multicarrier.devicemodel_inout()
        flow_oilequivalents_m3_per_s = 0
        for dev in export_devs:
            devmodel = model.paramDevice[dev]['model']
            carriers_in = inouts[devmodel]['in']
            carriers_incl = [v for v in carriers if v in carriers_in]
            for c in carriers_incl:
                inflow = sum(model.varDeviceFlow[dev,c,'in',t]
                                for t in timesteps)
                # average flow, expressed in m3/s
                inflow = inflow/len(timesteps)
                # Convert from Sm3 to Sm3 oil equivalents:
                # REF https://www.norskpetroleum.no/kalkulator/om-kalkulatoren/
                if c=='oil':
                    flow_oilequivalents_m3_per_s += inflow
                elif c=='gas':
                    # 1 Sm3 gas = 1/1000 Sm3 o.e.
                    flow_oilequivalents_m3_per_s += inflow/1000
                else:
                    pass
        return flow_oilequivalents_m3_per_s


    @staticmethod
    def compute_CO2_intensity(model,timesteps=None):
        '''CO2 emission per exported oil/gas (kgCO2/Sm3oe)'''
        if timesteps is None:
            timesteps = model.setHorizon

        co2_kg_per_time = Multicarrier.compute_CO2(
                model,devices=None,timesteps=timesteps)
        flow_oilequivalents_m3_per_time = Multicarrier.compute_oilgas_export(
                model,timesteps)
        if pyo.value(flow_oilequivalents_m3_per_time)!=0:
            co2intensity = co2_kg_per_time/flow_oilequivalents_m3_per_time
        if pyo.value(flow_oilequivalents_m3_per_time)==0:
            logging.warning("zero export, so co2 intensity set to NAN")
            co2intensity = np.nan

        return co2intensity

    @staticmethod
    def compute_compressor_demand(model,dev,linear=False,
                                  Q=None,p1=None,p2=None,t=None):
        # power demand depends on gas pressure ratio and flow
        # See LowEmission report DSP5_2020_04 for description
        if model.paramDevice[dev]['model'] not in ['compressor_el',
                            'compressor_gas']:
            print("{} is not a compressor".format(dev))
            return
        k = model.paramCarriers['gas']['k_heat_capacity_ratio']
        Z = model.paramCarriers['gas']['Z_compressibility']
        # factor 1e-6 converts R units from J/kgK to MJ/kgK:
        R = model.paramCarriers['gas']['R_individual_gas_constant']*1e-6
        rho = model.paramCarriers['gas']['rho_density']
        T1 = model.paramDevice[dev]['temp_in'] #inlet temperature
        eta = model.paramDevice[dev]['eta'] #isentropic efficiency
        a = (k-1)/k
        c = rho/eta*1/(k-1)*Z*R*T1
        node = model.paramDevice[dev]['node']
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
            p10 = model.paramNode[node]['pressure.gas.in']
            p20 = model.paramNode[node]['pressure.gas.out']
            Q0 = model.paramDevice[dev]['Q0']
            P = c*(a*(p20/p10)**a * Q0*(p2/p20-p1/p10)
                     +((p20/p10)**a-1)*Q )
        else:
            P = c*((p2/p1)**a-1)*Q
        return P

    @staticmethod
    def compute_pump_demand(model,dev,linear=False,
                            Q=None,p1=None,p2=None,t=None,carrier='water'):
        devmodel = model.paramDevice[dev]['model']
        if devmodel=='pump_oil':
            carrier='oil'
        elif devmodel=='pump_water':
            carrier='water'
        else:
            print("{} is not a pump".format(dev))
            return
        # power demand vs flow rate and pressure difference
        # see eg. doi:10.1016/S0262-1762(07)70434-0
        # P = Q*(p_out-p_in)/eta
        # units: m3/s*MPa = MW
        #
        # Assuming incompressible fluid, so flow rate m3/s=Sm3/s
        # (this approximation may not be very good for multiphase
        # wellstream)
        # assume nominal pressure and keep only flow rate dependence
        #TODO: Better linearisation?

        node = model.paramDevice[dev]['node']
        eta = model.paramDevice[dev]['eta']

        if t is None:
            t=0
        if Q is None:
            Q = model.varDeviceFlow[dev,carrier,'in',t]
        if p1 is None:
            p1 = model.varPressure[node,carrier,'in',t]
        if p2 is None:
            p2 = model.varPressure[node,carrier,'out',t]
        if linear:
            # linearised equations around operating point
            # p1=p10, p2=p20, Q=Q0
            p10 = model.paramNode[node]['pressure.{}.in'.format(carrier)]
            p20 = model.paramNode[node]['pressure.{}.out'.format(carrier)]
            delta_p = p20 - p10
#            Q0 = model.paramDevice[dev]['Q0']
            #P = (Q*(p20-p10)+Q0*(p10-p1))/eta
            P = Q*delta_p/eta
#        elif self._quadraticConstraints:
#            # Quadratic constraint...
#            delta_p = (model.varPressure[(node,carrier,'out',t)]
#                        -model.varPressure[(node,carrier,'in',t)])
        else:
            # non-linear equation - for computing outside optimisation
            P = Q*(p2-p1)/eta

        return P

    @staticmethod
    def compute_exps_and_k(model,edge,carrier):
        '''Derive exp_s and k parameters for Weymouth equation'''

        # gas pipeline parameters - derive k and exp(s) parameters:
        ga=model.paramCarriers[carrier]

        #if 'temperature_K' in model.paramEdge[edge]:
        temp = model.paramEdge[edge]['temperature_K']
        height_difference = model.paramEdge[edge]['height_m']
        length = model.paramEdge[edge]['length_km']
        diameter = model.paramEdge[edge]['diameter_mm']
        s = 0.0684 * (ga['G_gravity']*height_difference
                        /(temp*ga['Z_compressibility']))
        if s>0:
            # height difference - use equivalent length
            sfactor= (np.exp(s)-1)/s
            length = length*sfactor

        k = (4.3328e-8*ga['Tb_basetemp_K']/ga['Pb_basepressure_MPa']
            *(ga['G_gravity']*temp*length*ga['Z_compressibility'])**(-1/2)
            *diameter**(8/3))
        exp_s = np.exp(s)
        return exp_s,k


    @staticmethod
    def compute_edge_pressuredrop(model,edge,p1,
            Q,method=None,linear=False):
        '''Compute pressure drop in pipe

        parameters
        ----------
        model : pyomo optimisation model
        edge : string
            identifier of edge
        p1 : float
            pipe inlet pressure (MPa)
        Q : float
            flow rate (Sm3/s)
        method : string
            None, weymouth, darcy-weissbach
        linear : boolean
            whether to use linear model or not

        Returns
        -------
        p2 : float
            pipe outlet pressure (MPa)'''

#        height_difference=0
        height_difference = model.paramEdge[edge]['height_m']
#        delta_z = model.paramEdge[edge]['height_m']
        method = None
        carrier = model.paramEdge[edge]['type']
#        if p1 is None:
#            # use nominal pressure
#            n1 = model.paramEdge[edge]['nodeFrom']
#            p1 = model.paramNode[n1]['pressure.{}.out'.format(carrier)]
        if 'pressure_method' in model.paramCarriers[carrier]:
            method = model.paramCarriers[carrier]['pressure_method']
#        if Q is None:
#            #use actual flow
#            Q = model.varEdgeFlow[(edge,t)]
#            #Q = pyo.value(model.varEdgeFlow[(edge,t)])

        n_from = model.paramEdge[edge]['nodeFrom']
        n_to = model.paramEdge[edge]['nodeTo']
        if (('pressure.{}.out'.format(carrier) in model.paramNode[n_from]) &
            ('pressure.{}.in'.format(carrier) in model.paramNode[n_to])):
            p0_from = model.paramNode[n_from][
                'pressure.{}.out'.format(carrier)]
            p0_to = model.paramNode[n_to][
                'pressure.{}.in'.format(carrier)]
            if (linear & (p0_from==p0_to)):
                method=None
                logging.debug(("{}-{}: Pipe without pressure drop"
                    " ({} / {} MPa)").format(n_from,n_to,p0_from,p0_to))
        elif linear:
            # linear equations, but nominal values not given - assume no drop
            method=None

        if method is None:
            # no pressure drop
            p2 = p1
            return p2

        elif method=='weymouth':
            '''
            Q = k * sqrt( Pin^2 - e^s Pout^2 ) [Weymouth equation, nonlinear]
                => Pout = sqrt(Pin^2-Q^2/k^2)/e^2
            Q = c * (Pin_0 Pin - e^s Pout0 Pout) [linearised version]
                => Pout = (Pin0 Pin - Q/c)/(e^s Pout0)
            c = k/sqrt(Pin0^2 - e^s Pout0^2)

            REFERENCES:
            1) E Sashi Menon, Gas Pipeline Hydraulics, Taylor & Francis (2005),
            https://doi.org/10.1201/9781420038224
            2) A Tomasgard et al., Optimization  models  for  the  natural  gas
            value  chain, in: Geometric Modelling, Numerical Simulation and
            Optimization. Springer Verlag, New York (2007),
            https://doi.org/10.1007/978-3-540-68783-2_16
            '''
            exp_s,k = Multicarrier.compute_exps_and_k(model,edge,carrier)
            logging.debug("pipe {}: exp_s={}, k={}"
                .format(edge,exp_s,k))
            if linear:
                p_from=p1
#                p_from = model.varPressure[(n_from,carrier,'out',t)]
#                p_to = model.varPressure[(n_to,carrier,'in',t)]
                X0 = p0_from**2-exp_s*p0_to**2
#                logging.info("edge {}-{}: X0={}, p1={},Q={}"
#                    .format(n_from,n_to,X0,p1,Q))
                coeff = k*(X0)**(-1/2)
#                Q_computed = coeff*(p0_from*p_from - exp_s*p0_to*p_to)
                p2 = (p0_from*p_from-Q/coeff)/(exp_s*p0_to)
            else:
                # weymouth eqn (non-linear)
                p2 = 1/exp_s*(p1**2-Q**2/k**2)**(1/2)

        elif method=='darcy-weissbach':
            grav=9.98 #m/s^2
            rho = model.paramCarriers[carrier]['rho_density']
            D = model.paramEdge[edge]['diameter_mm']/1000
            L = model.paramEdge[edge]['length_km']*1000

            if (('viscosity' in model.paramCarriers[carrier])
                & (not linear)):
                #compute darcy friction factor from flow rate and viscosity
                mu = model.paramCarriers[carrier]['viscosity']
                Re = 2*rho*Q/(np.pi*mu*D)
                f = 1/(0.838*scipy.special.lambertw(0.629*Re))**2
                f = f.real
            elif 'darcy_friction' in model.paramCarriers[carrier]:
                f = model.paramCarriers[carrier]['darcy_friction']
                Re = None
            else:
                raise Exception(
                    "Must provide viscosity or darcy_friction for {}"
                    .format(carrier))
#            logging.debug("\t{} ({}): Reynolds={}, Darcy friction f={:5.3g}"
#                .format(edge,carrier,Re,f))
            if linear:
                p_from = p1
#                p_from = model.varPressure[(n_from,carrier,'out',t)]
                k = np.sqrt(np.pi**2 * D**5/(8*f*rho*L))
                sqrtX = np.sqrt(p0_from*1e6 - p0_to*1e6
                                - rho*grav*height_difference)
                Q0 = k*sqrtX
#                Q = model.varEdgeFlow[edge,t]
                logging.debug(("derived pipe ({}) flow rate:"
                    " Q={}, linearQ0={:5.3g},"
                    " friction={:5.3g}")
                         .format(edge,Q,Q0,f))
                p2 = p_from - 1e-6*(Q-Q0)*2*sqrtX/k - (p0_from-p0_to)
                # linearised darcy-weissbach:
                # Q = Q0 + k/(2*sqrtX)*(p_from-p_to - (p0_from-p0_to))
            else:
                # darcy-weissbach eqn (non-linear)
                p2 = 1e-6*(
                    p1*1e6
                    - rho*grav*height_difference
                    - 8*f*rho*L*Q**2/(np.pi**2*D**5) )
        else:
            raise Exception("Unknown pressure drop calculation method ({})"
                            .format(method))
        return p2

    def darcy_weissbach_Q(p1,p2,f,rho,diameter_mm,
            length_km,height_difference_m=0):
        '''compute flow rate from darcy-weissbach eqn

        parameters
        ----------
        p1 : float
            pressure at pipe input (Pa)
        p2 : float
            pressure at pipe output (Pa)
        f : float
            friction factor
        rho : float
            fluid density (kg/m3)
        diameter_mm : float
            pipe inner diameter (mm)
        length_km : float
            pipe length (km)
        height_difference_m : float
            height difference output vs input (m)

        '''

        grav = 9.98
        L=length_km*1000
        D=diameter_mm/1000
        k = 8*f*rho*L/(np.pi**2*D**5)
        Q = np.sqrt( ((p1-p2)*1e6 - rho*grav*height_difference_m) / k )
        return Q

    logging.info("TODO: elReseve not including load reduction")
    logging.info("TODO: elReseve reserve_factor parameter")
    @staticmethod
    def compute_elReserve(model,t):
        '''compute non-used generator capacity (and available loadflex)
        This is reserve to cope with forecast errors, e.g. because of wind
        variation or motor start-up
        (does not include load reduction yet)
        '''
        alldevs = model.setDevice
        # relevant devices are devices with el output or input
        inout = Multicarrier.devicemodel_inout()
        cap_avail = 0
        p_generating = 0
        loadreduction = 0
        for d in alldevs:
            devmodel = model.paramDevice[d]['model']
            rf = 1
            if 'el' in inout[devmodel]['out']:
                # Generators and storage
                maxValue = model.paramDevice[d]['Pmax']
                if 'profile' in model.paramDevice[d]:
                    extprofile = model.paramDevice[d]['profile']
                    maxValue = maxValue*model.paramProfiles[extprofile,t]
                if devmodel in ['gasturbine']:
                    ison = model.varDeviceIsOn[d,t]
                    maxValue = ison*maxValue
                elif devmodel in ['storage_el']:
                    #available power may be limited by energy in the storage
                    # charging also contributes (can be reversed)
                    #(it can go to e.g. -2 MW to +2MW => 4 MW,
                    # even if Pmax=2 MW)
                    maxValue = (model.varDeviceStoragePmax[d,t]
                                +model.varDeviceFlow[d,'el','in',t])
                if ('reserve_factor' in model.paramDevice[d]):
                    # safety margin - only count a part of the forecast power
                    # towards the reserve, relevant for wind power
                    # (equivalently, this may be seen as increaseing the
                    # reserve margin requirement)
                    reserve_factor=model.paramDevice[d]['reserve_factor']
                    maxValue = maxValue*reserve_factor
                    if reserve_factor==0:
                        # no reserve contribution
                        rf = 0
                cap_avail += rf*maxValue
                p_generating += rf*model.varDeviceFlow[d,'el','out',t]
            elif 'el' in inout[devmodel]['in']:
                # Loads
                if ('reserve_factor' in model.paramDevice[d]):
                    # load reduction possible
                    #f_lr = model.paramDevice[d]['reserve_factor']
                    #loadreduction += f_lr*model.varDeviceFlow[d,'el','in',t]
                    #TODO: update, use above
                    loadreduction = 0
        res_dev = (cap_avail-p_generating) + loadreduction
        #TODO: include flexible loads (that can absorve variations)
        res_dev = (cap_avail-p_generating)
        return res_dev

    @staticmethod
    def compute_elBackup(model,t,exclude_device=None):
        '''Compute available reserve power that can take over in case
        of a fault. Consists of:
        1. generator unused capacity (el out)
        2. sheddable load (el in)'''
        alldevs = model.setDevice
        # relevant devices are devices with el output or input
        inout = Multicarrier.devicemodel_inout()
        el_devices = []
        for d in alldevs:
            devmodel = model.paramDevice[d]['model']
            if (('el' in inout[devmodel]['out'])
                or ('el' in inout[devmodel]['in'])):
                 el_devices.append(d)
        if exclude_device is None:
            otherdevs = el_devices
        else:
            otherdevs = [d for d in el_devices if d!=exclude_device]
        cap_avail = 0
        p_generating = 0
        p_sheddable = 0
        for d in otherdevs:
            devmodel = model.paramDevice[d]['model']
            if 'el' in inout[devmodel]['out']:
                #generator
                maxValue = model.paramDevice[d]['Pmax']
                if 'profile' in model.paramDevice[d]:
                    extprofile = model.paramDevice[d]['profile']
                    maxValue = maxValue*model.paramProfiles[extprofile,t]
                ison = 1
                if devmodel in ['gasturbine']:
                    ison = model.varDeviceIsOn[d,t]
                elif devmodel in ['storage_el']:
                    #available power may be limited by energy in the storage
                    delta_t = model.paramParameters['time_delta_minutes']/60 #hours
                    storageP = model.varDeviceStorageEnergy[d,t]/delta_t #MWh to MW
    #TODO: Take into account available energy in the storage
    # cannot use non-linear min() function here
    #                maxValue= min(maxValue,storageP)
                cap_avail += ison*maxValue
                p_generating += model.varDeviceFlow[d,'el','out',t]
            if 'el' in inout[devmodel]['in']:
                if ('backup_factor' in model.paramDevice[d]):
                    # sheddable load
                    shed_factor = model.paramDevice[d]['backup_factor']
                    p_sheddable += shed_factor*model.varDeviceFlow[d,'el','in',t]
        res_dev = (cap_avail-p_generating) + p_sheddable
        return res_dev


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
        timesteps_use_actual = self.instance.paramParameters['forecast_timesteps']
        self._timestep = timestep
        # Update profile (using actual for first 4 timesteps, forecast for rest)
        # -this is because for the first timesteps, we tend to have updated
        #  and quite good forecasts - for example, it may become apparent
        #  that there will be much less wind power than previously forecasted
        #
        for prof in self.instance.setProfile:
            for t in range(timesteps_use_actual): #0,1,2,3
                self.instance.paramProfiles[prof,t] = (
                        self._df_profiles_actual.loc[timestep+t,prof])
            for t in range(timesteps_use_actual,horizon):
                self.instance.paramProfiles[prof,t] = (
                        self._df_profiles_forecast.loc[timestep+t,prof])

        def _updateOnTimesteps(t_prev,dev):
            # sum up consequtive timesteps starting at tprev going
            # backwards, where device has been in preparation phase
            sum_on=0
            docontinue = True
            for tt in range(t_prev,-1,-1):
                #if (self.instance.varDeviceIsOn[dev,tt]==1):
                if (self.instance.varDeviceIsPrep[dev,tt]==1):
                    sum_on = sum_on+1
                else:
                    docontinue = False
                    break #exit for loop
            if docontinue:
                # we got all the way back to 0, so must include initial value
                sum_on = sum_on + self.instance.paramDevicePrepTimestepsInitially[dev]
            return sum_on

        # Update startup/shutdown info
        # pick the last value from previous optimistion prior to the present time
        if not first:
            t_prev = opt_timesteps-1
            for dev in self.instance.setDevice:
                # On/off status:
                self.instance.paramDeviceIsOnInitially[dev] = (
                        self.instance.varDeviceIsOn[dev,t_prev])
                self.instance.paramDevicePrepTimestepsInitially[dev] = (
                        _updateOnTimesteps(t_prev,dev))
                # Power output (relevant for ramp rate constraint):
                self.instance.paramDevicePowerInitially[dev] = (
                    self.getDevicePower(self.instance,dev,t_prev))
                # Energy storage:
                if self.instance.paramDevice[dev]['model'] in self.models_with_storage:
                    self.instance.paramDeviceEnergyInitially[dev] = (
                            self.instance.varDeviceStorageEnergy[dev,t_prev])

        # These constraints need to be reconstructed to update properly
        # (pyo.value(...) and/or if sentences...)
        self.instance.constrDevice_startup_delay.reconstruct()
        self.instance.constrDevice_startup_shutdown.reconstruct()
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


    def saveOptimisationResult(self,timestep,store_duals=None):
        '''save results of optimisation for later analysis'''

        #TODO: Implement result storage
        # hdf5? https://stackoverflow.com/questions/47072859/how-to-append-data-to-one-specific-dataset-in-a-hdf5-file-with-h5py)

        timelimit = self.instance.paramParameters['optimisation_timesteps']
        timeshift = timestep

        # Retrieve variable values
        varDeviceFlow = self.getVarValues(self.instance.varDeviceFlow,
              names=('device','carrier','terminal','time'))
        if (varDeviceFlow<-ZERO_WARNING_THRESHOLD).any():
            #get first index where this is true
            ind = varDeviceFlow[varDeviceFlow<-ZERO_WARNING_THRESHOLD].index[0]
            logging.warning("Negative number in varDeviceFlow - set to zero ({}:{})"
                .format(ind,varDeviceFlow[ind]))
            varDeviceFlow = varDeviceFlow.clip(lower=0)
        varDeviceIsPrep = self.getVarValues(self.instance.varDeviceIsPrep,
              names=('device','time'))
        varDeviceIsOn = self.getVarValues(self.instance.varDeviceIsOn,
              names=('device','time'))
        varDeviceStorageEnergy = self.getVarValues(self.instance.varDeviceStorageEnergy,
              names=('device','time'))
        varDeviceStoragePmax = self.getVarValues(self.instance.varDeviceStoragePmax,
              names=('device','time'))
        varDeviceStarting = self.getVarValues(self.instance.varDeviceStarting,
              names=('device','time'))
        varDeviceStopping = self.getVarValues(self.instance.varDeviceStopping,
              names=('device','time'))
        varEdgeFlow = self.getVarValues(self.instance.varEdgeFlow,
              names=('edge','time'))
        varElVoltageAngle = self.getVarValues(self.instance.varElVoltageAngle,
              names=('node','time'))
        varPressure = self.getVarValues(self.instance.varPressure,
              names=('node','carrier','terminal','time'))
        varTerminalFlow = self.getVarValues(self.instance.varTerminalFlow,
              names=('node','carrier','time'))

        if store_duals is not None:
            # Save dual values
            # store_duals = {
            #   'elcost': {'constr':'constrDevicePmin','indx':('util',None)}
            #   }
            horizon_steps = self.instance.paramParameters['planning_horizon']
            if self._dfDuals is None:
                self._dfDuals = pd.DataFrame(columns=store_duals.keys())
            for key,val in store_duals.items():
                #vrs=('util',None)
                vrs = val['indx']
                constr = getattr(self.instance,val['constr'])
                sumduals = 0
                for t in range(timelimit):
                    # Replace None by the timestep, ('util',None) -> ('util',t)
                    vrs1=tuple(x if x is not None else t for x in vrs)
                    dual = self.instance.dual[constr[vrs1]]
                    # The dual gives the improvement in the objective function
                    # if the constraint is relaxed by one unit.
                    # The units of the dual prices are the units of the
                    # objective function divided by the units of the constraint.
                    #
                    # A constraint is for a single timestep, whereas the
                    # objective function averages over all timesteps in the
                    # optimisation horizon. To get the improvement of relaxing
                    # the constraint not just in the single timestep, but in
                    # all timesteps we therefore scale up the dual value
                    dual = dual * horizon_steps
                    self._dfDuals.loc[timeshift+t,key] = dual

        # CO2 emission rate (sum all emissions)
        co2 = [pyo.value(self.compute_CO2(self.instance,timesteps=[t]))
                for t in range(timelimit)]
        self._dfCO2rate = pd.concat(
                [self._dfCO2rate,
                 pd.Series(data=co2,index=range(timestep,timestep+timelimit))])

        # CO2 emission rate per device:
        df_co2=pd.DataFrame()
        for d in self.instance.setDevice:
            for t in range(timelimit):
                co2_dev = self.compute_CO2(self.instance,devices=[d],timesteps=[t])
                df_co2.loc[t+timestep,d] = pyo.value(co2_dev)
        self._dfCO2rate_per_dev = pd.concat([self._dfCO2rate_per_dev,df_co2])
        #self._dfCO2dev.sort_index(inplace=True)

        # CO2 emission intensity (sum)
        df_df_co2intensity = [pyo.value(self.compute_CO2_intensity(self.instance,timesteps=[t]))
                for t in range(timelimit)]
        self._dfCO2intensity = pd.concat([self._dfCO2intensity,
                  pd.Series(data=df_df_co2intensity,
                            index=range(timestep,timestep+timelimit))])

        # Revenue from exported energy
        df_exportRevenue=pd.DataFrame()
        for c in self.instance.setCarrier:
            for t in range(timelimit):
                exportRevenue_dev = self.compute_exportRevenue(
                        self.instance,carriers=[c],timesteps=[t])
                df_exportRevenue.loc[t+timestep,c] = pyo.value(exportRevenue_dev)
        self._dfExportRevenue = pd.concat(
                [self._dfExportRevenue,df_exportRevenue])

        # Reserve capacity
        # for all generators, should have reserve_excl_generator>p_out
        df_reserve=pd.DataFrame()
        devs_elout = self.getDevicesInout(carrier_out='el')
        for t in range(timelimit):
            rescap = pyo.value(self.compute_elReserve(self.instance,t))
            df_reserve.loc[t+timestep,'reserve'] = rescap
        self._dfElReserve = pd.concat([self._dfElReserve,df_reserve])

        # Backup capacity
        # for all generators, should have reserve_excl_generator>p_out
        df_backup=pd.DataFrame()
        devs_elout = self.getDevicesInout(carrier_out='el')
        for t in range(timelimit):
            for d in devs_elout:
                rescap = pyo.value(self.compute_elBackup(
                        self.instance,t,exclude_device=d))
                df_backup.loc[t+timestep,d] = rescap
        self._dfElBackup = pd.concat([self._dfElBackup,df_backup])

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
        self._dfDeviceIsPrep = _addToDf(self._dfDeviceIsPrep,varDeviceIsPrep)
        self._dfDeviceStorageEnergy = _addToDf(self._dfDeviceStorageEnergy,varDeviceStorageEnergy)
        self._dfDeviceStoragePmax = _addToDf(self._dfDeviceStoragePmax,varDeviceStoragePmax)
        self._dfDeviceStarting = _addToDf(self._dfDeviceStarting,varDeviceStarting)
        self._dfDeviceStopping = _addToDf(self._dfDeviceStopping,varDeviceStopping)
        self._dfEdgeFlow = _addToDf(self._dfEdgeFlow,varEdgeFlow)
        self._dfElVoltageAngle = _addToDf(self._dfElVoltageAngle,varElVoltageAngle)
        self._dfTerminalPressure = _addToDf(self._dfTerminalPressure,varPressure)
        self._dfTerminalFlow = _addToDf(self._dfTerminalFlow,varTerminalFlow)
        return


    def solve(self,solver="gurobi",write_yaml=False,timelimit=None):
        """Solve problem for planning horizon at a single timestep"""

        opt = pyo.SolverFactory(solver)
        if timelimit is not None:
            if solver == 'gurobi':
                opt.options['TimeLimit'] = timelimit
            elif solver == 'cbc':
                opt.options['sec'] = timelimit
            elif solver == 'cplex':
                opt.options['timelimit'] = timelimit
            elif solver == 'glpk':
                opt.options['tmlim'] = timelimit
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

    def solveMany(self,solver="gurobi",timerange=None,write_yaml=False,
                  timelimit=None,store_duals=None):
        """Solve problem over many timesteps - rolling horizon
        timelimit : int
            Time limit spent on each optimisation (sec)
        store_duals : dict
            Store dual values of constraints. The dictionary contains a
            key:value list where value is a new dictionary specifying the
            constraint and indices. None in the index is replaced by time
            Example:
            store_duals = {'elcost':{constr=constrDevicePmin, indx:('util',None)}
        """

        steps = self.instance.paramParameters['optimisation_timesteps']
        horizon = self.instance.paramParameters['planning_horizon']
        if timelimit is not None:
            logging.debug("Using solver timelimit={}".format(timelimit))
        if timerange is None:
            time_start = 0
            time_end = self._df_profiles_forecast.index.max()+1-horizon
        else:
            time_start = timerange[0]
            time_end=timerange[1]
        df_profiles = pd.DataFrame()

        first=True
        for step in trange(time_start,time_end,steps):
            if not has_tqdm:
                # no progress bar
                logging.info("Solving timestep={}".format(step))
            # 1. Update problem formulation
            self.updateModelInstance(step,df_profiles,first=first)
            # 2. Solve for planning horizon
            self.solve(solver=solver,write_yaml=write_yaml,timelimit=timelimit)
            # 3. Save results (for later analysis)
            self.saveOptimisationResult(step,store_duals)
            first = False



    def printSolution(self,instance):

        print("\nSOLUTION - edgeFlow:")
        for k in instance.varEdgeFlow.keys():
            power = instance.varEdgeFlow[k].value
            print("  {}: {}".format(k,power))

        print("\nSOLUTION - Pressure:")
        for k,v in instance.varPressure.get_values().items():
            pressure = v #pyo.value(v)
            print("  {}: {}".format(k,pressure))
            #df_edge.loc[k,'gasPressure'] = pressure

        # display all duals
        print ("Duals")
        for c in instance.component_objects(pyo.Constraint, active=True):
            print ("   Constraint",c)
            for index in c:
                print ("      ", index, instance.dual[c[index]])

    def nodeIsNonTrivial(self,node,carrier):
        '''returns True if edges or devices are connected to node for this carrier
        '''
        model = self.instance
        isNontrivial = False
        # edges connected?
        if (((carrier,node) in model.paramNodeEdgesFrom) or
                ((carrier,node) in model.paramNodeEdgesTo)):
            isNontrivial = True
            return isNontrivial
        # devices connected?
        if node in model.paramNodeDevices:
            mydevs = model.paramNodeDevices[node]
            devmodels = [model.paramDevice[d]['model'] for d in mydevs]
            for dev_model in devmodels:
                carriers_used = [item for sublist in
                     #list(model.paramDevicemodel[dev_model].values())
                     list(self._devmodels[dev_model].values())
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


    def getDevicesInout(self,carrier_in=None,carrier_out=None):
        '''devices that have the specified connections in and out'''
        model = self.instance
        inout = self.devicemodel_inout()
        devs = []
        for d in model.setDevice:
            devmodel = model.paramDevice[d]['model']
            ok_in = ((carrier_in is None)
                        or (carrier_in in inout[devmodel]['in']))
            ok_out = ((carrier_out is None)
                        or (carrier_out in inout[devmodel]['out']))
            if ok_in and ok_out:
                devs.append(d)
        return devs


    def checkEdgePressureDrop(self,timestep=0,var="outer"):
        model = self.instance
        for k,edge in model.paramEdge.items():
            carrier = edge['type']
            if carrier in ['gas','oil','wellstream','water']:
                node_from = edge['nodeFrom']
                node_to = edge['nodeTo']
                print("{} edge {}:{}-{}"
                       .format(carrier,k,node_from,node_to),end=" ")
                if not (('pressure.{}.out'.format(carrier)
                    in model.paramNode[node_from]) and (
                     'pressure.{}.in'.format(carrier)
                    in model.paramNode[node_to])):
                    print("--")
                    continue
                p_in = model.paramNode[node_from][
                    'pressure.{}.out'.format(carrier)]
                p_out = model.paramNode[node_to][
                    'pressure.{}.in'.format(carrier)]
                if p_out==p_in:
                    # no nominal pressure drop - i.e. pressure drop is not
                    # modelled, so skip to next
                    print("---")
                    continue
                print("{} edge {}:{}-{}"
                       .format(carrier,k,node_from,node_to))
                if var=="inner":
                    # get value from inner optimisation (over rolling horizon)
                    Q = model.varEdgeFlow[(k,timestep)]
                    var_p_in = model.varPressure[
                        (node_from,carrier,'out',timestep)]
                    var_p_out = model.varPressure[
                        (node_to,carrier,'in',timestep)]
                elif var=="outer":
                    # get value from outer loop (decisions)
                    Q = self._dfEdgeFlow[(k,timestep)]
                    var_p_in = self._dfTerminalPressure[
                        (node_from,carrier,'out',timestep)]
                    var_p_out = self._dfTerminalPressure[
                        (node_to,carrier,'in',timestep)]

                print("{} edge {}:{}-{} (Q={} m3/s)"
                       .format(carrier,k,node_from,node_to,Q))
                diameter = model.paramEdge[k]['diameter_mm']/1000

                if 'num_pipes' in edge:
                    Q = Q/edge['num_pipes']

                if var in ["inner","outer"]:
                    p_out_comp = self.compute_edge_pressuredrop(model,edge=k,
                            p1=p_in,Q=Q,linear=False)
                    p_out_comp_linear = pyo.value(
                        self.compute_edge_pressuredrop(
                            model,edge=k,p1=p_in,Q=Q,linear=True))
                    p_out_comp2 = self.compute_edge_pressuredrop(model,edge=k,
                             p1=var_p_in,Q=Q,linear=False)
                    p_out_comp2_linear = pyo.value(
                        self.compute_edge_pressuredrop(model,edge=k,
                             p1=var_p_in,Q=Q,linear=True))

                pressure0 = 0.1 # MPa, standard condition (Sm3)
                velocity=4*Q/(np.pi*diameter**2)
                if carrier=='gas':
                    # convert flow rate from Sm3/s to m3/s at the actual pressure:
                    # ideal gas pV=const => pQ=const => Q1=Q0*(p0/p1)
                    pressure1 = (var_p_in+var_p_out)/2
                    Q1 = Q*(pressure0/pressure1)
                    velocity=4*Q1/(np.pi*diameter**2)
                print(("\tNOMINAL:    pin={}  pout={}  pout_computed={:3.5g}"
                        " pout_linear={:3.5g}")
                        .format(p_in,p_out,p_out_comp,p_out_comp_linear))
                if var in ["inner","outer"]:
                    print(("\tSIMULATION: pin={}  pout={}"
                           "  pout_computed={:3.5g} pout_linear={:3.5g}")
                          .format(var_p_in,var_p_out,
                          p_out_comp2,p_out_comp2_linear))
                print("\tflow velocity = {:3.5g} m/s".format(velocity))



    def exportSimulationResult(self,filename):
        '''Write saved simulation results to file'''

        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter(filename, engine='xlsxwriter')

        if not self._dfCO2intensity.empty:
            self._dfCO2intensity.to_excel(writer,sheet_name="CO2intensity")
        if not self._dfCO2rate.empty:
            self._dfCO2rate.to_excel(writer,sheet_name="CO2rate")
        if not self._dfCO2rate_per_dev.empty:
            self._dfCO2rate_per_dev.to_excel(writer,sheet_name="CO2rate_per_dev")
        if not self._dfDeviceStorageEnergy.empty:
            self._dfDeviceStorageEnergy.to_excel(writer,sheet_name="DeviceStorageEnergy")
        if not self._dfDeviceFlow.empty:
            self._dfDeviceFlow.to_excel(writer,sheet_name="DeviceFlow")
        if not self._dfDeviceIsOn.empty:
            self._dfDeviceIsOn.to_excel(writer,sheet_name="DeviceIsOn")
        #if not self._dfDevicePower.empty:
        #    self._dfDevicePower.to_excel(writer,sheet_name="DevicePower")
#        if not self._dfDeviceStarting.empty:
#            self._dfDeviceStarting.to_excel(writer,sheet_name="DeviceStarting")
#        if not self._dfDeviceStopping.empty:
#            self._dfDeviceStopping.to_excel(writer,sheet_name="DeviceStopping")
#        if not self._dfEdgeFlow.empty:
#            self._dfEdgeFlow.to_excel(writer,sheet_name="EdgeFlow")
#        if not self._dfElVoltageAngle.empty:
#            self._dfElVoltageAngle.to_excel(writer,sheet_name="ElVoltageAngle")
        if not self._dfExportRevenue.empty:
            self._dfExportRevenue.to_excel(writer,sheet_name="ExportRevenue")
#        if not self._dfTerminalFlow.empty:
#            self._dfTerminalFlow.to_excel(writer,sheet_name="TerminalFlow")
        if not self._dfTerminalPressure.empty:
            self._dfTerminalPressure.to_excel(writer,sheet_name="TerminalPressure")

        # Close the Pandas Excel writer and output the Excel file.
        writer.save()

    def compute_kpis(self,windturbines=[]):
        '''Compute key indicators of simulation results'''
        hour_per_year=8760
        sec_per_year=3600*hour_per_year
        kpi = {}
        mc = self

        num_sim_timesteps=mc._dfCO2rate.shape[0]
        timesteps=mc._dfCO2rate.index
        kpi['hours_simulated'] = num_sim_timesteps*mc.instance.paramParameters['time_delta_minutes']/60

        # CO2 emissions
        kpi['kgCO2_per_year'] = mc._dfCO2rate.mean()*sec_per_year
        kpi['kgCO2_per_Sm3oe'] = mc._dfCO2intensity.mean()

        # hours with reduced load
#        kpi['reducedload_hours_per_year'] = None

        # hours with load shedding
#        kpi['loadshed_hours_per_year'] = None

        # fuel consumption
        gasturbines = [i for i,g in mc.instance.paramDevice.items()
                        if g['model']=='gasturbine']
        mask_gt=mc._dfDeviceFlow.index.get_level_values('device').isin(gasturbines)
        gtflow = mc._dfDeviceFlow[mask_gt]
        fuel = gtflow.unstack('carrier')['gas'].unstack('terminal')['in'].unstack().mean(axis=1)
        kpi['gt_fuel_sm3_per_year'] = fuel.sum()*sec_per_year

        # electric power consumption
        el_dem=mc._dfDeviceFlow.unstack('carrier')['el'].unstack('terminal')['in'].dropna().unstack().mean(axis=1)
        kpi['elconsumption_mwh_per_year'] = el_dem.sum()*hour_per_year


        # number of generator starts
        gt_starts = mc._dfDeviceStarting.unstack().sum(axis=1)[gasturbines].sum()
        kpi['gt_starts_per_year'] = gt_starts*hour_per_year/kpi['hours_simulated']

        # number of generator stops
        gt_stops = mc._dfDeviceStopping.unstack().sum(axis=1)[gasturbines].sum()
        kpi['gt_stops_per_year'] = gt_stops*hour_per_year/kpi['hours_simulated']

        # running hours of generators
        gt_ison = mc._dfDeviceIsOn.unstack().sum(axis=1)[gasturbines].sum()
        kpi['gt_hoursrunning_per_year'] = gt_ison*hour_per_year/kpi['hours_simulated']

        # wind power output
        el_sup=mc._dfDeviceFlow.unstack('carrier')['el'].unstack('terminal')['out'].dropna().unstack()
        p_wind = el_sup.T[windturbines]
        kpi['wind_output_mwh_per_year'] = p_wind.sum(axis=1).mean()*hour_per_year

        # curtailed wind energy
        p_avail = pd.DataFrame(index=timesteps)
        for d in windturbines:
            devparam=mc.instance.paramDevice[d]
            Pmax = devparam['Pmax']
            p_avail[d] = Pmax
            if 'profile' in devparam:
                profile_ref = devparam['profile']
                p_avail[d] = Pmax * mc._df_profiles_actual.loc[timesteps,profile_ref]
        p_curtailed = (p_avail - p_wind).sum(axis=1)
        kpi['wind_curtailed_mwh_per_year'] = p_curtailed.mean()*hour_per_year
        return kpi


class Plots:


    def plotDevicePowerLastOptimisation1(mc,device,filename=None):
        model = mc.instance
        devname = model.paramDevice[device]['name']
        maxP = model.paramDevice[device]['Pmax']
        plt.figure(figsize=(12,4))
        ax=plt.gca()
        #plt.title("Results from last optimisation")
        plt.xlabel("Timestep in planning horizon")
        plt.ylabel("Device power (MW)")
        label_profile=""
        if 'profile' in model.paramDevice[device]:
            profile = model.paramDevice[device]['profile']
            dfa = mc.getProfiles(profile)
            dfa = dfa*maxP
            dfa[profile].plot(ax=ax,label='available')
            label_profile="({})".format(profile)

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
                    df_this[device].plot(ax=ax,linestyle='-',drawstyle="steps-post",marker=".",
                           label="actual ({} {})".format(carr,term))
        ax.legend(loc='upper left')#, bbox_to_anchor =(1.01,0),frameon=False)

        dfE = pd.DataFrame.from_dict(model.varDeviceStorageEnergy.get_values(),
                                    orient="index")
        dfE.index = pd.MultiIndex.from_tuples(dfE.index,
                               names=('device','time'))
        dfE = dfE[0].dropna()
        df_this = dfE.unstack(0)
        # shift by one because storage at t is storage value _after_ t
        # (just before t+1)
        df_this.index = df_this.index+1
        if device in df_this:
            ax2=ax.twinx()
            ax2.set_ylabel("Energy (MWh)",color="red")
            ax2.tick_params(axis='y', labelcolor="red")
            df_this[device].plot(ax=ax2,linestyle='--',color='red',
                   label="storage".format(carr,term))
            ax2.legend(loc='upper right',)
        ax.set_xlim(left=0)


        plt.title("{}:{} {}".format(device,devname,label_profile))
        if filename is not None:
            plt.savefig(filename,bbox_inches = 'tight')


    # def plotDevicePowerLastOptimisation(model,devices='all',filename=None):
    #     """Plot power schedule over planning horizon (last optimisation)"""
    #     if devices=='all':
    #         devices = list(model.setDevice)
    #     varPower = model.getDevicePower()
    #     df = pd.DataFrame.from_dict(model.varDevicePower.get_values(),
    #                                 orient="index")
    #     df.index = pd.MultiIndex.from_tuples(df.index,names=('device','time'))
    #     df = df[0].unstack(level=0)
    #     df_info = pd.DataFrame.from_dict(dict(model.paramDevice.items())).T
    #
    #     plt.figure(figsize=(12,4))
    #     ax=plt.gca()
    #     df[devices].plot(ax=ax)
    #     labels = (df_info.loc[devices].index.astype(str)
    #               +'_'+df_info.loc[devices,'name'])
    #     plt.legend(labels,loc='lower left', bbox_to_anchor =(1.01,0),
    #                frameon=False)
    #     plt.xlabel("Timestep")
    #     plt.ylabel("Device power (MW)")
    #     if filename is not None:
    #         plt.savefig(filename,bbox_inches = 'tight')
    #

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

        axes[1].set_xlabel("Timestep in planning horizon")
        plt.suptitle("Result from last optimisation")
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
        plt.ylabel("Emission rate (kgCO2/s)")
        ax.legend(loc='lower left', bbox_to_anchor =(1.01,0),
                  frameon=False)
        if filename is not None:
            plt.savefig(filename,bbox_inches = 'tight')




#########################################################################

def _nodeCarrierHasSerialDevice(df_node,df_device):
    devmodel_inout = Multicarrier.devicemodel_inout()
    node_devices = df_device.groupby('node').groups

    # extract carriers (from defined device models)
    sublist = [v['in']+v['out'] for k,v in devmodel_inout.items()]
    flatlist = [item for sublist2 in sublist for item in sublist2]
    allCarriers = set(flatlist)
    node_carrier_has_serialdevice = {}
    for n in df_node.index:
        devs_at_node=[]
        if n in node_devices:
            devs_at_node = node_devices[n]
        node_carrier_has_serialdevice[n] = {}
        for carrier in allCarriers:
            # default to false:
            node_carrier_has_serialdevice[n][carrier] = False
            for dev_mod in df_device.loc[devs_at_node,'model']:
                #has_series = ((carrier in devmodel_inout[dev_mod]['in'])
                #                and (carrier in devmodel_inout[dev_mod]['out']))
                if (('serial' in devmodel_inout[dev_mod]) and
                            (carrier in devmodel_inout[dev_mod]['serial'])):
                    node_carrier_has_serialdevice[n][carrier] = True
                    break
    return node_carrier_has_serialdevice


def _convert_xls_input(df,columns,index_col='id'):
    '''Convert from XLSX format input to flat DataFrame
    '''
    df[columns] = df[columns].fillna(method='ffill')
    if index_col is None:
        df.reset_index(drop=True)

    df2=df[[index_col,'param_id','param_value']].set_index([index_col,'param_id'])
    df2=df2.squeeze().unstack()
    df2=df2.dropna(axis=1,how='all')
    df4=df[columns].set_index(index_col)
    # drop duplicate (indices):
    df4=df4.loc[~df4.index.duplicated(keep='first')]
    df5 = df4.join(df2)
    return df5



def read_data_from_yaml(filename):
    """Read input data from yaml file"""
    data_dict = None
    with open(filename,"r") as text_file:
        data_dict = yaml.safe_load(text_file)
    #data = create_initdata(data_dict=data_dict)
    return data_dict

def read_profiles_from_hd5(filename,key_actual="actual",key_forecast="forecast"):
    profiles = {}
    profiles['actual'] = pd.read_hdf(filename, key=key_actual)
    profiles['forecast'] = pd.read_hdf(filename, key=key_forecast)
    return profiles

def save_profiles_to_hd5(filename,profiles):
    '''Save profiles to HDF5'''
    for k in profiles:
        profiles[k].to_hdf(filename,key=k,mode="a")
    return

def read_profiles_from_xlsx(filename,
        sheet_forecast="profiles",sheet_nowcast="profiles_forecast",
        exclude_cols=[]):
    df_profiles = pd.read_excel(filename,sheet_name=sheet_forecast,
        index_col="timestep",
        usecols=lambda col: col not in exclude_cols)
    df_profiles_forecast = pd.read_excel(filename,
        sheet_name=sheet_nowcast,index_col="timestep",
        usecols=lambda col: col not in exclude_cols)
    profiles = {'actual':df_profiles,'forecast':df_profiles_forecast}
    return profiles

def read_data_from_xlsx(filename,includeforecast=True):
    """Read input data from spreadsheet.

    filename : str
        name of file
    """
    dfs = {}
    dfs['node'] = _convert_xls_input(pd.read_excel(filename,sheet_name="node"),
        columns=['id','name'],index_col='id')
    dfs['edge'] = _convert_xls_input(pd.read_excel(filename,sheet_name="edge"),
        columns=['id','include','nodeFrom','nodeTo','type',
        'length_km'],index_col='id')
    dfs['device'] = _convert_xls_input(pd.read_excel(filename,sheet_name="device"),
        columns=['id','include','node','model','name'],index_col='id')
    dfs['parameters'] = pd.read_excel(filename,sheet_name="parameters",index_col=0)
    dfs['parameters'].rename(
        columns={'param_id':'id','param_value':'value'},inplace=True)
    dfs['carriers'] = _convert_xls_input(pd.read_excel(
        filename,sheet_name="carriers"),
        columns=['id'],index_col='id')

    data = create_initdata(dfs=dfs)
    if includeforecast:
        profiles = read_profiles_from_xlsx(filename)
        return data, profiles
    return data


def _oogeso_dict_to_df(data_dict):
    '''convert dict to dataframes for further processing in create_initdata'''
    def d2d(key):
        '''convert from dict to dataframe, makins sure to not miss indices
        without parameters'''
        return pd.DataFrame.from_dict(data_dict[key],orient="index"
                    ).reindex(data_dict[key].keys())

    dfs = {}
    dfs['parameters'] = pd.DataFrame.from_dict(
        data_dict['paramParameters'],orient='index',columns=['value'])
    dfs['carriers'] = d2d('paramCarriers')
    dfs['node'] = d2d('paramNode')
    dfs['edge'] = d2d('paramEdge')
    dfs['device'] = d2d('paramDevice')
    return dfs

#def _df_to_yaml(df):
#    '''convert dataframe to yaml, ignoring None/nan entries'''
#    dfdict={i:{i2:v2 for i2,v2 in v.items() if not pd.isnull(v2)} for i,v in df_nodes.iterrows()}
#    yaml_str = yaml.safe_dump(dfdict)
#    return yaml_str

#def _df_to_dict_dropnull(df):
#    '''convert dataframe to dict, ignoring None/nan entries'''
#    dfdict={i:{i2:v2 for i2,v2 in v.items() if not pd.isnull(v2)} for i,v in df_nodes.iterrows()}
#    return dfdict


def NOTUSED_filter_data_include(data_dict):
    '''filter out items not to be included'''
    data2={}
    for k1,v1 in data_dict.items():
        data2[k1]={}
        for k,v in v1.items():
            is_excluded = (type(v) is dict) and ('include' in v) and (v['include']==0)
            if not is_excluded:
                data2[k1][k] = v
            else:
                 logging.debug("Excluding {}:{}".format(k1,k))
    return data2

def to_dict_dropna(df):
  return {k:r.dropna().to_dict() for k,r in df.iterrows()}

def create_initdata(data_dict=None,dfs=None):
    """Convert input data to data structure required by pyomo model"""

    if data_dict is not None:
        if dfs is not None:
            raise Exception("Provide only one of data_dict and dfs parameters")
        dfs = _oogeso_dict_to_df(data_dict)
    df_node = dfs['node']
    df_edge= dfs['edge']
    df_device= dfs['device']
    df_carriers= dfs['carriers']
    df_parameters= dfs['parameters']

    # default values if missing from input:
    if 'height_m'  in df_edge:
        df_edge['height_m'] = df_edge['height_m'].fillna(0)
    else:
        df_edge['height_m']=0

    # discard edges and devices not to be included:
    df_edge = df_edge[df_edge['include']==1]
    df_device = df_device[df_device['include']==1]

    if not 'profile' in df_device:
        df_device['profile'] = np.nan

    # Set node terminal nominal pressure based on edge from/to pressure values
    for i,edg in df_edge.iterrows():
        if 'pressure.from' not in edg:
            continue
        if np.isnan(edg['pressure.from']):
            continue
        n_from=edg['nodeFrom']
        n_to =edg['nodeTo']
        typ=edg['type']
        p_from=edg['pressure.from']
        #m_from=(df_node.index==n_from)
        c_out='pressure.{}.out'.format(typ)
        p_to=edg['pressure.to']
        #m_to=(df_node.index==n_to)
        c_in='pressure.{}.in'.format(typ)
        # Check that pressure values are consistent:
        is_consistent = True
        existing_p_from=None
        existing_p_to=None
        try:
            existing_p_from = df_node.loc[n_from,c_out]
            logging.debug("{} p_from = {} (existing) / {} (new)"
                         .format(i,existing_p_from,p_from))
            if ((not np.isnan(existing_p_from)) and (existing_p_from!=p_from)):
                msg =("Input data edge pressure from values are"
                      " inconsistent (edge={}, {}!={})"
                      ).format(i,existing_p_from,p_from)
                is_consistent=False
        except:
            pass
        try:
            existing_p_to = df_node.loc[n_to,c_in]
            logging.debug("{} p_to = {} (existing) / {} (new)"
                         .format(i,existing_p_to,p_to))
            if ((not np.isnan(existing_p_to)) and (existing_p_to!=p_to)):
                msg =("Input data edge pressure to values are"
                      " inconsistent (edge={})").format(i)
                is_consistent=False
        except:
            pass
        if not is_consistent:
            print(df_node)
            raise Exception(msg)

        df_node.loc[n_from,c_out] = p_from
        df_node.loc[n_to,c_in] = p_to

    carrier_properties = to_dict_dropna(df_carriers)
    allCarriers = list(carrier_properties.keys())

    node_carrier_has_serialdevice = _nodeCarrierHasSerialDevice(df_node,df_device)

    # # gas pipeline parameters - derive k and exp(s) parameters:
    # ga=carrier_properties['gas']
    # temp = df_edge['temperature_K']
    # height_difference = df_edge['height_m']
    # s = 0.0684 * (ga['G_gravity']*height_difference
    #                 /(temp*ga['Z_compressibility']))
    # sfactor= (np.exp(s)-1)/s
    # sfactor.loc[s==0] = 1
    # length = df_edge['length_km']*sfactor
    # diameter = df_edge['diameter_mm']
    #
    # gas_edge_k = (4.3328e-8*ga['Tb_basetemp_K']/ga['Pb_basepressure_MPa']
    #     *(ga['G_gravity']*temp*length*ga['Z_compressibility'])**(-1/2)
    #     *diameter**(8/3))
    # df_edge['gasflow_k'] = gas_edge_k
    # df_edge['exp_s'] = np.exp(s)

    coeffB,coeffDA = computePowerFlowMatrices(df_node,df_edge,baseZ=1)
    planning_horizon = df_parameters.loc['planning_horizon','value']
    data = {}
    data['setCarrier'] = {None:allCarriers}
    data['setNode'] = {None:df_node.index.tolist()}
    data['setEdge'] = {None:df_edge.index.tolist()}
    data['setEdgeWithFlowComponents'] = {
        None:df_edge[df_edge['type']=='wellstream'].index.tolist()}
    data['setDevice'] = {None:df_device.index.tolist()}
    #data['setDevicemodel'] = {None:Multicarrier.devicemodel_inout().keys()}
    data['setHorizon'] = {None:range(planning_horizon)}
    data['setParameters'] = {None:df_parameters.index.tolist()}
    data['setProfile'] = {None:df_device['profile'].dropna().unique().tolist()}
    data['paramNode'] = to_dict_dropna(df_node)
    data['paramNodeCarrierHasSerialDevice'] = node_carrier_has_serialdevice
    data['paramNodeDevices'] = df_device.groupby('node').groups
#    data['paramDevice'] = df_device.to_dict(orient='index')
    data['paramDevice'] = to_dict_dropna(df_device)
    data['paramDeviceIsOnInitially'] = {k:v['isOn_init']
        for k,v in data['paramDevice'].items() if 'isOn_init' in v }
    data['paramDevicePowerInitially'] = {k:v['P_init']
        for k,v in data['paramDevice'].items() if 'P_init' in v }
    data['paramDeviceEnergyInitially'] = {k:v['E_init']
        for k,v in data['paramDevice'].items() if 'E_init' in v }
    data['paramEdge'] = to_dict_dropna(df_edge)
    data['paramNodeEdgesFrom'] = df_edge.groupby(['type','nodeFrom']).groups
    data['paramNodeEdgesTo'] = df_edge.groupby(['type','nodeTo']).groups
    #data['paramDevicemodel'] = devmodel_inout
    data['paramParameters'] = df_parameters['value'].to_dict()#orient='index')
    #unordered set error - but is this needed - better use dataframe diretly instead?
#    data['paramProfiles'] = df_profiles_forecast.loc[
#            range(planning_horizon),data['setProfile'][None]
#            ].T.stack().to_dict()
    data['paramCarriers'] = carrier_properties
    data['paramCoeffB'] = coeffB
    data['paramCoeffDA'] = coeffDA
    return data


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
    #b0 = np.asarray(_susceptancePu(baseZ))

    # MultiDiGraph to allow parallel lines
    G = nx.MultiDiGraph()
    edges = [(df_branch['nodeFrom'][i],
              df_branch['nodeTo'][i],
              i,{'i':i,'b':b[i]})
              for i in df_branch.index]
    G.add_nodes_from(df_node.index)
    G.add_edges_from(edges)
    A_incidence_matrix = -nx.incidence_matrix(G,oriented=True,
                                             nodelist=df_node.index,
                                             edgelist=edges).T
    # Diagonal matrix
    D = scipy.sparse.diags(-b,offsets=0)
    DA = D*A_incidence_matrix

    # Bf constructed from incidence matrix with branch susceptance
    # used as weight (this is quite fast)
    Bf = -nx.incidence_matrix(G,oriented=True,
                             nodelist=df_node.index,
                             edgelist=edges,
                             weight='b').T
    Bbus = A_incidence_matrix.T * Bf

    n_i = df_node.index
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
