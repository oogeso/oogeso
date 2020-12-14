"""This module contains helper functions to compute things.
See milp_problem.py for optimisation model formulation"""

import logging
import pyomo.environ as pyo
import pyomo.opt as pyopt
import numpy as np
import scipy


_devmodels = {
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
    'source_el':        {'in':[],'out':['el']},
    'heatpump':         {'in':['el'],'out':['heat']},
    'storage_el':       {'in':['el'], 'out':['el']},
    'pump_oil':         {'in':['oil','el'], 'out':['oil'],
                         'serial':['oil']},
    'pump_water':       {'in':['water','el'],'out':['water'],
                         'serial':['water']},
    'pump_wellstream':  {'in':['wellstream','el'],'out':['wellstream'],
                         'serial':['wellstream']},
    'electrolyser':     {'in':['el'], 'out':['hydrogen','heat']},
    'fuelcell':         {'in':['hydrogen'],'out':['el','heat']},
    'storage_hydrogen': {'in':['hydrogen'],'out':['hydrogen']},
    }

def devicemodel_inout():
    return _devmodels

models_with_storage = ['storage_el','well_injection','storage_hydrogen']


def getDevicePower(model,dev,t):
    '''returns the variable that defines device power (depends on model)
    used for ramp rate limits, and print/plot'''
    devmodel = model.paramDevice[dev]['model']
    if devmodel in ['gasturbine','source_el','fuelcell']:
        devpower = model.varDeviceFlow[dev,'el','out',t]
    elif devmodel in ['sink_el','pump_water','pump_oil','compressor_el',
                        'electrolyser']:
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
        # this represents energy taken out of the storage
        eta = model.paramDevice[dev]['eta']
        devpower = -(model.varDeviceFlow[dev,'el','in',t]*eta
                    - model.varDeviceFlow[dev,'el','out',t]/eta)
    else:
        # TODO: Complete this
        # no need to define devpower for other devices
        #raise Exception("Undefined power variable for {}".format(devmodel))
        devpower = 0
    return devpower


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
                          'source_water','source_oil','electrolyser',
                          'fuelcell','storage_hydrogen']:
            # no CO2 emission contribution
            thisCO2 = 0
        else:
            raise NotImplementedError(
                "CO2 calculation for {} not implemented".format(devmodel))
        sumCO2 = sumCO2 + thisCO2

    # Average per s
    sumCO2 = sumCO2/len(timesteps)
    return sumCO2


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
def compute_operatingCosts(model):
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
                varP = getDevicePower(model,dev,t)
                #delta_t = model.paramParameters['time_delta_minutes']*60
                #energy_in_dt = varP*delta_t
                #sumCost += opcost*energy_in_dt
                sumCost += opcost*varP
                #logging.info('opcost={}, e={}, scost={}'.format(opcost,energy_in_dt,opcost*energy_in_dt))
    # average per sec
    sumCost = sumCost/len(timesteps)
    return sumCost


def compute_costForDepletedStorage(model):
    '''term in objective function to discourage depleting battery,
    making sure it is used only when required'''
    storCost = 0
    for dev in model.setDevice:
        devmodel = model.paramDevice[dev]['model']
        if devmodel == 'storage_el':
            stor_cost=0
            if 'Ecost' in model.paramDevice[dev]:
                stor_cost = model.paramDevice[dev]['Ecost']
            Emax = model.paramDevice[dev]['Emax']
            for t in model.setHorizon:
                varE = model.varDeviceStorageEnergy[dev,t]
                storCost += stor_cost*(Emax-varE)
        elif devmodel == 'storage_hydrogen':
            # cost if storage level at end of optimisation deviates from
            # target profile (user input based on expectations)
            deviation = model.varDeviceStorageDeviationFromTarget[dev]
            stor_cost = model.paramDevice[dev]['Ecost']
            storCost += stor_cost*deviation
    return storCost


def compute_exportRevenue(model,carriers=None,timesteps=None):
    '''revenue from exported oil and gas - average per sec ($/s)'''
    if carriers is None:
        carriers = model.setCarrier
    if timesteps is None:
        timesteps = model.setHorizon

    inouts = _devmodels
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


def compute_oilgas_export(model,timesteps=None):
    '''Export volume (Sm3oe/s)'''
    if timesteps is None:
        timesteps = model.setHorizon

    carriers = model.setCarrier
    export_node = model.paramParameters['export_node']
    export_devs = model.paramNodeDevices[export_node]
    inouts = _devmodels
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


def compute_CO2_intensity(model,timesteps=None):
    '''CO2 emission per exported oil/gas (kgCO2/Sm3oe)'''
    if timesteps is None:
        timesteps = model.setHorizon

    co2_kg_per_time = compute_CO2(
            model,devices=None,timesteps=timesteps)
    flow_oilequivalents_m3_per_time = compute_oilgas_export(
            model,timesteps)
    if pyo.value(flow_oilequivalents_m3_per_time)!=0:
        co2intensity = co2_kg_per_time/flow_oilequivalents_m3_per_time
    if pyo.value(flow_oilequivalents_m3_per_time)==0:
        logging.warning("zero export, so co2 intensity set to NAN")
        co2intensity = np.nan

    return co2intensity


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
        exp_s,k = compute_exps_and_k(model,edge,carrier)
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


def compute_elReserve(model,t,exclude_device=None):
    '''compute non-used generator capacity (and available loadflex)
    This is reserve to cope with forecast errors, e.g. because of wind
    variation or motor start-up
    (does not include load reduction yet)

    exclue_device : str (default None)
        compute reserve by devices excluding this one
    '''
    #alldevs = model.setDevice
    alldevs = [d for d in model.setDevice if d!=exclude_device]
    # relevant devices are devices with el output or input
    #inout = Multicarrier.devicemodel_inout()
    inout = _devmodels
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
            # Loads (only consider if resere factor has been set)
            if ('reserve_factor' in model.paramDevice[d]):
                # load reduction possible
                f_lr = model.paramDevice[d]['reserve_factor']
                loadreduction += f_lr*model.varDeviceFlow[d,'el','in',t]
    res_dev = (cap_avail-p_generating) + loadreduction
    #TODO: include flexible loads (that can absorve variations)
    res_dev = (cap_avail-p_generating)
    return res_dev


def OBSOLETE_compute_elBackup(model,t,exclude_device=None):
    '''Compute available reserve power that can take over in case
    of a fault. Consists of:
    1. generator unused capacity (el out)
    2. sheddable load (el in)'''
    alldevs = model.setDevice
    # relevant devices are devices with el output or input
    inout = _devmodels
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
