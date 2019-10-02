import pyomo.environ as pyo
import pyomo.opt as pyopt
import pandas as pd
#import networkx as nx
import pydot
import matplotlib.pyplot as plt
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
    physical flow equations
        el: power flow equations
        gas:...linearised Weymouth equation (pressure/flow)
        (see note)


'''

model = pyo.AbstractModel()

# Sets
model.setCarrier = pyo.Set(doc="energy carrier")
model.setNode = pyo.Set()
model.setEdge1= pyo.Set(within=model.setCarrier*model.setNode*model.setNode)
model.setEdge= pyo.Set()
model.setDevice = pyo.Set()
model.setTerminal = pyo.Set(initialize=['in','out'])
model.setDevicemodel = pyo.Set()

# Parameters (input data)
model.paramNode = pyo.Param(model.setNode)
model.paramEdge = pyo.Param(model.setEdge)
model.paramDevice = pyo.Param(model.setDevice)
model.paramDeviceDispatchIn = pyo.Param(model.setDevice)
model.paramDeviceDispatchOut = pyo.Param(model.setDevice)
model.paramNodeCarrierHasSerialDevice = pyo.Param(model.setNode)
model.paramNodeDevices = pyo.Param(model.setNode)
model.paramNodeEdgesFrom = pyo.Param(model.setCarrier,model.setNode)
model.paramNodeEdgesTo = pyo.Param(model.setCarrier,model.setNode)
model.paramDevicemodel = pyo.Param(model.setDevicemodel)

# Variables
#model.varNodeVoltageAngle = pyo.Var(model.setNode,within=pyo.Reals)
model.varEdgePower = pyo.Var(model.setEdge,within=pyo.Reals)
model.varDevicePower = pyo.Var(model.setDevice,within=pyo.NonNegativeReals)
model.varDeviceIsOn = pyo.Var(model.setDevice,within=pyo.Binary)
model.varGasPressure = pyo.Var(model.setNode,model.setTerminal, 
                               within=pyo.NonNegativeReals)
model.varDeviceFlow = pyo.Var(model.setDevice,model.setCarrier,
                              model.setTerminal,within=pyo.Reals)
model.varTerminalFlow = pyo.Var(model.setNode,model.setCarrier,
                                within=pyo.Reals)


# Objective
def rule_objective(model):
    sumE = sum(model.varDevicePower[k]
               for k in model.setDevice)
    return sumE
model.objObjective = pyo.Objective(rule=rule_objective,sense=pyo.minimize)


def devicemodel_inout():
    inout = {
            'compressor_el':    {'in':['el','gas'],'out':['gas']},
            'compressor_gas':   {'in':['gas'],'out':['gas']},
            #'separator':        {'in':['el'],'out':[]},
            'sink_gas':         {'in':['gas'],'out':[]},
            'source_gas':       {'in':[],'out':['gas']},
            'gasturbine':       {'in':['gas'],'out':['el','heat']},
            'gen_el':           {'in':[],'out':['el']},
            'sink_el':          {'in':['el'],'out':[]},
            'sink_heat':        {'in':['heat'],'out':[]},
            'gasheater':        {'in':['gas'],'out':['heat']},
            'heatpump':         {'in':['el'],'out':['heat']},
            }
    return inout
    

def rule_devmodel_compressor_el(model,dev,i):
    if model.paramDevice[dev]['model'] != 'compressor_el':
        return pyo.Constraint.Skip
    if i==1:
        ''' power demand depends on gas pressure difference'''
        node = model.paramDevice[dev]['node']
        lhs = model.varDevicePower[dev]
        k = model.paramDevice[dev]['eta']
        rhs = k*(model.varGasPressure[(node,'out')]
                        -model.varGasPressure[(node,'in')])
        return (lhs==rhs)
    elif i==2:
        '''gas flow in equals gas flow out'''
        lhs = model.varDeviceFlow[dev,'gas','in']
        rhs = model.varDeviceFlow[dev,'gas','out']
        return (lhs==rhs)
    elif i==3:
        '''el in equals power demand'''
        lhs = model.varDeviceFlow[dev,'el','in']
        rhs = model.varDevicePower[dev]
        return (lhs==rhs)
model.constrDevice_compressor_el = pyo.Constraint(model.setDevice,
          pyo.RangeSet(1,3),
          rule=rule_devmodel_compressor_el)

def rule_devmodel_compressor_gas(model,dev,i):
    if model.paramDevice[dev]['model'] != 'compressor_gas':
        return pyo.Constraint.Skip
    if i==1:
        '''compressor gas demand related to pressure difference'''
        node = model.paramDevice[dev]['node']
        k = model.paramDevice[dev]['eta']
        lhs = model.varDevicePower[dev]
        rhs = (k*(model.varGasPressure[(node,'out')]
                    -model.varGasPressure[(node,'in')]) )
        return lhs==rhs
    elif i==2:
        '''matter conservation'''
        node = model.paramDevice[dev]['node']
        lhs = model.varDeviceFlow[dev,'gas','out']
        rhs = model.varDeviceFlow[dev,'gas','in'] - model.varDevicePower[dev]
        return lhs==rhs
model.constrDevice_compressor_gas = pyo.Constraint(model.setDevice,
          pyo.RangeSet(1,2),
          rule=rule_devmodel_compressor_gas)
        

def rule_devmodel_sink_gas(model,dev):
    if model.paramDevice[dev]['model'] != 'sink_gas':
        return pyo.Constraint.Skip
    lhs = model.varDeviceFlow[dev,'gas','in']
    rhs = model.varDevicePower[dev]
    return (lhs==rhs)
model.constrDevice_sink_gas = pyo.Constraint(model.setDevice,
          rule=rule_devmodel_sink_gas)

def rule_devmodel_source_gas(model,dev,i):
    if model.paramDevice[dev]['model'] != 'source_gas':
        return pyo.Constraint.Skip
    if i==1:
        lhs = model.varDeviceFlow[dev,'gas','out']
        rhs = model.varDevicePower[dev]
        return (lhs==rhs)
    elif i==2:
        node = model.paramDevice[dev]['node']
        lhs = model.varGasPressure[(node,'out')]
        #TODO set source pressure from input data
        rhs = 200
        #return pyo.Constraint.Skip
        return (lhs==rhs)
model.constrDevice_source_gas = pyo.Constraint(model.setDevice,
          pyo.RangeSet(1,2),
          rule=rule_devmodel_source_gas)

def rule_devmodel_gasheater(model,dev,i):
    if model.paramDevice[dev]['model'] != 'gasheater':
        return pyo.Constraint.Skip
    if i==1:
        # gas in = device power * efficiency
        lhs = model.varDeviceFlow[dev,'gas','in']*model.paramDevice[dev]['eta']
        rhs = model.varDevicePower[dev]
        return (lhs==rhs)
    elif i==2:
        # heat out = device power
        lhs = model.varDeviceFlow[dev,'heat','out']
        rhs = model.varDevicePower[dev]
        return (lhs==rhs)
model.constrDevice_gasheater = pyo.Constraint(model.setDevice,
          pyo.RangeSet(1,2),
          rule=rule_devmodel_gasheater)

def rule_devmodel_heatpump(model,dev,i):
    if model.paramDevice[dev]['model'] != 'heatpump':
        return pyo.Constraint.Skip
    if i==1:
        # device power = el in * efficiency
        lhs = model.varDevicePower[dev]
        rhs = model.varDeviceFlow[dev,'el','in']*model.paramDevice[dev]['eta']
        return (lhs==rhs)
    elif i==2:
        # heat out = device power
        lhs = model.varDeviceFlow[dev,'heat','out']
        rhs = model.varDevicePower[dev]
        return (lhs==rhs)
model.constrDevice_heatpump = pyo.Constraint(model.setDevice,
          pyo.RangeSet(1,2),
          rule=rule_devmodel_heatpump)


print("TODO: gas turbine power vs heat output")
def rule_devmodel_gasturbine(model,dev,i):
    if model.paramDevice[dev]['model'] != 'gasturbine':
        return pyo.Constraint.Skip
    if i==1:
        '''turbine power vs gas fuel usage'''       
        # fuel consumption is a linear function of power output
        # fuel = A + B*power
        # => efficiency = power/(A+B*power)
        A = model.paramDevice[dev]['fuelA']
        B = model.paramDevice[dev]['fuelB']
        Pmax = model.paramDevice[dev]['Pmax']
        lhs = model.varDeviceFlow[dev,'gas','in']/Pmax
        rhs = A*model.varDeviceIsOn[dev] + B*model.varDevicePower[dev]/Pmax
        return lhs==rhs
    elif i==2:
        '''turbine power = power output'''
        lhs = model.varDeviceFlow[dev,'el','out']
        rhs = model.varDevicePower[dev]
        return lhs==rhs
    elif i==3:
        '''power only if turbine is on'''
        lhs = model.varDevicePower[dev]
        rhs = model.varDeviceIsOn[dev]*model.paramDevice[dev]['Pmax']
        return lhs <= rhs
    elif i==4:
        '''turbine power = heat output * heat fraction'''
        lhs = model.varDeviceFlow[dev,'heat','out']
        rhs = model.varDevicePower[dev]*model.paramDevice[dev]['heat']
        return lhs==rhs
model.constrDevice_gasturbine = pyo.Constraint(model.setDevice,
          pyo.RangeSet(1,4),
          rule=rule_devmodel_gasturbine)

def plotGasTurbineEfficiency(model,dev):
    fuelA = model.paramDevice[dev]['fuelA']
    fuelB = model.paramDevice[dev]['fuelB']
    x_pow = pd.np.linspace(0,1,50)
    y_fuel = fuelA + fuelB*x_pow
    plt.figure()
    plt.subplot(3,1,1)
    plt.ylabel("Fuel usage")
    plt.plot(x_pow,y_fuel)
    plt.ylim(bottom=0,top=20)
    plt.subplot(3,1,2)
    plt.ylabel("Specific fuel usage")
    plt.xlabel("Power output")
    plt.plot(x_pow,y_fuel/x_pow)
    plt.subplot(3,1,3)
    plt.ylabel("Efficiency")
    plt.xlabel("Power output")
    plt.plot(x_pow,x_pow/y_fuel)
    
    
    

def rule_devmodel_gen_el(model,dev):
    if model.paramDevice[dev]['model'] != 'gen_el':
        return pyo.Constraint.Skip
    '''turbine power = power infeed'''
    lhs = model.varDeviceFlow[dev,'el','out']
    rhs = model.varDevicePower[dev]
    return lhs==rhs
model.constrDevice_gen_el = pyo.Constraint(model.setDevice,
          rule=rule_devmodel_gen_el)

def rule_devmodel_sink_el(model,dev):
    if model.paramDevice[dev]['model'] != 'sink_el':
        return pyo.Constraint.Skip
    '''sink power = power out'''
    lhs = model.varDeviceFlow[dev,'el','in']
    rhs = model.varDevicePower[dev]
    return lhs==rhs
model.constrDevice_sink_el = pyo.Constraint(model.setDevice,
          rule=rule_devmodel_sink_el)

def rule_devmodel_sink_heat(model,dev):
    if model.paramDevice[dev]['model'] != 'sink_heat':
        return pyo.Constraint.Skip
    '''sink heat = heat out'''
    lhs = model.varDeviceFlow[dev,'heat','in']
    rhs = model.varDevicePower[dev]
    return lhs==rhs
model.constrDevice_sink_heat = pyo.Constraint(model.setDevice,
          rule=rule_devmodel_sink_heat)



def rule_terminalEnergyBalance(model,carrier,node,terminal):
    ''' energy balance at in and out terminals
    "in" terminal: flow into terminal is positive
    "out" terminal: flow out of terminal is positive
    '''
    
    Pinj = 0
           
    # devices:
    if (node in model.paramNodeDevices):
        for dev in model.paramNodeDevices[node]:
            # Power into terminal:
            dev_model = model.paramDevice[dev]['model']
            if pd.isna(dev_model):
                print("No device model specified - using dispatch factor")  
            elif dev_model in model.paramDevicemodel:
                #print("carrier:{},node:{},terminal:{},model:{}"
                #      .format(carrier,node,terminal,dev_model))
                if carrier in model.paramDevicemodel[dev_model][terminal]:
                    Pinj -= model.varDeviceFlow[dev,carrier,terminal]
            else:
                raise Exception("Undefined device model ({})".format(dev_model))

    # connect terminals:
    if not model.paramNodeCarrierHasSerialDevice[node][carrier]:
        Pinj -= model.varTerminalFlow[node,carrier]

            
    # edges:
    if (carrier,node) in model.paramNodeEdgesTo and (terminal=='in'):
        for edg in model.paramNodeEdgesTo[(carrier,node)]:
            # power into node from edge
            Pinj += (model.varEdgePower[edg])
    elif (carrier,node) in model.paramNodeEdgesFrom and (terminal=='out'):
        for edg in model.paramNodeEdgesFrom[(carrier,node)]:
            # power out of node into edge
            Pinj += (model.varEdgePower[edg])
    
    if not(Pinj is 0):
        return (Pinj==0)
    else:
        return pyo.Constraint.Skip
model.constrTerminalEnergyBalance = pyo.Constraint(model.setCarrier,
              model.setNode, model.setTerminal,
              rule=rule_terminalEnergyBalance)


#def rule_nodeEnergyBalance(model,carrier,node,terminal):
#    Pinj = 0
#    
#    if ((node in model.paramNodeNontrivial) and 
#        (model.paramNodeNontrivial[node][carrier])):
#            # (carrier,node) is non-trivial
#        terminalOut = (terminal=='out')
#        #print("Non-trivial node: ({},{})".format(node,carrier))
#    else:
#        #print("Trivial node: ({},{})".format(node,carrier))
#        # trivial node
#        # add constraint only for 'in' terminal, include all devices
#        # and edges in single constraint
#        if (terminal=='out'):
#            # single constraint associated with in-terminal, so skip this one
#            return pyo.Constraint.Skip
#        # setting this True ensures that out-terminal connected devices and
#        # edges are merged with in-terminal
#        terminalOut = True
#        
#    # devices:
#    if (node in model.paramNodeDevices):
#        # TODO: different energy demand models
#        # e.g. pump el demand is proportional to pressure difference
#        for dev in model.paramNodeDevices[node]:
#            dev_model = model.paramDevice[dev]['model']
#            if dev_model=='compressor_el':
#                print("{}, Node:{}, Device model={}".format(carrier,node,dev_model))
#                Pinj += dev_model_compressor_el(model,dev,carrier,node,terminal)
#            else:
#                # standard dispatchmodel
#                if (terminal=='in'):
#                    # pos dispatch= power out from grid, i.e .Pinj negative
#                    Pinj -= (model.varDevicePower[dev]
#                            *model.paramDeviceDispatchIn[dev][carrier])
#                if terminalOut:
#                    # pos dispatch=power into grid, i.e. Pinj positive
#                    Pinj += (model.varDevicePower[dev]
#                            *model.paramDeviceDispatchOut[dev][carrier])
#    
#            
#    # edges:
#    if (carrier,node) in model.paramNodeEdgesTo and (terminal=='in'):
#        for edg in model.paramNodeEdgesTo[(carrier,node)]:
#            # power into node from edge
#            Pinj += (model.varEdgePower[edg])
#    if (carrier,node) in model.paramNodeEdgesFrom and terminalOut:
#        for edg in model.paramNodeEdgesFrom[(carrier,node)]:
#            # power out of node into edge
#            Pinj -= (model.varEdgePower[edg])
#    
#    if not(Pinj is 0):
#        return (Pinj==0)
#    else:
#        return pyo.Constraint.Skip
#model.constrNodePowerBalance = pyo.Constraint(model.setCarrier,model.setNode,
#                                              model.setTerminal,
#                                              rule=rule_nodeEnergyBalance)

def rule_gasPressureAndFlow(model,edge):
    if model.paramEdge[edge]['type'] != 'gas':
        return pyo.Constraint.Skip
    n_from = model.paramEdge[edge]['nodeFrom']
    n_to = model.paramEdge[edge]['nodeTo']
    p_from = model.varGasPressure[(n_from,'out')]
    p_to = model.varGasPressure[(n_to,'in')]
    exp_s = 1 # elevation factor
    p0_from = model.paramNode[n_from]['gaspressure_out']
    p0_to = model.paramNode[n_to]['gaspressure_in']
    k = model.paramEdge[edge]['gasflow_k']
    coeff = k*(p0_from**2-exp_s*p0_to**2)**(-1/2)
    lhs = model.varEdgePower[edge]
    rhs = coeff*(p0_from*p_from - exp_s*p0_to*p_to)
    print(n_from,n_to,p0_from,p0_to,coeff)
    return (lhs==rhs)
model.constrGasPressureAndFlow = pyo.Constraint(model.setEdge,
                                                rule=rule_gasPressureAndFlow)

def rule_gasPressureAtNode(model,node):
    #if not model.paramNodeNontrivial[node]['gas']:
    if not model.paramNodeCarrierHasSerialDevice[node]['gas']:
        # trivial connection. pressure out=pressure in
        expr = (model.varGasPressure[(node,'out')]
                == model.varGasPressure[(node,'in')] )
        return expr
    else:
        return pyo.Constraint.Skip    
model.constrGasPressureAtNode = pyo.Constraint(model.setNode,
                                               rule=rule_gasPressureAtNode)

# TODO: Set bounds from input data
def rule_gasPressureBounds(model,node,t):
    col = 'gaspressure_{}'.format(t)
    nom_p = model.paramNode[node][col]
    lb = nom_p*0.8
    ub = nom_p*1.2
    return (lb,model.varGasPressure[(node,t)],ub)
model.constrGasPressureBounds = pyo.Constraint(model.setNode,model.setTerminal,
                                               rule=rule_gasPressureBounds)
    

def rule_devicePmax(model,dev):
    return (model.paramDevice[dev]['Pmin'] 
            <= model.varDevicePower[dev] 
            <= model.paramDevice[dev]['Pmax'])
model.constrDevicePmax = pyo.Constraint(model.setDevice,rule=rule_devicePmax)

def rule_devicePmin(model,dev):
    return (model.varDevicePower[dev] >= model.paramDevice[dev]['Pmin'])
model.constrDevicePmin = pyo.Constraint(model.setDevice,rule=rule_devicePmin)

def rule_edgePmaxmin(model,edge):
    return (-model.paramEdge[edge]['capacity'] 
            <= model.varEdgePower[edge] 
            <= model.paramEdge[edge]['capacity'])
model.constrEdgeBounds = pyo.Constraint(model.setEdge,rule=rule_edgePmaxmin)



def plotNetworkCombined(model,filename='pydotCombined.png',only_carrier=None):
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
            label_in = carrier+'_in'
            label_out= carrier+'_out'
            if carrier=='gas':
                label_in +=':{:3.1f}'.format(pyo.value(instance.varGasPressure[n_id,'in']))
                label_out +=':{:3.1f}'.format(pyo.value(instance.varGasPressure[n_id,'out']))
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
                edgelabel = '{:.2f}'.format(pyo.value(model.varEdgePower[i]))
                dotG.add_edge(pydot.Edge(src=e['nodeFrom']+'_'+carrier+'_out',
                                         dst=e['nodeTo']+'_'+carrier+'_in',
                                         color=col['e'][carrier],
                                         fontcolor=col['e'][carrier],
                                         label=edgelabel))
    
    # plot devices and device connections:
    for n,devs in model.paramNodeDevices.items():
        for d in devs:
            dev_model = model.paramDevice[d]['model']
            p_dev = pyo.value(model.varDevicePower[d])
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
                f_in = pyo.value(model.varDeviceFlow[d,carrier,'in'])
                dotG.add_edge(pydot.Edge(dst=d,src=n+'_'+carrier+'_in',
                     color=col['e'][carrier],
                     fontcolor=col['e'][carrier],
                     label="{:.2f}".format(f_in)))
            for carrier in carriers_out_lim:
                f_out = pyo.value(model.varDeviceFlow[d,carrier,'out'])
                dotG.add_edge(pydot.Edge(dst=n+'_'+carrier+'_out',src=d,
                     color=col['e'][carrier],
                     fontcolor=col['e'][carrier],
                     label="{:.2f}".format(f_out)))
    
    # plot terminal in-out links:
    for n in model.setNode:
        for carrier in carriers:
             if not model.paramNodeCarrierHasSerialDevice[n][carrier]:
                flow = pyo.value(model.varTerminalFlow[n,carrier])
                dotG.add_edge(pydot.Edge(dst=n+'_'+carrier+'_out',
                                         src=n+'_'+carrier+'_in',
                     color='"{0}:invis:{0}"'.format(col['e'][carrier]),
                     label='{:.2f}'.format(flow),fontcolor=col['e'][carrier]))
                     #arrowhead='none'))

    #dotG.write_dot('pydotCombinedNEW.dot',prog='dot')                
    dotG.write_png(filename,prog='dot')    


#########################################################################


# Input data
df_node = pd.read_excel("data_example.xlsx",sheet_name="node")
df_edge = pd.read_excel("data_example.xlsx",sheet_name="edge")
df_device = pd.read_excel("data_example.xlsx",sheet_name="device")
#df_devicemodel = pd.read_excel("data_example.xlsx",sheet_name="devicemodel")
#
#df_devicemodel['in'] =df_devicemodel['in'].str.split(',')
#df_devicemodel['out'] =df_devicemodel['out'].str.split(',')
##replace nan by []:
#for row in df_devicemodel.loc[df_devicemodel['in'].isnull(), 'in'].index:
#    df_devicemodel.at[row, 'in'] = []
#for row in df_devicemodel.loc[df_devicemodel['out'].isnull(), 'out'].index:
#    df_devicemodel.at[row, 'out'] = []

df_edge = df_edge[df_edge['include']==1]
df_device = df_device[df_device['include']==1]

#into node:
cols_in = {col:col.split("in_")[1] 
           for col in df_device.columns if "in_" in col }
dispatch_in = df_device[list(cols_in.keys())].rename(columns=cols_in).fillna(0)
#out of node:
cols_out = {col:col.split("out_")[1] 
           for col in df_device.columns if "out_" in col }
dispatch_out = df_device[list(cols_out.keys())].rename(columns=cols_out).fillna(0)

df_deviceR = df_device.drop(cols_in,axis=1).drop(cols_out,axis=1)

## find nodes where no devices connect in-out terminals:
## (node is non-trivial if any device connects both in and out)
#dev_nontrivial = ((dispatch_in!=0) & (dispatch_out!=0))
#node_nontrivial = pd.concat([df_device[['node']],
#                             dev_nontrivial],axis=1).groupby('node').any()

node_devices = df_device.groupby('node').groups
devmodel_inout = devicemodel_inout()
#allCarriers = df_edge['type'].unique().tolist()
allCarriers = ['el','gas','heat']
node_carrier_has_serialdevice = {}
for n,devs in node_devices.items():
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
        
    
    
data = {}
data['setCarrier'] = {None:allCarriers}
data['setNode'] = {None:df_node['id'].tolist()}
data['setEdge'] = {None: df_edge.index.tolist()}
data['setDevice'] = {None:df_device.index.tolist()}
data['setDevicemodel'] = {None:devmodel_inout.keys()}
data['paramDeviceDispatchIn'] = dispatch_in.to_dict(orient='index') 
data['paramDeviceDispatchOut'] = dispatch_out.to_dict(orient='index') 
data['paramNode'] = df_node.set_index('id').to_dict(orient='index')
#data['paramNodeNontrivial'] = node_nontrivial.to_dict(orient='index') 
data['paramNodeCarrierHasSerialDevice'] = node_carrier_has_serialdevice
data['paramNodeDevices'] = df_device.groupby('node').groups
data['paramDevice'] = df_deviceR.to_dict(orient='index')
data['paramEdge'] = df_edge.to_dict(orient='index')
data['paramNodeEdgesFrom'] = df_edge.groupby(['type','nodeFrom']).groups
data['paramNodeEdgesTo'] = df_edge.groupby(['type','nodeTo']).groups
#data['paramDevicemodel'] = df_devicemodel.set_index('id').to_dict(orient='index')
data['paramDevicemodel'] = devmodel_inout

instance = model.create_instance(data={None:data},name='MultiCarrier')
instance.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
instance.pprint(filename='model.txt')

opt = pyo.SolverFactory('gurobi')

sol = opt.solve(instance) 

sol.write_yaml()

if ((sol.solver.status == pyopt.SolverStatus.ok) and 
   (sol.solver.termination_condition == pyopt.TerminationCondition.optimal)):
    print("Solved OK")
elif (sol.solver.termination_condition == pyopt.TerminationCondition.infeasible):
    raise Exception("Infeasible solution")
else:
    # Something else is wrong
    print("Solver Status:{}".format(sol.solver.status))

print("\nSOLUTION - devicePower:")
for k in instance.varDevicePower.keys():
    print("  {}: {}".format(k,instance.varDevicePower[k].value))

print("\nSOLUTION - edgePower:")
for k in instance.varEdgePower.keys():
    power = instance.varEdgePower[k].value
    print("  {}: {}".format(k,power))
    df_edge.loc[k,'edgePower'] = power

print("\nSOLUTION - gasPressure:")
for k,v in instance.varGasPressure.get_values().items():
    pressure = v #pyo.value(v)
    print("  {}: {}".format(k,pressure))
    #df_edge.loc[k,'gasPressure'] = pressure

# display all duals
#print ("Duals")
#for c in instance.component_objects(pyo.Constraint, active=True):
#    print ("   Constraint",c)
#    for index in c:
#        print ("      ", index, instance.dual[c[index]])


plotNetworkCombined(instance)
plotNetworkCombined(instance,only_carrier='el')
plotNetworkCombined(instance,only_carrier='gas')
plotNetworkCombined(instance,only_carrier='heat')

#
#df_edge['edgePower']
#graph={}
##pos = {v['id']:(v['coord_x'],v['coord_y']) for i,v in df_node.iterrows()}
#fixedpos = {v['id']+'_in':(v['coord_x'],v['coord_y']) for i,v in df_node.iterrows()}
#pos = fixedpos.copy()
#pos.update({v['id']+'_out':(v['coord_x'],v['coord_y']+0.1) for i,v in df_node.iterrows()})
#labels_edge = df_edge.set_index(['nodeFrom','nodeTo'])['edgePower'].to_dict()
#




