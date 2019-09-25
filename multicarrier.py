import pyomo.environ as pyo
import pyomo.opt as pyopt
import pandas as pd
import networkx as nx
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

network consisting of nodes and edges
devices: node_in, node_out, dispatch factors, power (var)

TODO:
    physical flow equations
        el: power flow equations
        gas:...linearised Weymouth equation (pressure/flow)
        (see note)

    natural pressure (well)

'''










model = pyo.AbstractModel()

# Sets
model.setCarrier = pyo.Set(doc="energy carrier")
model.setNode = pyo.Set()
model.setEdge1= pyo.Set(within=model.setCarrier*model.setNode*model.setNode)
model.setEdge= pyo.Set()
model.setDevice = pyo.Set()
model.setTerminal = pyo.Set(initialize=['in','out'])

# Parameters (input data)
model.paramEdge = pyo.Param(model.setEdge)
model.paramDevice = pyo.Param(model.setDevice)
model.paramDeviceDispatchIn = pyo.Param(model.setDevice)
model.paramDeviceDispatchOut = pyo.Param(model.setDevice)
model.paramNodeNontrivial = pyo.Param(model.setNode)
model.paramNodeDevices = pyo.Param(model.setNode)
model.paramNodeEdgesFrom = pyo.Param(model.setCarrier,model.setNode)
model.paramNodeEdgesTo = pyo.Param(model.setCarrier,model.setNode)

# Variables
#model.varNodeVoltageAngle = pyo.Var(model.setNode,within=pyo.Reals)
model.varEdgePower = pyo.Var(model.setEdge,within=pyo.Reals)
model.varEdgePower2 = pyo.Var(within=pyo.Reals)
model.varDevicePower = pyo.Var(model.setDevice,within=pyo.NonNegativeReals)
model.varGasPressure = pyo.Var(model.setNode,model.setTerminal, 
                               within=pyo.NonNegativeReals)


# Objective
def rule_objective(model):
    sumE = sum(model.varDevicePower[k]
               for k in model.setDevice)
    return sumE
model.objObjective = pyo.Objective(rule=rule_objective,sense=pyo.minimize)

# Constraints
def rule_nodeEnergyBalance(model,carrier,node,terminal):
    Pinj = 0
    
    if ((node in model.paramNodeNontrivial) and 
        (model.paramNodeNontrivial[node][carrier])):
            # (carrier,node) is non-trivial
        terminalOut = (terminal=='out')
        print("Non-trivial node: ({},{})".format(node,carrier))
    else:
        print("Trivial node: ({},{})".format(node,carrier))
        # trivial node
        # add constraint only for 'in' terminal, include all devices
        # and edges in single constraint
        if (terminal=='out'):
            # single constraint associated with in-terminal, so skip this one
            return pyo.Constraint.Skip
        # setting this True ensures that out-terminal connected devices and
        # edges are merged with in-terminal
        terminalOut = True
        
    # devices:
    if (node in model.paramNodeDevices):
        if (terminal=='in'):
            for dev in model.paramNodeDevices[node]:
                # pos dispatch= power out from grid, i.e .Pinj negative
                Pinj -= (model.varDevicePower[dev]
                        *model.paramDeviceDispatchIn[dev][carrier])
        if terminalOut:
            for dev in model.paramNodeDevices[node]:
                # pos dispatch=power into grid, i.e. Pinj positive
                Pinj += (model.varDevicePower[dev]
                        *model.paramDeviceDispatchOut[dev][carrier])
    
            
    # edges:
    if (carrier,node) in model.paramNodeEdgesTo and (terminal=='in'):
        for edg in model.paramNodeEdgesTo[(carrier,node)]:
            # power into node from edge
            Pinj += (model.varEdgePower[edg])
    if (carrier,node) in model.paramNodeEdgesFrom and terminalOut:
        for edg in model.paramNodeEdgesFrom[(carrier,node)]:
            # power out of node into edge
            Pinj -= (model.varEdgePower[edg])
    
    if not(Pinj is 0):
        return (Pinj==0)
    else:
        return pyo.Constraint.Skip
model.constrNodePowerBalance = pyo.Constraint(model.setCarrier,model.setNode,
                                              model.setTerminal,
                                              rule=rule_nodeEnergyBalance)

def rule_gasPressureAndFlow(model,edge):
    if model.paramEdge[edge]['type'] != 'gas':
        return pyo.Constraint.Skip
    n_from = model.paramEdge[edge]['nodeFrom']
    n_to = model.paramEdge[edge]['nodeTo']
    p_from = model.varGasPressure[(n_from,'out')]
    p_to = model.varGasPressure[(n_to,'in')]
    exp_s = 1 # elevation factor
    p0_from = model.paramEdge[edge]['pressureFrom']
    p0_to = model.paramEdge[edge]['pressureTo']
    k = model.paramEdge[edge]['gasflow_k']
    eq_lhs = model.varEdgePower[edge]
    eq_rhs = k*(p0_from**2-exp_s*p0_to**2)**(-1/2)*(
            p0_from*p_from - exp_s*p0_to*p_to)
    expr = (eq_lhs==eq_rhs)
    return expr
#model.constrGasPressureAndFlow = pyo.Constraint(model.setEdge,
#                                                rule=rule_gasPressureAndFlow)

def rule_gasPressureAtNode(model,node):
    if not model.paramNodeNontrivial[node]['gas']:
        # trivial connection. pressure out=pressure in
        expr = (model.varGasPressure[(node,'out')]
                == model.varGasPressure[(node,'in')] )
        return expr
    else:
        p_ratio = None
        p_ratio_last = None
        for d in model.paramNodeDevices[node]:
            if ((model.paramDeviceDispatchIn[d]['gas']>0) and
                (model.paramDeviceDispatchOut[d]['gas']>0)):
                p_ratio = 1.5 
                # pressure ratio must be the same for all devices at same node
                if (not(p_ratio_last is None) and (p_ratio != p_ratio_last)):
                    raise Exception("Inconsistent input data: Pressure ratio"
                                    " devie {}".format(d))
        if p_ratio is None:
            # There is no gas in-out device at this node
            expr = pyo.Constraint.Skip
        else:    
            expr = (model.varGasPressure[(node,'out')]
                    == p_ratio*model.varGasPressure[(node,'in')] )
        return expr
    
model.constrGasPressureAtNode = pyo.Constraint(model.setNode,
                                               rule=rule_gasPressureAtNode)


def rule_devicePmax(model,dev):
    return (model.paramDevice[dev]['Pmin'] 
            <= model.varDevicePower[dev] 
            <= model.paramDevice[dev]['Pmax'])
model.constrDevicePmax = pyo.Constraint(model.setDevice,rule=rule_devicePmax)

def rule_devicePmin(model,dev):
    return (model.varDevicePower[dev] >= model.paramDevice[dev]['Pmin'])
model.constrDevicePmin = pyo.Constraint(model.setDevice,rule=rule_devicePmin)

print("TODO: Infeasible if including devicePower max/min")





# Input data
df_node = pd.read_excel("data_example.xlsx",sheet_name="node")
df_edge = pd.read_excel("data_example.xlsx",sheet_name="edge")
df_device = pd.read_excel("data_example.xlsx",sheet_name="device")

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

# find nodes where no devices connect in-out terminals:
# (node is non-trivial if any device connects both in and out)
dev_nontrivial = ((dispatch_in!=0) & (dispatch_out!=0))
node_nontrivial = pd.concat([df_device[['node']],
                             dev_nontrivial],axis=1).groupby('node').any()

data = {}
data['setCarrier'] = {None:df_edge['type'].unique().tolist()}
data['setNode'] = {None:df_node['id'].tolist()}
data['setEdge'] = {None: df_edge.index.tolist()}
data['setDevice'] = {None:df_device.index.tolist()}
data['paramDeviceDispatchIn'] = dispatch_in.to_dict(orient='index') 
data['paramDeviceDispatchOut'] = dispatch_out.to_dict(orient='index') 
data['paramNodeNontrivial'] = node_nontrivial.to_dict(orient='index') 
data['paramNodeDevices'] = df_device.groupby('node').groups
data['paramDevice'] = df_deviceR.to_dict(orient='index')
data['paramEdge'] = df_edge.to_dict(orient='index')
data['paramNodeEdgesFrom'] = df_edge.groupby(['type','nodeFrom']).groups
data['paramNodeEdgesTo'] = df_edge.groupby(['type','nodeTo']).groups


instance = model.create_instance(data={None:data},name='MultiCarrier')
instance.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
instance.pprint(filename='model.txt')

opt = pyo.SolverFactory('cbc')

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


df_edge['edgePower']
graph={}
#pos = {v['id']:(v['coord_x'],v['coord_y']) for i,v in df_node.iterrows()}
fixedpos = {v['id']+'_in':(v['coord_x'],v['coord_y']) for i,v in df_node.iterrows()}
pos = fixedpos.copy()
pos.update({v['id']+'_out':(v['coord_x'],v['coord_y']+0.1) for i,v in df_node.iterrows()})
labels_edge = df_edge.set_index(['nodeFrom','nodeTo'])['edgePower'].to_dict()

def plotNetwork(data,df_node,model):
    for carrier in data['setCarrier'][None]:
        cluster = {}
        dotG = pydot.Dot(graph_type='digraph',overlap=False)
        for i,n in df_node.iterrows():
            n_id = n['id']
            cluster[n_id] = pydot.Cluster(graph_name=n['id'],label=n['id'])
            cluster[n_id].add_node(pydot.Node(name=n['id']+'_in',
                                   pos='"{},{}"'.format(n['coord_x'],n['coord_y'])))
            cluster[n_id].add_node(pydot.Node(name=n['id']+'_out'))
            dotG.add_subgraph(cluster[n_id])
        for i,e in data['paramEdge'].items():
            if e['type']==carrier:
                dotG.add_edge(pydot.Edge(src=e['nodeFrom']+'_out',
                                 dst=e['nodeTo']+'_in',
                                 color='red',
                                 label=pyo.value(model.varEdgePower[i])))
        for n,devs in data['paramNodeDevices'].items():
            for d in devs:
                f_in = data['paramDeviceDispatchIn'][d][carrier]
                if f_in!=0:
                    cluster[n].add_node(pydot.Node(d,color='blue'))
                    dotG.add_edge(pydot.Edge(dst=d,src=n+'_in',color='blue',
                       label=pyo.value(model.varDevicePower[d])*f_in))
                f_out = data['paramDeviceDispatchOut'][d][carrier]
                if f_out!=0:
                    cluster[n].add_node(pydot.Node(d,color='blue'))
                    dotG.add_edge(pydot.Edge(dst=n+'_out',src=d,color='blue',
                         label=pyo.value(model.varDevicePower[d])*f_out))
        for n,v in data['paramNodeNontrivial'].items():
            if not v[carrier]:
                dotG.add_edge(pydot.Edge(dst=n+'_out',src=n+'_in',
                     color='"black:invis:black"',arrowhead='none'))
                    
        dotG.write_png('pydot_{}.png'.format(carrier),prog='dot')    

def plotNetworkCombined(model):
    cluster = {}
    col = {'t': {'el':'red','gas':'blue'},
           'e': {'el':'red','gas':'blue'},
           'd': 'orange'
           }
    dotG = pydot.Dot(graph_type='digraph') #rankdir='LR',newrank='false')
    for n_id in model.setNode:
        cluster[n_id] = pydot.Cluster(graph_name=n_id,label=n_id)
        nodes_in=pydot.Subgraph(rank='same')
        nodes_out=pydot.Subgraph(rank='same')
        for carrier in model.setCarrier:
            label_in = carrier+'_in'
            label_out= carrier+'_out'
            if carrier=='gas':
                label_in +=':{:3.1f}'.format(pyo.value(instance.varGasPressure[n_id,'in']))
                label_out +=':{:3.1f}'.format(pyo.value(instance.varGasPressure[n_id,'out']))
            nodes_in.add_node(pydot.Node(name=n_id+'_'+carrier+'_in',
                   color=col['t'][carrier],label=label_in))
            nodes_out.add_node(pydot.Node(name=n_id+'_'+carrier+'_out',
                   color=col['t'][carrier],label=label_out))
        cluster[n_id].add_subgraph(nodes_in)
        cluster[n_id].add_subgraph(nodes_out)
        dotG.add_subgraph(cluster[n_id])
    
    for carrier in model.setCarrier:
        for i,e in model.paramEdge.items():
            if e['type']==carrier:
                dotG.add_edge(pydot.Edge(src=e['nodeFrom']+'_'+carrier+'_out',
                                         dst=e['nodeTo']+'_'+carrier+'_in',
                                         color=col['e'][carrier],
                                         fontcolor=col['e'][carrier],
                                         label=pyo.value(model.varEdgePower[i])))
        for n,devs in model.paramNodeDevices.items():
            for d in devs:
                cluster[n].add_node(pydot.Node(d,color=col['d'],style='filled',
                       label='"{}:{}"'.format(d,model.paramDevice[d]['name'])))
                f_in = model.paramDeviceDispatchIn[d][carrier]
                if f_in!=0:
                    dotG.add_edge(pydot.Edge(dst=d,src=n+'_'+carrier+'_in',
                         color=col['e'][carrier],
                         fontcolor=col['e'][carrier],
                         label=pyo.value(model.varDevicePower[d])*f_in))
                f_out = model.paramDeviceDispatchOut[d][carrier]
                if f_out!=0:
                    dotG.add_edge(pydot.Edge(dst=n+'_'+carrier+'_out',src=d,
                         color=col['e'][carrier],
                         fontcolor=col['e'][carrier],
                         label=pyo.value(model.varDevicePower[d])*f_out))
        for n,v in model.paramNodeNontrivial.items():
            if not v[carrier]:
                dotG.add_edge(pydot.Edge(dst=n+'_'+carrier+'_out',
                                         src=n+'_'+carrier+'_in',
                     color='"{0}:invis:{0}"'.format(col['e'][carrier]),
                     arrowhead='none'))
                
    dotG.write_png('pydotCombined.png',prog='dot')    


plotNetwork(data,df_node,instance)
plotNetworkCombined(instance)
#
#for carrier in data['setCarrier'][None]:
#    edges=df_edge[df_edge['type']==carrier].copy()
#    edges['node1'] = edges['node1']+'_in'
#    edges['node2'] = edges['node2']+'_out'
#    G = nx.from_pandas_edgelist(
#            create_using=nx.DiGraph(),
#            df=edges,
#            source='node1',target='node2',edge_attr='edgePower')
#    G.add_nodes_from(df_node['id']+'_in')
#    G.add_nodes_from(df_node['id']+'_out')
#    mask_dev = (df_device[['out_'+carrier,'in_'+carrier]
#                           ].fillna(0)!=0).all(axis=1)
#    devedges = df_device[mask_dev].copy()
#    devedges['node_in'] = devedges['node']+'_in'
#    devedges['node_out'] = devedges['node']+'_out'   
#    Gdev = nx.from_pandas_edgelist(create_using=nx.DiGraph(),
#                                             df=devedges,
#                                             source="node_out",
#                                             target="node_in")
#    #G.add_edges_from(Gdev.edges)
#    graph[carrier] = G
#    lab= nx.get_edge_attributes(G,'edgePower')
#    plt.figure()
#    pos = nx.spring_layout(G, fixed = fixedpos.keys(), pos = pos,k=0.01)
#    nx.draw_networkx(G,with_labels=True,pos=pos)
#    #nx.draw_networkx_edges(G,pos, label = carrier)
#    nx.draw_networkx_edge_labels(G, pos, alpha=0.5,edge_labels = lab)
#    nx.draw_networkx_edges(Gdev,pos,edge_color="green")
#    
#    trivials=pd.DataFrame(columns=['node_out','node_in'])
#    for n,v in data['paramNodeNontrivial'].items():
#        if not v[carrier]:
#            trivials= trivials.append({'node_out':n+'_out','node_in':n+'_in'},
#                            ignore_index=True)
#    Gtrivial = nx.from_pandas_edgelist(df=trivials,
#                                       create_using=nx.DiGraph(),
#                                       source='node_out',
#                                       target='node_in')
#    nx.draw_networkx_edges(Gtrivial,pos,edge_color="red")
#    
#    plt.title(carrier)
    
