import pyomo.environ as pyo
import pandas as pd
import networkx as nx
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


TODO: model all nodes with two terminals, input and output. 
Devices put inbetween.
Terminals can have different pressure, voltage etc.
'''










model = pyo.AbstractModel()

# Sets
model.setCarrier = pyo.Set(doc="energy carrier")
model.setNode = pyo.Set()
model.setEdge1= pyo.Set(within=model.setCarrier*model.setNode*model.setNode)
model.setEdge= pyo.Set()
model.setDevice = pyo.Set()

# Parameters (input data)
# "out"=out of the associated network node
# "in"=into the associated network node
model.paramEdge = pyo.Param(model.setEdge)
model.paramDevice = pyo.Param(model.setDevice)
model.paramDeviceDispatchIn = pyo.Param(model.setDevice)
model.paramDeviceDispatchOut = pyo.Param(model.setDevice)
model.paramNodeDevicesIn = pyo.Param(model.setNode)
model.paramNodeDevicesOut = pyo.Param(model.setNode)
model.paramNodeEdgesIn = pyo.Param(model.setCarrier,model.setNode)
model.paramNodeEdgesOut = pyo.Param(model.setCarrier,model.setNode)

# Variables
#model.varNodeVoltageAngle = pyo.Var(model.setNode,within=pyo.Reals)
model.varEdgePower = pyo.Var(model.setEdge,within=pyo.Reals)
model.varEdgePower2 = pyo.Var(within=pyo.Reals)
model.varDevicePower = pyo.Var(model.setDevice,within=pyo.NonNegativeReals)


# Objective
def rule_objective(model):
    sumE = sum(model.varDevicePower[k]
               for k in model.setDevice)
    return sumE
model.objObjective = pyo.Objective(rule=rule_objective,sense=pyo.minimize)

# Constraints
def rule_nodeEnergyBalance(model,carrier,node):
    Pinj = 0
    # devices:
    if node in model.paramNodeDevicesIn:
        for dev in model.paramNodeDevicesIn[node]:
            # power into node (from device)
            Pinj += (model.varDevicePower[dev]
                    *model.paramDeviceDispatchIn[dev][carrier])
    if node in model.paramNodeDevicesOut:
        for dev in model.paramNodeDevicesOut[node]:
            # power out from node (into device)
            Pinj -= (model.varDevicePower[dev]
                    *model.paramDeviceDispatchOut[dev][carrier])
    # edges:
    if (carrier,node) in model.paramNodeEdgesIn:
        for edg in model.paramNodeEdgesIn[(carrier,node)]:
            # power into node from edge
            Pinj += (model.varEdgePower[edg])
    if (carrier,node) in model.paramNodeEdgesOut:
        for edg in model.paramNodeEdgesOut[(carrier,node)]:
            # power out of node into edge
            Pinj -= (model.varEdgePower[edg])
    
    if not(Pinj is 0):
        return (Pinj==0)
    else:
        return pyo.Constraint.Skip
model.constrNodePowerBalance = pyo.Constraint(model.setCarrier,model.setNode,
                                              rule=rule_nodeEnergyBalance)


def rule_devicePmax(model,dev):
    return (model.paramDevice[dev]['Pmin'] 
            <= model.varDevicePower[dev] 
            <= model.paramDevice[dev]['Pmax'])
model.constrDevicePmax = pyo.Constraint(model.setDevice,rule=rule_devicePmax)

def rule_devicePmin(model,dev):
    return (model.varDevicePower[dev] >= model.paramDevice[dev]['Pmin'])
#model.constrDevicePmin = pyo.Constraint(model.setDevice,rule=rule_devicePmin)






# Input data
df_node = pd.read_excel("data_example.xlsx",sheet_name="node")
df_edge = pd.read_excel("data_example.xlsx",sheet_name="edge")
df_device = pd.read_excel("data_example.xlsx",sheet_name="device")

df_edge = df_edge[df_edge['include']==1]
df_device = df_device[df_device['include']==1]

cols_in = {col:col.split("in_")[1] 
           for col in df_device.columns if "in_" in col }
dispatch_in = df_device[list(cols_in.keys())].rename(columns=cols_in).fillna(0)
cols_out = {col:col.split("out_")[1] 
           for col in df_device.columns if "out_" in col }
dispatch_out = df_device[list(cols_out.keys())].rename(columns=cols_out).fillna(0)


data = {}
data['setCarrier'] = {None:df_edge['type'].unique().tolist()}
data['setNode'] = {None:df_node['id'].tolist()}
#data['setEdge1'] = {None: list(zip(
#        df_edge['type'],df_edge['node1'],df_edge['node2']))}
data['setEdge'] = {None: df_edge.index.tolist()}
data['setDevice'] = {None:df_device.index.tolist()}
data['paramDeviceDispatchIn'] = dispatch_in.to_dict(orient='index') 
data['paramDeviceDispatchOut'] = dispatch_out.to_dict(orient='index') 
data['paramNodeDevicesIn'] = df_device.groupby('node_in').groups
data['paramNodeDevicesOut'] = df_device.groupby('node_out').groups
data['paramDevice'] = df_device[['external','Pmax','Pmin']
                            ].to_dict(orient='index')

# change edge index to multi-index, then to tuple
#df_edge1=df_edge.set_index(['type','node1','node2'])
#df_edge1.index = df_edge1.index.to_native_types()

data['paramEdge'] = df_edge.to_dict(orient='index')
data['paramNodeEdgesOut'] = df_edge.groupby(['type','node1']).groups
data['paramNodeEdgesIn'] = df_edge.groupby(['type','node2']).groups


instance = model.create_instance(data={None:data},name='MultiCarrier')
instance.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
instance.pprint(filename='model.txt')

opt = pyo.SolverFactory('gurobi')

sol = opt.solve(instance) 

sol.write_yaml()

print("\nSOLUTION - devicePower:")
for k in instance.varDevicePower.keys():
    print("  {}: {}".format(k,instance.varDevicePower[k].value))

print("\nSOLUTION - edgePower:")
for k in instance.varEdgePower.keys():
    power = instance.varEdgePower[k].value
    print("  {}: {}".format(k,power))
    df_edge.loc[k,'edgePower'] = power

# display all duals
#print ("Duals")
#for c in instance.component_objects(pyo.Constraint, active=True):
#    print ("   Constraint",c)
#    for index in c:
#        print ("      ", index, instance.dual[c[index]])


df_edge['edgePower']
graph={}
pos = {v['id']:(v['coord_x'],v['coord_y']) for i,v in df_node.iterrows()}
labels_edge = df_edge.set_index(['node1','node2'])['edgePower'].to_dict()

for carrier in data['setCarrier'][None]:
    G = nx.from_pandas_edgelist(
            create_using=nx.DiGraph(),
            df=df_edge[df_edge['type']==carrier],
            source='node1',target='node2',edge_attr='edgePower')
    G.add_nodes_from(df_node['id'])
    mask_dev = (df_device[['out_'+carrier,'in_'+carrier]
                           ].fillna(0)!=0).all(axis=1)
    Gdev = nx.from_pandas_edgelist(create_using=nx.DiGraph(),
                                             df=df_device[mask_dev],
                                             source="node_out",
                                             target="node_in")
    #G.add_edges_from(Gdev.edges)
    graph[carrier] = G
    lab= nx.get_edge_attributes(G,'edgePower')
    plt.figure()
    nx.draw_networkx(G,with_labels=True,pos=pos)
    #nx.draw_networkx_edges(G,pos, label = carrier)
    nx.draw_networkx_edge_labels(G, pos, alpha=0.5,edge_labels = lab)
    nx.draw_networkx_edges(Gdev,pos,edge_color="green")
    plt.title(carrier)