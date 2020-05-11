
import pyomo.environ as pyo

model = pyo.AbstractModel()

# Sets
model.setCarrier = pyo.Set()

# Parameters (input data)
model.paramMaxFlow = pyo.Param(model.setCarrier)
model.paramCoupling = pyo.Param(model.setCarrier,model.setCarrier,default=0)
model.paramFlowOut = pyo.Param(model.setCarrier,mutable=True)
model.paramCost1 = pyo.Param(model.setCarrier)
model.paramCost2 = pyo.Param(model.setCarrier)

# Variables
model.varFlow = pyo.Var(model.setCarrier,within=pyo.NonNegativeReals)

# Objective
def rule_objective(model):
    sumE = sum(model.paramCost1[k]*model.varFlow[k]
               +model.paramCost2[k]*model.varFlow[k]**2 
               for k in model.setCarrier)
    return sumE
model.objObjective = pyo.Objective(rule=rule_objective,sense=pyo.minimize)

# Constraints
def rule_maxflow(model,k):
    return (model.varFlow[k]<= model.paramMaxFlow[k])
model.constrMaxFlow = pyo.Constraint(model.setCarrier,rule=rule_maxflow)

def rule_coupling(model,k):
    flowOut_k = sum(model.paramCoupling[k,j]*model.varFlow[j] 
                    for j in model.setCarrier)
    return (model.paramFlowOut[k] == flowOut_k)
model.constrFlowCoupling = pyo.Constraint(model.setCarrier,rule=rule_coupling)

def rule_marginalcost_input(model,k):
    psi_k = model.paramCost1[k] + 2*model.paramCost2[k]*model.varFlow[k]
    return psi_k
model.exprMarginalCost = pyo.Expression(model.setCarrier,
                                        rule=rule_marginalcost_input)

# Input data

data = {None:{
        'setCarrier': {None: ['el','heat','gas']},
        'paramMaxFlow': {'el':1000,'heat':1000,'gas':1000},
        'paramFlowOut': {'el':2,'heat':5,'gas':0},
        'paramCoupling': {
                ('el','el'):1,
                ('gas','el'):0,
                ('heat','el'):0,
                ('heat','heat'):0.9,
                ('el','gas'):0.3,
                ('heat','gas'):0.4,
                ('gas','gas'):0.0},
        'paramCost1': {'el':12,'gas':5,'heat':4},
        'paramCost2': {'el':0.12,'gas':0.05,'heat':0.04}
        }}


instance = model.create_instance(data)
instance.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

opt = pyo.SolverFactory('gurobi')

sol = opt.solve(instance) 

sol.write_yaml()

print("SOLUTION:")
for k in instance.varFlow.keys():
    print("  {}: {}".format(k,instance.varFlow[k].value))

# display all duals
print ("Duals")
for c in instance.component_objects(pyo.Constraint, active=True):
    print ("   Constraint",c)
    for index in c:
        print ("      ", index, instance.dual[c[index]])
print("Marginal costs (input)")
for index in instance.exprMarginalCost:
    print("   {:8}:{:6.3f}".format(index,pyo.value(instance.exprMarginalCost[index])))