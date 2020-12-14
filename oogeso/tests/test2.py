%load_ext autoreload
%autoreload 2
import matplotlib.pyplot as plt
import IPython
import pyomo.environ as pyo
import logging
import pandas as pd
import plotly.express as px
import os
import sys
# adding to path required when package has not been installed
# such as when used online with Binder
#module_path = os.path.abspath(os.path.join('..'))
#if module_path not in sys.path:
#    sys.path.append(module_path)
import oogeso

oogeso.plots.plotter='plotly'



data0 = oogeso.file_io.read_data_from_yaml('../../examples/test case2.yaml')
profiles = oogeso.file_io.read_profiles_from_xlsx('../../examples/test case2 profiles.xlsx')
store_duals = {
    'elcost':{'constr':'constrDevicePmin','indx':('dem',None)},
    #'wind': {'constr':'constrTerminalEnergyBalance','indx':('el','windfarm','out',None)},
    }

# MODIFY input data 
for dev in ['GT1','GT2']:
    data0['paramDevice'][dev]['isOn_init'] = 1
#data['paramParameters']['elReserveMargin'] = 0

mc = oogeso.Multicarrier(loglevel="INFO")
data = oogeso.file_io.create_initdata(data0)
mc.createModelInstance(data,profiles)

# SOLVE
status = mc.solveMany(solver="cbc",timerange=[0,90],write_yaml=False,
    store_duals=store_duals)

fig = oogeso.plots.plot_SumPowerMix(mc,carrier="el")
fig.show()
