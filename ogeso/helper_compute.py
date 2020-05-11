# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:03:02 2020

@author: hsven
"""

import pandas as pd

T1=300
eta_is = 0.7
Z=1
rho_s=0.84 #kg/m3
R=438 #J/(kg K), specific gas constant = R_u/M = 8.314/0.019
k=1.27 # heat capacity ratio, c_p/c_c
#k=1.4
CV = 40e6 #J/m3

# Isentropic compression

def P_demand_per_W(p_ratio):
    '''Returns compressor demand per mass flow , P/W'''
    return 1/eta_is*k/(k-1)*Z*R*T1*((p_ratio)**(k/(k-1))-1)



df =pd.DataFrame()
df['p_ratio'] = [1, 1.2, 1.5, 2, 5, 10]

df['P_demand_per_W'] = P_demand_per_W(df['p_ratio'])

df['P_demand_per_Q'] = rho_s*df['P_demand_per_W']

df['P_demand_per_P'] = df['P_demand_per_Q']/CV

print("R rho_s = {} J/m3 K".format(R*rho_s))

print("Menon, R rho_s = {} J/m3 K".format(4.0639e3 * 24*3600/1e6))