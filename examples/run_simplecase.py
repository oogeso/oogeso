import multicarrier
import plots
import matplotlib.pyplot as plt
plt.close("all")
outpath = "simplecase/"


carrier_properties = {
    'wellstream': {
            'composition': {'gas':0.995, 'oil':0.002, 'water':0.003} #Sm3
            },
    'water': {},
    'oil': {'export_price': 419, #EUR/Sm3
            #'energy_value':36000, #MJ/Sm3
            'CO2content':260 #kg/MWh
            },
    'gas':{'energy_value':40, #MJ/Sm3
           'Tb_basetemp_K':273+15,
           'Pb_basepressure_MPa':0.101,
           'G_gravity':0.6,
           'Z_compressibility':0.9,
           'k_heat_capacity_ratio':1.27,
           'R_individual_gas_constant': 500, # J/kg K
           'rho_density': 0.84, # kg/Sm3
           'CO2content':200, #kg/MWh
           'export_price': 50, #EUR/MWh
           }, 
    'el':{'energy_value':1,
          'CO2content':0,
          'export_price': 0.419, #$/Sm3
          },
    'heat':{'CO2content':0},
    }


mc = multicarrier.Multicarrier(loglevel="INFO",quadraticConstraints=False)

data,profiles = multicarrier.read_data_from_xlsx("simplecase.xlsx",
                                                 carrier_properties)
instance = mc.createModelInstance(data,profiles)

status = mc.solveMany(solver="gurobi",timerange=[0,50],write_yaml=False,
                      timelimit=60)

# Save results to file:
mc.exportSimulationResult(filename=outpath+"simresult.xlsx")

# Analyse results:

tstep=5
plots.plotNetwork(mc,timestep=tstep,filename=outpath+"network_combined.png")
plots.plotNetwork(mc,timestep=tstep,only_carrier='el',
                  filename=outpath+"network_el.png")
plots.plotNetwork(mc,timestep=tstep,only_carrier='heat',
                  filename=outpath+"network_heat.png")
plots.plotNetwork(mc,timestep=tstep,only_carrier='gas',
                  filename=outpath+"network_gas.png")
plots.plotNetwork(mc,timestep=tstep,only_carrier='oil',
                  filename=outpath+"network_oil.png")
plots.plotNetwork(mc,timestep=tstep,only_carrier='water',
                  filename=outpath+"network_water.png")

plots.plotProfiles(profiles,filename=outpath+"profiles.png")
plots.plot_SumPowerMix(mc,carrier="el",filename=outpath+"el_sum_opt.png")

#plots.plot_df(mc._dfDevicePower,id_var="device",filename=outpath+"plotly.html",
#              title="Device Power",ylabel="Power (MW)")

# Last optimisisation (results for a horizon)
#multicarrier.Plots.plotDeviceSumPowerLastOptimisation(instance,
#                                                      filename=outpath+"lastopt_devsum_el.png")
#multicarrier.Plots.plotEmissionRateLastOptimisation(instance,filename=outpath+"lastopt_co2out.png")


print("CHECK nominal pressure values at t=...")
mc.checkEdgePressureDrop(tstep)