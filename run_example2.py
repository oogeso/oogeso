import multicarrier
import plots
import matplotlib.pyplot as plt
plt.close("all")
outpath = "result_example2/"

doProfile = True
if doProfile:
    import cProfile
    import pstats
    pr = cProfile.Profile()
    pr.enable()

# 1 oil barrel = 0.158987 m3
# 1 $/barrel = 8.4 $/m3
# 50 $/barrel = 419 $ /m3
carrier_properties = {
    'wellstream': {
#            'composition': {'gas':0.5, 'oil':0.25, 'water':0.25} #Sm3
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

data,profiles = multicarrier.read_data_from_xlsx("data_example2.xlsx",
                                                 carrier_properties)
#data['paramDeviceEnergyInitially'][17]=2.5 # MWh - battery
instance = mc.createModelInstance(data,profiles)
#print("Writing instance to file.")
#instance.pprint(outpath+"problem_instance.txt")
print("Plotting networks")
plots.plotNetwork(mc,timestep=None,filename=outpath+"network_combined0.png")

print("Entering solve loop...")
status = mc.solveMany(solver="gurobi",timerange=[0,144],write_yaml=False,
                      timelimit=60)

if doProfile:
    pr.disable()
    sortby = pstats.SortKey.CUMULATIVE
    with open(outpath+'stats_profiling.txt', 'w') as stream:
        stats = pstats.Stats(pr, stream=stream)
        stats =stats.sort_stats(sortby)
        stats.print_stats()

# Save results to file:
mc.exportSimulationResult(filename=outpath+"simresult.xlsx")

# Analyse results:

sumCO2 = mc._dfCO2rate.mean()
co2intensity = mc._dfCO2intensity.mean()
exportrevenue = mc._dfExportRevenue.mean()
print("Mean CO2 emission rate      = {:.1f} kgCO2/hour".format(sumCO2))
print("Mean CO2 emission intensity = {:.1f} kgCO2/Sm3oe".format(co2intensity))
print("Mean export revenue         =",*["{}:{:.1f} ".format(x,v/1e6) for 
         x,v in mc._dfExportRevenue.mean().items() if v!=0],"M$/hour")

tstep=130
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
plots.plot_deviceprofile(mc,devs=['GT1','GT2'],profiles=profiles,
                         filename=outpath+"GTs_opt.png")
plots.plot_deviceprofile(mc,devs=['wind'],profiles=profiles,
                         filename=outpath+"WT_opt.png")
plots.plot_CO2rate_per_dev(mc,filename=outpath+"co2rate_opt.png",reverseLegend=True)
plots.plot_CO2_intensity(mc,filename=outpath+"co2intensity_opt.png")
plots.plot_devicePowerEnergy(mc,'battery',filename=outpath+"battery_opt.png")

#plots.plot_df(mc._dfDevicePower,id_var="device",filename=outpath+"plotly.html",
#              title="Device Power",ylabel="Power (MW)")

# Last optimisisation (results for a horizon)
#multicarrier.Plots.plotDeviceSumPowerLastOptimisation(instance,
#                                                      filename=outpath+"lastopt_devsum_el.png")
#multicarrier.Plots.plotEmissionRateLastOptimisation(instance,filename=outpath+"lastopt_co2out.png")
multicarrier.Plots.plotDevicePowerLastOptimisation1(mc,device='battery',
                filename=outpath+"lastopt_battery.png")


#print("CHECK nominal pressure values at t=...")
#mc.computeEdgePressureDrop(tstep)