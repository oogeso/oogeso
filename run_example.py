# Main script

'''
Energy carrier units are in J, or flow = J/s=W

1 Btu/ft^3 = 37258.9 J/m^3 
(google search "1 btu/ft3 in j/m3")
natural gas: 1000 Btu/ft3 = 37 MJ/m3

See: https://en.wikipedia.org/wiki/Heat_of_combustion#Higher_heating_values_of_natural_gases_from_various_sources
Norway energy_value = 39.24 MJ/m3
https://www.norskpetroleum.no/en/calculator/about-energy-calculator/
energy_value=40

One of the largest gas fields:
Åsgard July 2019: 0.824 bill m3 gas production per month
=> 12.71 GW average (energy value)
'''

import multicarrier
import plots
#import pandas as pd
import matplotlib.pyplot as plt
plt.close("all")

doProfile = False
if doProfile:
    import cProfile, pstats, io
    import pstats
    pr = cProfile.Profile()
    pr.enable()
    
carrier_properties = {
    'gas':{'energy_value':40,
           'Tb_basetemp_K':273+15,
           'Pb_basepressure_kPa':101,
           'G_gravity':0.6,
           'Z_compressibility':0.9,
           'CO2content':5}, #5 kg per MWh
    'el':{'energy_value':1,
          'CO2content':0},
    'heat':{'CO2content':0},
    }


mc = multicarrier.Multicarrier(loglevel="INFO")
datafile = "data_example.xlsx"
data,profiles = multicarrier.read_data_from_xlsx(datafile,carrier_properties)

data['paramDeviceEnergyInitially'][17]=2.5 # MWh - battery
instance = mc.createModelInstance(data,profiles)

status = mc.solveMany(solver="cbc",time_end=48,write_yaml=False)

if doProfile:
    pr.disable()
    sortby = pstats.SortKey.CUMULATIVE
    with open('stats.txt', 'w') as stream:
        stats = pstats.Stats(pr, stream=stream)
        stats =stats.sort_stats(sortby)
        stats.print_stats()
    #s = io.StringIO()
    #ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    #ps.print_stats()
    #print(s.getvalue())
    #pr.dump_stats("stats.txt")
    

# Analyse results:

multicarrier.Plots.plotNetworkCombined(mc)
multicarrier.Plots.plotNetworkCombined(mc,only_carrier='el')
multicarrier.Plots.plotNetworkCombined(mc,only_carrier='gas')
multicarrier.Plots.plotNetworkCombined(mc,only_carrier='heat')

multicarrier.Plots.plotDeviceSumPowerLastOptimisation(instance,
                                                      filename="devsum_el.png")
#multicarrier.Plots.plotDeviceSumPowerLastOptimisation(instance,carrier="gas",
#                                                      filename="devsum_gas.png")
multicarrier.Plots.plotEmissionRateLastOptimisation(instance,filename="co2out.png")
#multicarrier.Plots.plotNetworkCombined(instance,timestep=18,filename="t18.png")

# Show device output vs available power for wind turbine:
#multicarrier.Plots.plotDevicePowerLastOptimisation1(mc,device=7,
#                                                      filename="wind.png")
multicarrier.Plots.plotDevicePowerLastOptimisation1(mc,device=17,
                                                    filename="lastopt_battery.png")

multicarrier.Plots.plotProfiles(profiles,filename="profiles.png")

sumCO2 = multicarrier.pyo.value(mc.compute_CO2(instance))
print("last optimisation: CO2 emitted = {} kg".format(sumCO2))

#plots.plot_df(mc._dfDevicePower,id_var="device",filename="plotly.html",
#              title="Device Power",ylabel="Power (MW)")

plots.plot_SumPowerMix(mc,carrier="el",filename="el_sum_opt.png")
plots.plot_deviceprofile(mc,dev=7,profiles=profiles,filename="wind_opt.png")
#plots.plot_deviceprofile(mc,dev=17,profiles=profiles) # battery
plots.plot_CO2_rate(mc,filename="co2rate_opt.png",reverseLegend=True)
plots.plot_devicePowerEnergy(mc,17,filename="battery_opt.png")
