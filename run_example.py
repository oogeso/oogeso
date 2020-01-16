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

runSingle = False
if runSingle:
    print("Without battery")
    #mc = multicarrier.Multicarrier(loglevel="INFO")
    data['paramDevice'][17]['Emax']=0 # MWh - battery
    instance = mc.createModelInstance(data,profiles,filename="model0.txt")
    sol = mc.solve(solver="cbc",write_yaml=False)
    multicarrier.Plots.plotDeviceSumPowerLastOptimisation(instance,filename="a_devsum_el0.png")
    multicarrier.Plots.plotDevicePowerLastOptimisation1(mc,device=17,filename="a_battery0.png")
    sumCO2 = multicarrier.pyo.value(mc.compute_CO2(instance))
    print("CO2 emitted = {} kg/hour".format(sumCO2))
    
    print("With battery")
    #mc = multicarrier.Multicarrier(loglevel="INFO")
    #data,profiles = multicarrier.read_data_from_xlsx(datafile,carrier_properties)
    data['paramDevice'][17]['Emax']=2.5 # MWh - battery
    instance = mc.createModelInstance(data,profiles)
    mc.instance.paramDeviceEnergyInitially[17]=2.5 #MWh (fully charged at start)
    sol = mc.solve(solver="cbc",write_yaml=False)
    multicarrier.Plots.plotDeviceSumPowerLastOptimisation(instance,filename="b_devsum_el0.png")
    multicarrier.Plots.plotDevicePowerLastOptimisation1(mc,device=17,filename="b_battery0.png")
    sumCO2 = multicarrier.pyo.value(mc.compute_CO2(instance))
    print("CO2 emitted = {} kg/hour".format(sumCO2))
    
    print("Done")

runMany=True
if runMany:
    data['paramDeviceEnergyInitially'][17]=2.5 # MWh - battery
    instance = mc.createModelInstance(data,profiles)
    
    status = mc.solveMany(solver="cbc",time_end=24,write_yaml=False)
    mc.instance.pprint(filename="model.txt")
    
    multicarrier.Plots.plotNetworkCombined(instance)
    multicarrier.Plots.plotNetworkCombined(instance,only_carrier='el')
    multicarrier.Plots.plotNetworkCombined(instance,only_carrier='gas')
    multicarrier.Plots.plotNetworkCombined(instance,only_carrier='heat')
    
    multicarrier.Plots.plotDeviceSumPowerLastOptimisation(instance,
                                                          filename="devsum_el.png")
    #multicarrier.Plots.plotDeviceSumPowerLastOptimisation(instance,carrier="gas",
    #                                                      filename="devsum_gas.png")
    multicarrier.Plots.plotEmissionRateLastOptimisation(instance,filename="co2out.png")
    #multicarrier.Plots.plotNetworkCombined(instance,timestep=18,filename="t18.png")
    
    # Show device output vs available power for wind turbine:
    multicarrier.Plots.plotDevicePowerLastOptimisation1(mc,device=7,
                                                          filename="wind.png")
    
    multicarrier.Plots.plotProfiles(profiles,filename="profiles.png")
    
    sumCO2 = multicarrier.pyo.value(mc.compute_CO2(instance))
    print("CO2 emitted = {} kg".format(sumCO2))
    
    plots.plot_df(mc._dfDevicePower,id_var="device",filename="plotly.html",
                  title="Device Power",ylabel="Power (MW)")
    
    plots.plot_deviceprofile(mc,dev=7,profiles=profiles) # wind
    #plots.plot_deviceprofile(mc,dev=17,profiles=profiles) # battery
    plots.plot_devicePowerEnergy(mc,17)