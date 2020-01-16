import multicarrier
import matplotlib.pyplot as plt
plt.close("all")

# TEST case with single step optimisation

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
    
multicarrier.Plots.plotNetworkCombined(mc)
multicarrier.Plots.plotNetworkCombined(mc,only_carrier='el')
multicarrier.Plots.plotNetworkCombined(mc,only_carrier='gas')
multicarrier.Plots.plotNetworkCombined(mc,only_carrier='heat')
print("Done")



#multicarrier.Plots.plotDevicePowerLastOptimisation1(mc,device=7,
#                                                      filename="wind.png")
    
multicarrier.Plots.plotDeviceSumPowerLastOptimisation(instance,
                                                      filename="devsum_el.png")
#multicarrier.Plots.plotDeviceSumPowerLastOptimisation(instance,carrier="gas",
#                                                      filename="devsum_gas.png")
multicarrier.Plots.plotEmissionRateLastOptimisation(instance,filename="co2out.png")
#multicarrier.Plots.plotNetworkCombined(instance,timestep=18,filename="t18.png")

multicarrier.Plots.plotProfiles(profiles,filename="profiles.png")

