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

carrier_properties = {
    'gas':{'energy_value':40e6,
           'Tb_basetemp_K':273+15,
           'Pb_basepressure_kPa':101,
           'G_gravity':0.6,
           'Z_compressibility':0.9,
           'CO2content':5}, #5 kg per MW
    'el':{'energy_value':1,
          'CO2content':0},
    'heat':{'CO2content':0},
    }


mc = multicarrier.Multicarrier(loglevel="INFO")

data = multicarrier.read_data_from_xlsx("data_example.xlsx",carrier_properties)

instance = mc.createModelInstance(data,filename="model.txt")


sol = mc.solve(instance,solver="gurobi",write_yaml=False)


multicarrier.Plots.plotNetworkCombined(instance)
multicarrier.Plots.plotNetworkCombined(instance,only_carrier='el')
multicarrier.Plots.plotNetworkCombined(instance,only_carrier='gas')
multicarrier.Plots.plotNetworkCombined(instance,only_carrier='heat')

multicarrier.Plots.plotDevicePowerLastOptimisation(instance,
                                                   filename="devoutput.png")
multicarrier.Plots.plotNetworkCombined(instance,timestep=18,filename="t18.png")
sumCO2 = multicarrier.pyo.value(mc.compute_CO2(instance))
print("CO2 emitted = {} kg".format(sumCO2))
