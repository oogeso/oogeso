# Extra input data with multi-energy modelling

This page summarises input data used with multi-energy moddeling with Oogeso. This data is in addition to the data described in 

## Energy carriers (```carriers```)

### ```heat, hydrogen```
No additional data required


### ```oil, water, wellstream```
parameter | type | description
----------|------|-------------
darcy_friction  | float | Darcy friction factor
pressure_method | string | method for pressure drop calculation (darcy-weissbach/weymouth)
rho_density     | float | density (kg/m3)
viscosity       | float | viscosity (kg/(m s))

### ```gas```
parameter | type | description
----------|------|-------------
co2_content          | float |   amount of CO2 per volume (kg/Sm3)
G_gravity           | float |  gas gravity constant
Pb_basepressure_MPa | float | base pressure (MPa) (1 atm=0.101 MPa)
Tb_basetemp_K       | float | base temperature (K) (15 degC=288 K)
R_individual_gas_constant   | float     | individual gas constant (J/(kg K))
Z_compressibility   | float |   gas compressibility
energy_value        | float | energy content, calorific value (MJ/Sm3)
k_heat_capacity_ratio   | float | heat capacity ratio
pressure_method     | string | method used co compute pressure drop in pipe (weymouth/darcy-weissbach)
rho_density         | float | density (kg/Sm3)


## Edges (```edges```)
These tables shows edgeparameters used in addition to the [generic edge parameters](input_data.md#network-edges-edges).

Additional edge data specific to edge type:

### ```heat, hydrogen```
No additional data used.

### ```gas, oil, water```

parameter | type | description
----------|------|------------
height_m      | float | height difference (metres) endpoint vs startpoint
pressure.from | float | nominal pressure (MPa) at start point (required if it is not given by device parameter)
pressure.to     | float | nominal pressure (MPa) at start point (required if it is not given by device parameter)
diameter_mm     | float | pipe internal diameter (mm)
temperature_K   | float | fluid temperature (K)


## Devices (```devices```)
These tables shows additional devices and device parameters used in addition to the [generic device parameters](input_data.md#devices-devices) in multi-energy modelling.


### ```gasturbine```

parameter | type | description
----------|------|------------
eta_heat    | float | efficiency of converting energy (heat) loss to usable heat
fuelA       | float | fuel vs power parameter A (fuel = A*power + B)
fuelB       | float | fuel vs power parameter B (fuel = A*power + B)
isOn_init   | int   | whether device is on at simulation start (1=yes, 0=no)
maxRampDown | float | maximum ramp down rate, relative to capacity per minute (1=100%/min)
maxRampUp   | float | maximum ramp down rate, relative to capacity per minute (1=100%/min)
startupCost | float | cost (NOK) for each start-up
startupDelay    | float | delay (min) from start-up activation to powr output

### ```source_el```

parameter | type | description
----------|------|------------
co2em       | float | Emissions per electric power delivered (kgCO2/MWh)
opCost      | float | Operating costs (eg. fuel) (NOK/MJ) (10 NOK/MWh = 0.003 NOK/MJ)

### ```sink_el, sink_heat, sink_water```
No additional data required

### ```pump_oil, pump_water```

parameter | type | description
----------|------|------------
eta     | float | Pump efficiency

### ```compressor_el```

parameter | type | description
----------|------|------------
eta     | float | Compressor efficiency
Q0      | float | Nominal flow rate (used for linearisation) (Sm3/s)
temp_in | float | Inlet gas temperate (K)

### ```separator```
No additional data.

This separator model assues a single wellstream input flow.

### ```separator2```
This separator model assumes separate input flows for oil, gas, and water.

parameter | type | description
----------|------|------------
eta_el      | float | electricity demand as fraction of flow rate (MW/(Sm3/s))
eta_heat    | float | heat demand as fraction of flow rate (MW/(Sm3/s))

### ```sink_gas, sink_oil```
No additional data

### ```source_water```
parameter | type | description
----------|------|------------
naturalpressure | float | Outlet pressure (MPa)

### ```well_gaslift```
parameter | type | description
----------|------|------------
gas_oil_ratio   | float | Gas to oil ratio (GOR), ratio of produced gas to produced oil
water_cut       | float | Water cut (WC), ratio of produced water to total produced liquids
f_inj           | float | Ratio of gas injection rate (Sm3/s) to oil production rate (Sm3/s)
injectionpressure   | float | Gas injection pressure (MPa)
separatorpressure   | float | Pressure (from well) into separator (MPa)

