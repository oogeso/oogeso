# Oogeso - user guide ![logo](doc/media/logo_oogeso.png)

[Home](README.md)

## Contents:
1. [Introduction](#introduction)
2. [Modelling documentation](#modelling-documentation)
3. [Examples](#examples)
4. [How Oogeso works](#how-oogeso-works)
5. [Input data](#input-data)
    + [Network topology and parameter values](#network-topology-and-parameter-values)
        * [paramParameters](#paramparameters)
        * [paramCarriers](#paramcarriers)
        * [paramNode](#paramnode)
        * [paramEdge](#paramedge)
        * [paramDevice](#paramdevice)
    + [Timeseries](#timeseries)

## Introduction
The Oogeso tool is open-source Python based software for simulating the operation of offshore oil and gas platform energy systems.

It is intended for analysing systems with variability and flexibility associated with the integration of low-emission technologies such as wind power supply, batteries, other energy storage, and flexible energy demand.

It may be used to compute and compare key performance indicators such as greenhouse gas emissions, gas turbine starts and stops, etc, with different system configurations and operating strategies. For example, the performance with different sizes of wind turbines and batteries, different amounts of online reserve power required, or different amounts of allowed variation in petroleum production.

The simulator is based on a rolling horizon mixed-integer linear optimisation. The system modelling simplified and linear, but includes energy and mass flows, with the basic links between oil/gas/water flows and energy demand by pumps and compressors, as well as gas turbine efficiency curves.

## Modelling documentation
A separate (draft and not always up-to-date) document provides more details
about the modelling framework and the theoretical context of the Oogeso model.
This is available [here](doc/oogeso_manual.pdf).

## Examples
For a quick demonstration of how Oogeso works, have a look at these
Jupyter notebooks:

(TODO)

## How Oogeso works
The program is essentially a mixed-integer linear optimisation problem solved
with a rolling horizon.
The offshore energy system is represented by a set of linear equations
specifying energy supply, distribution and demand. It is an integrated model
that includes multiple energy/mass carriers (electricity, heat, gas, oil,
hydrogen, water) and dependencies between these.


## Input data
The input data consists of a YAML file containing specifications of all
elements (nodes, edges, devices) included, how they are connected (network
topology) and parameters for each element. A separate file (XLSX or HD5)
contains time-series profiles.

### Network topology and parameter values

The network topology and parameter values are specified in a YAML file with the following structure:
```yaml
paramParameters:
    param: <value>
paramCarriers:
    gas: {...}        
    oil: {...}
    wellstream: {...}
    water: {...}
    hydrogen: {...}
    el: {}
    heat: {}
paramDevice:
    device1:
        param: <value>
    ...
paramNode:
    node1: {}
    ...
paramEdge:
    edge1:
        param: <value>
    ...
```

#### paramParameters

parameter | type | description
----------|------|------------
time_delta_minutes      | int   | minutes per timestep
planning_horizon        | int   | number of timesteps in each rolling optimisation
optimisation_timesteps  | int   | number of timesteps between each optimisation
forecast_timesteps      | int   | number of timesteps beyond which forecast (instead of nowcast) profile is used
time_reserve_minutes    | int   | how long (minutes) stored energy must be sustained to count as reserve
co2_tax                 | float | CO2 emission costs (NOK/kgCO2)
elBackupMargin          | float | required electrical backup margin (MW), -1=no limit
elReserveMargin         | float | required electrical reserve margin (MW), -1=no limit
emissionIntensityMax    | float | maximum allowed emission intensity (kgCO2/Sm3oe), -1=no limit
emissionRateMax         | float | maximum allowed emission rate (kgCO2/hour), -1=no limit
max_pressure_deviation  | float | global limit for allowable relative pressure deviation from nominal, -1=no limit
objective               | string    | name of objective function to use (e.g. exportRevenue, costs)
reference_node          | string    | name of node used as electrical voltage angle reference

#### paramCarriers

- el: no parameters required
- heat: no parameters required

- oil, water, wellstream:

parameter | type | description
----------|------|------------
darcy_friction  | float | Darcy friction factor
pressure_method | string | method for pressure drop calculation (darcy-weissbach/weymouth)
rho_density     | float | density (kg/m3)
viscosity       | float | viscosity (kg/(m s))

- gas:

parameter | type | description
----------|------|------------
CO2content          | float |   amount of CO2 per volume (kg/Sm3)
G_gravity           | float |  gas gravity constant
Pb_basepressure_MPa | float | base pressure (MPa) (1 atm=0.101 MPa)
Tb_basetemp_K       | float | base temperature (K) (15 degC=288 K)
R_individual_gas_constant   | float     | individual gas constant (J/(kg K))
Z_compressibility   | float |   gas compressibility
energy_value        | float | energy content, calorific value (MJ/Sm3)
k_heat_capacity_ratio   | float | heat capacity ratio
pressure_method     | string | method used co compute pressure drop in pipe (weymouth/darcy-weissbach)
rho_density         | float | density (kg/Sm3)

- hydrogen: no parameters required (simplified model)


#### paramNode
This consists of a set of nodes with an node identifier, but no parameters generally required.

Optional parameters that specifies allowable pressure deviations
from nominal values:

parameter | type | description
----------|------|------------
maxdeviation_pressure.CARRIER.INOUT| float | max relative deviation from nominal value

Above, CARRIER indicates the carrier type (gas, oil, etc.) and INOUT refers to whether it is into node (in) or out of node (out).
For example, a 30% deviation allowed on gas input pressure is written as
"maxdeviation_pressure.gas.in: 0.3"


#### paramEdge
This consists a set of edges with a set of parameters for each edge
that depends on the edge type. These are as follows.

- Common parameters:

parameter | type | description
----------|------|------------
type        | string |   type of edge (el, heat, gas, oil, wellstream, water, hydrogen)
nodeFrom    | string | identifier of "from" node
nodeTo      | string | identifier of "from" node
include     | int    | wheter to include edge or not (1=yes, 0=no)
height_m    | float | height difference (metres) endpoint vs startpoint
length_km   | float | distance (km)
Pmax        | float | (optional) maximum power flow allowed (MW)
Qmax        | float | (optional) maximum mass flow allowed (Sm3/s)

Additional parameters specific to edge type:

- type: el

parameter | type | description
----------|------|------------
reactance   | float | reactance in system per units
resistance  | float | resistance in system per units

- type: gas, oil, water

parameter | type | description
----------|------|------------
pressure.from | float | nominal pressure (MPa) at start point (required if it is not given by device parameter)
pressure.to     | float | nominal pressure (MPa) at start point (required if it is not given by device parameter)
diameter_mm     | float | pipe internal diameter (mm)
temperature_K   | float | fluid temperature (K)



#### paramDevice
This consists a set of devices with a set of parameters for each device
that depends on the device type.

- Common parameters:

parameter | type | description
----------|------|------------
name        | string    | display name of node
include     | int       | wether to include the node (1=yes, 0=no)
model       | string    | name of device model
node        | string    | name of node where it is connected
Pmax        | float     | (optional) maximum power allowed (MW)
Pmin        | float     | (optional) minimum power allowed (MW)
reserve_factor  | float | (optional) how much of electric power counts towards the reserve (1=all, 0=none)
Qmax        | float     | (optional) maximum flow rate allowed (Sm3/s)
Qmin        | float     | (optional) maximum flow rate allowed (Sm3/s)
profile     | string    | (optional) name of time series profile to use for scaling power (Pmax) or flow (Qmax)
price.CARRIER | float   | (optional) price for power/flow in or out of device (NOK/MW or NOK/Sm3)

Parameters specific to device model types

- gasturbine:

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

- source_el:

parameter | type | description
----------|------|------------
co2em       | float | Emissions per electric power delivered (kgCO2/MWh)
opCost      | float | Operating costs (eg. fuel) (NOK/MJ) (10 NOK/MWh = 0.003 NOK/MJ)

- storage_el:

parameter | type | description
----------|------|------------
Emax    | float | Energy capacity (MWh)
eta     | float | Round-trip charge/discharge efficiency

- sink_el, sink_heat, sink_water: No additional parameters

- pump_oil, pump_water

parameter | type | description
----------|------|------------
eta     | float | Pump efficiency

- compressor_el:

parameter | type | description
----------|------|------------
eta     | float | Compressor efficiency
Q0      | float | Nominal flow rate (used for linearisation) (Sm3/s)
temp_in | float | Inlet gas temperate (K)

- separator: no additional parameters

- separator2

parameter | type | description
----------|------|------------
eta_el      | float | electricity demand as fraction of flow rate (MW/(Sm3/s))
eta_heat    | float | heat demand as fraction of flow rate (MW/(Sm3/s))


- sink_gas, sink_oil: no additional parameters

- source_water

parameter | type | description
----------|------|------------
naturalpressure | float | Outlet pressure (MPa)

- well_gaslift

parameter | type | description
----------|------|------------
gas_oil_ratio   | float | Gas to oil ratio (GOR), ratio of produced gas to produced oil
water_cut       | float | Water cut (WC), ratio of produced water to total produced liquids
f_inj           | float | Ratio of gas injection rate (Sm3/s) to oil production rate (Sm3/s)
injectionpressure   | float | Gas injection pressure (MPa)
separatorpressure   | float | Pressure (from well) into separator (MPa)


### Time-series profiles

Multiple time-series profiles can be specified. For each profile there are two separate time-series: One representing forecasted values for planning ahead, and one representing the "nowcast" or an updated forecast relevant for the near-real-time decisions.

Note that the actual real-time values (e.g. of power demand or of available wind power) is not used in the simulation, as it only concerns up to near real-time operational planning. To address real-time deviations and balancing, reserve power and backup capacity is required.

Profiles may be read from an XLSX file with one "profiles" tab and one "profiles_forecast" tab. Updated forecasts are relevant for wind power availability where forecasts for the next hour are more accurate than forecasts e.g. 12 hours ahead.
