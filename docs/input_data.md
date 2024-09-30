
# Input data

User guide: [Home](index.md)

## Contents

1. [Introduction](#introduction)
1. [General parameters](#general-parameters-parameters)
1. [Timeseries profiles](#time-series-profiles-profiles)
1. Other data:
    * [Energy carriers](input_data_carriers.md)
    * [Networks](input_data_networks.md) (nodes, edges)
    * [Devices](input_data_devices.md)


## Introduction

The Oogeso input is an EnergySytemData object, that may be constructed programmatically, or specified in a YAML. This object or yaml file specifies all
elements (nodes, edges, devices), how they are connected (network
topology) and parameters for each element. 

The input data can be specified in a YAML file with the following structure:
```yaml
parameters:
    <param>: <value>
profiles:
    - id: timeseries1
      data: <list of values>
      data_nowcast: <list of values>
    ...
carriers:
    - id: el
      <param>: <value>
    - id: heat
      <param>: <value>
    ...
devices:
    - id: device1
      <param>: <value>
    ...
nodes:
    - id: node1
      <param>: <value>    
    - id: node2
    ...
edges:
    - id: edge1
      <param>: <value>
    - id: edge2
    ...
```

The notation ```<param>: <value>``` above indicates a set of parameter-value pairs. The relevant parameters are described in the following:


## General parameters (```parameters```)

parameter | type | description
----------|------|------------
time_delta_minutes      | int   | minutes per timestep
planning_horizon        | int   | number of timesteps in each rolling optimisation
optimisation_timesteps  | int   | number of timesteps between each optimisation
forecast_timesteps      | int   | number of timesteps beyond which forecast (instead of nowcast) profile is used
emission_intensity_max    | float | maximum allowed emission intensity (kgCO2/Sm3oe), -1=no limit
emission_rate_max         | float | maximum allowed emission rate (kgCO2/hour), -1=no limit
objective     | string    | name of objective function to use (penalty, exportRevenue, costs)
piecewise_repn | string | method for impelementation of peicewise linear constraints in pyomo
optimisaion_return_data | list | (optional) list of variables to return from simulation



## Time-series profiles (```profiles```)

Multiple time-series profiles can be specified. For each profile there are two separate time-series: One representing forecasted values for planning ahead, and one representing the "nowcast" or an updated forecast relevant for the near-real-time decisions.

Note that the actual real-time values (e.g. of power demand or of available wind power) is not used in the simulation, as it only concerns up to near real-time operational planning. To address real-time deviations and balancing, reserve power and backup capacity is required.

Profiles may be specified in the YAML file as indicated above, or they may be read from separate files.

---
**Note regarding the use of profiles** 

The max/min flow ($Q$) constraint (e.g. electric power output from a wind turbine, or power demand by a load) is generally written on the form 
$Q_{min}\cdot \text{profile}(t) \le Q(t)  \le Q_{max}\cdot \text{profile}(t)$,
where $Q(t)$ is the flow being constrained.
That means you can choose where to put the units:
* ```flow_max``` gives the absolute value (e.g. MW) and the profile is given in relative units (typically in the range 0-1)
* ```flow_max = 1``` and the profile is given in absolute units (e.g. MW)

All examples use the first alternative.

---
