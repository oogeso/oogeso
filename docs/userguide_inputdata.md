
# Input data

User guide: [Home](userguide.md)

## Contents

1. [Introduction](#introduction)
1. [Data]()
    * [General parameters](#general-parameters-parameters)
    * [Timeseries profiles](#time-series-profiles-profiles)
    * [Energy carriers](#energy-carriers-carriers)
    * [Nodes](#network-nodes-nodes)
    * [Edges](#network-edges-edges)
    * [Devices](#devices-devices)
1. [Electric system only modelling](#electric-system-only-modelling)
1. [Multi-energy system modelling](#multi-energy-system-modelling)

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



### General parameters (```parameters```)

parameter | type | description
----------|------|------------
time_delta_minutes      | int   | minutes per timestep
planning_horizon        | int   | number of timesteps in each rolling optimisation
optimisation_timesteps  | int   | number of timesteps between each optimisation
forecast_timesteps      | int   | number of timesteps beyond which forecast (instead of nowcast) profile is used
time_reserve_minutes    | int   | how long (minutes) stored energy must be sustained to count as reserve
co2_tax                 | float | CO2 emission costs (NOK/kgCO2)
emission_intensity_max    | float | maximum allowed emission intensity (kgCO2/Sm3oe), -1=no limit
emission_rate_max         | float | maximum allowed emission rate (kgCO2/hour), -1=no limit
max_pressure_deviation  | float | global limit for allowable relative pressure deviation from nominal, -1=no limit
objective               | string    | name of objective function to use (e.g. exportRevenue, costs)

### Time-series profiles (```profiles```)

Multiple time-series profiles can be specified. For each profile there are two separate time-series: One representing forecasted values for planning ahead, and one representing the "nowcast" or an updated forecast relevant for the near-real-time decisions.

Note that the actual real-time values (e.g. of power demand or of available wind power) is not used in the simulation, as it only concerns up to near real-time operational planning. To address real-time deviations and balancing, reserve power and backup capacity is required.

Profiles may be specified in the YAML file as indicated above, or they may be read from separate files.


### Energy carriers (```carriers```)

See below.



### Network nodes (```nodes```)
This consists simly of a set of nodes with an node identifier, but no additional parameters.


### Network edges (```edges```)
This consists a set of edges with a set of parameters for each edge
that depends on the edge type. These are as follows.

Common parameters for all edges:

parameter | type | description
----------|------|------------
type        | string |   type of edge (el, heat, gas, oil, wellstream, water, hydrogen)
node_from    | string | identifier of "from" node
node_to      | string | identifier of "from" node
include     | int    | wheter to include edge or not (1=yes, 0=no)
height_m    | float | height difference (metres) endpoint vs startpoint
length_km   | float | distance (km)
flow_max        | float | (optional) maximum power flow allowed (MW or Sm3/s)

### Devices (```devices```)
This consists a set of devices with a set of parameters for each device
that depends on the device type.

Common parameters for all device types:

parameter | type | description
----------|------|------------
node_id     | string    | identifier of node where it is connected
name        | string    | display name
model       | string    | name of device model type (see below)
include     | bool      | (optional) wether to include the node
profile     | string    | (optional) name of time series profile to use for scaling  (flow_max)
flow_max        | float | (optional) maximum flow allowed (MW or Sm3/s)
flow_min        | float | (optional) minimum flow allowed (MW or Sm3/s)
max_ramp_up     | float | (optional) maximum ramping (% of flow_max per min)
max_ramp_down   | float | (optioanl) maximum ramping (% of flow_max per min)
start_stop      | StartStopData | (optional) see below
reserve_factor  | float | (optional) how much of electric power counts towards the reserve (1=all, 0=none)
op_cost         | float | (optional) Operating cost
price.CARRIER | float   | (optional) price for power/flow in or out of device (NOK/MW or NOK/Sm3)

In addition to these parameters there are parameters that depend on the device *model*.  These are specified further down.

#### Start and stop data (```StartStopData```)

Start-stop data may be provided if the device is modelled with start/stop constraints and or penalties.

parameter | type | description
----------|------|------------
is_on_init          | bool  | device is initially on (True) or off (False)
penalty_start       | float | startup penalty (cost)
penalty_stop        | float | shutdown penalty (cost)
delay_start_minutes | int | minutes delay from activation to being online
minimum_time_on_minutes  | int | time device must be on once started (min)
minimum_time_off_minutes | int | time device must be off once stopped (min)


## Electric system only modelling

A common use case is to use Oogeso to model only the electricity network. Additional input data needed in this case is shown below.

### Carriers

el:
parameter | type | description
----------|------|-------------
powerflow_method  | str | "transport"  or "dc_pf"
reference_node    | str | id of node used as reference for voltage angles (with dc_pf method)
el_reserve_margin | float | required globally spinning reserve (MW). 
el_backup_margin  | float | (optional) required backup margin in case of genertor failure (MW)


### Edges (electric)

#### ```el```
parameter | type | description
----------|------|------------
reactance   | float | reactance in system per units
resistance  | float | resistance in system per units



### Devices (electric only)
The table below shows device models used with electric only modelling, and their additional parameters. These are specified in addition to the [generic device parameters](#devices-devices).

#### ```powersource```
parameter | type | description
----------|------|------------
penalty_function    | tuple | flow vs penalty function (piecewise linear)

#### ```powersink```
No extra data

#### ```storage_el```
parameter | type | description
----------|------|------------
E_max    | float | Energy capacity (MWh)
E_min    | float | Energy capacity (MWh)
E_init   | float | Storage level initially (MWh)
E_end    | float | (optional) Required storage level at end of each  optimisation (MWh)
E_cost    | float | Cost for deviation from target storage
eta     | float | Round-trip charge/discharge efficiency
target_profile | string | (optional) name of profile used for desired storage filling level


## Multi-energy system modelling

Additional edge data and device data used with multi-energy modelling is given [here](userguide_inputdata_multienergy.md).
