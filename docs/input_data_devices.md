# Devices (```devices```)

User guide: [Home](index.md) / [input data](input_data.md)

## Contents

* [Data common for all types](#data-common-for-all-types)
* [Data specific for each type](#data-specific-for-each-type)


### Data common for all types
This consists a set of devices with a set of parameters for each device
that depends on the device type.

Common parameters for all device types:

parameter | type | default | description
----------|------|---------|---
id          | string    || unique device identifier
node_id     | string    || identifier of node where it is connected
name        | string    |""| display name
model       | string    || name of device model type (see below)
include     | bool      |True| wether to include the node
profile     | string    |None| name of time series profile to use for scaling  (flow_max)
flow_max        | float | None| maximum flow allowed (MW or Sm3/s)
flow_min        | float | None| minimum flow allowed (MW or Sm3/s)
max_ramp_up     | float | None|maximum ramping (% of flow_max per min)
max_ramp_down   | float | None| maximum ramping (% of flow_max per min)
start_stop      | StartStopData | None| see below
reserve_factor  | float | None| how much of electric power counts towards the reserve (1=all, 0=none)
op_cost         | float | None| Operating cost
penalty_function    | tuple | None| flow vs penalty function (piecewise linear). A tuple containing a list of x values and a list of y values `([x1,x2,..],[y1,y2,...])`

In addition to these parameters there are parameters that depend on the device *model*.  These are specified further down.

#### Start and stop data (```StartStopData```)

Start-stop data may be provided if the device is modelled with start/stop constraints and or penalties.

parameter | type | default | description
----------|------|---------|---
is_on_init          | bool  | False | device is initially on (True) or off (False)
penalty_start       | float | 0 | startup penalty (cost)
penalty_stop        | float | 0 | shutdown penalty (cost)
delay_start_minutes | int | 0 | minutes delay from activation to being online
minimum_time_on_minutes  | int | 0 | time device must be on once started (min)
minimum_time_off_minutes | int | 0 | time device must be off once stopped (min)



### Data specific for each type


### ```powersource```
No extra data


### ```powersink```
No extra data


### ```storage_el```
parameter | type | description
----------|------|------------
E_max    | float | Energy capacity (MWh)
E_min    | float | Energy capacity (MWh)
E_init   | float | Storage level initially (MWh)
E_end    | float | (optional) Required storage level at end of each  optimisation (MWh)
E_cost    | float | Cost for deviation from target storage
eta     | float | Round-trip charge/discharge efficiency
target_profile | string | (optional) name of profile used for desired storage filling level



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
hydrogen_blend_max | float | maximum fraction of hydrogen in fuel blend (Sm3 hydrogen vs Sm3 total)
hydrogen_blend_min | float | minimum fraction of hydrogen in fuel blend (Sm3 hydrogen vs Sm3 total)

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

### ```sink_gas, sink_oil, sink_carbon```
parameter | type | description
----------|------|------------
price     | float | revenue for exported fluid (negative cost in the optimisation)

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

### ```carbon_capture```
parameter | type | description
----------|------|------------
carbon_capture_rate | float | Carbon capture rate (CCR), typically 0.9
exhaust_gas_recirculation   | float | Exhaust gas recirculatio rate (EGR), typically in the range 0-0.6
compressor_pressure_in      | float | Inlet pressure for compression of captured CO2 (MPa)
compressor_pressure_out     | float | Outlet pressure for compression of captured CO2 (MPa)
compressor_eta              | float | Efficiency of CO2 compresstion (0-1)
compressor_temp_in          | float | Inlet gas temperature (K)

### ```storage_hydrogen```
parameter | type | description
----------|------|------------
E_max    | float | Energy capacity (MWh)
E_min    | float | Energy capacity (MWh)
E_init   | float | Storage level initially (MWh)
E_cost    | float | Cost for deviation from target storage
eta     | float | Round-trip charge/discharge efficiency
target_profile | string | (optional) name of profile used for desired storage filling level


### ```storage_gas_linepack```
parameter | type | description
----------|------|------------
E_init   | float | Storage level initially (Sm3)
volume_m3 | float | Volume of pipeline

Maximum storage can be specified by setting pressure limits on associated terminals (done via edge parameters)