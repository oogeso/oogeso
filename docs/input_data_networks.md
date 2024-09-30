# Network (```nodes```, ```edges```)

User guide: [Home](index.md) / [input data](input_data.md)





## Contents
* [Nodes](#network-nodes-nodes)
* [Edges](#network-edges-edges)
    * [Data common for all types](#data-common-for-all-types)
    * [Data specific for each type](#data-specific-for-each-type)


# Network nodes (```nodes```)

parameter | type | description
----------|------|------------
id          | string  | unique node identifier

(No additional data needed)

# Network edges (```edges```)
This consists a set of edges with a set of parameters for each edge
that depends on the edge type. These are as follows.

## Data common for all types

parameter | type | description
----------|------|------------
id          | string  | unique edge identifier
carrier     | string |  edge carrier type (el, heat, gas, oil, wellstream, water, hydrogen)
node_from   | string | identifier of "from" node
node_to     | string | identifier of "from" node
include     | int    | wheter to include edge or not (1=yes, 0=no)
length_km   | float | distance (km)
flow_max        | float | (optional) maximum power flow allowed (MW or Sm3/s)


## Data specific for each type

Additional edge data specific to edge type:

### ```el```
parameter | type | description
----------|------|------------
reactance   | float | reactance (ohm/km) (used only with method "dc_pf")
resistance  | float | resistance (ohm/km) (used only with method "dc_pf")
voltage | float | voltage (kV) of line (single value) or transformer (tuple) (used only with method "dc_pf")
power_loss_function | tuple|  (optional) piecewise linear function/table of power transfer (MW) vs loss (MW). A tuple containing a list of x values and a list of y values `([x1,x2,..],[y1,y2,...])`

Note that for the (default) "transport" electricity power flow method, only the optional power_loss_function is relevant.


### ```heat, hydrogen, carbon```
No additional data used.

### ```gas, oil, water```

parameter | type | description
----------|------|------------
height_m      | float | height difference (metres) endpoint vs startpoint
pressure.from | float | nominal pressure (MPa) at start point (required if it is not given by device parameter)
pressure.to     | float | nominal pressure (MPa) at start point (required if it is not given by device parameter)
diameter_mm     | float | pipe internal diameter (mm)
temperature_K   | float | fluid temperature (K)
