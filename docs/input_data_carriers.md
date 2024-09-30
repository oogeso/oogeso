# Carriers (```carriers```)

User guide: [Home](index.md) / [input data](input_data.md)

## Contents
* [Data common for all types](#data-common-for-all-types)
* [Data specific for each type](#data-specific-for-each-type)


## Data common for all types

parameter | type | description
----------|------|------------
id          | string  | carrier type (el, heat, hydrogen, gas, oil, water, wellstream)

Other data differs for the different carrier types.

## Data specific for each type

### ```el```
parameter | type | description
----------|------|-------------
powerflow_method  | str | "transport"  or "dc_pf"
reference_node    | str | id of node used as reference for voltage angles (with dc_pf method)
el_reserve_margin | float | required globally spinning reserve (MW). 
el_backup_margin  | float | (optional) required backup margin in case of genertor failure (MW)
reserve_storage_minutes    | int   | how long (minutes) stored energy must be sustained to count as reserve


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
Pb_basepressure_MPa | float | base pressure (MPa) (1 atm=0.101 MPa)
Tb_basetemp_K       | float | base temperature (K) (15 degC=288 K)
R_individual_gas_constant   | float     | individual gas constant (J/(kg K))
Z_compressibility   | float |   gas compressibility
energy_value        | float | energy content, calorific value (MJ/Sm3)
k_heat_capacity_ratio   | float | heat capacity ratio
pressure_method     | string | method used co compute pressure drop in pipe (weymouth/darcy-weissbach)
rho_density         | float | density (kg/Sm3)

### ```carbon``` (CO2)
parameter | type | description
----------|------|-------------
R_individual_gas_constant | float | individual gas constant (J/(kg K))
Z_compressibility         | float | gas compressibility
k_heat_capacity_ratio     | float | heat capacity ratio


