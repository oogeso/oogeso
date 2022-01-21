# Output data

User guide: [Home](userguide.md)


The Oogeso simulation output is an `oogeso.dto.SimulationResult` object, which contains a set of Pandas multi-index Series.

These are:


variable | index | description
---------|-------|------------
`device_flow`           | ['device', 'carrier', 'terminal', 'time'] |Input/output flow per device and network type
`device_is_prep`        | ['device', 'time'] | Device startup preparation status (boolean)
`device_is_on`          | ['device', 'time'] | Device on/off status (boolean)
`device_starting`       | ['device', 'time'] | Device starting status (boolean)
`device_stopping`       | ['device', 'time'] | Device stopping status (boolean)
`device_storage_energy` | ['device', 'time'] | Energy storage filling level (Sm3 or MJ)
`device_storage_pmax`   | ['device', 'time'] |  Max available "flow" (power/fluid) from storage (Sm3/s or MW)
`penalty`               | ['time', 'device']| Device assosiated penalty rate (PENALTY_UNIT/s)
`edge_flow`             | ['edge', 'time']  | Flow rate (Sm3/s or MW)
`edge_loss`             | ['edge', 'time']  | Loss rate (MW) - only relevant for energy flow (el and heat)
`el_voltage_angle`      | ['node', 'time']  | Voltage angle at node - only relevant for electricity floc computed via dc-pf
`terminal_pressure`     | ['node', 'carrier', 'terminal', 'time'] | Pressure at node (MPa)
`terminal_flow`         | ['node', 'carrier', 'time'] | Direct flow between in and out terminal of node - relevant if there is no device inbetween
`co2_rate`              | ['time']          | Emission rate (sum of all devices) (kgCO2/s)
`co2_rate_per_dev`      | ['time', 'device']| Emission rate per device (kgCO2/s)
`export_revenue`        | ['time', 'carrier'] | Revenue rate for exported oil/gas (CURRENCY/s)
`co2_intensity`         | ['time']          | CO2 intensity of exported oil/gas (kgCO2/Sm3oe)
`el_reserve`            | ['time']          | Available online electrical reserve capacity (MW)
`el_backup`             | ['time', 'device']| Available online electrical backup per device (MW)


In addition, it contains the following Pandas DataFrames:

variable | index | columns | description
---------|-------|---------|------------
`duals`            | time-step | name of dual   | Dual value
`profiles_forecast`| time step | profile id     | Time-series profiles used in simulation (coped from input)
`profiles_nowcast` | time step | profile id     | Time-series profiles used in simulation (copied from input)

Depending on the setup and particular case, some elements may not be present (set to `None`).

Information about duals (marginal changes in objective value if a constraint is relaxed) are included if the Oogeso simulation was called with a non-empty `store_duals` parameter.