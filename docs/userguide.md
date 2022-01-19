<img src=media/logo_oogeso.png width=200 alt="Oogeso logo">

# Oogeso - user guide 
[Back](../README.md) | [Input data](userguide_inputdata.md)

## Contents:
1. [What Oogeso does](#what-oogeso-does)
4. [How Oogeso works](#how-oogeso-works)
1. [Input data](#input-data)
1. [Basic usage](#basic-usage)
3. [Examples](#examples)
2. [Read more](#read-more)

## What Oogeso does
Oogeso is a Python package for optimising/simulating the energy system of offshore oil and gas platforms or similar systems.

It is intended for analysing systems with variability and flexibility associated with the integration of low-emission technologies such as wind power supply, batteries, other energy storage, and flexible energy demand.

It may be used to compute and compare key indicators such as greenhouse gas emissions, gas turbine starts and stops, etc, with different system configurations and operating strategies. For example, the performance with different sizes of wind turbines and batteries, different amounts of online reserve power required, or different amounts of allowed variation in petroleum production.


## How Oogeso works
Oogeso simulates the energy system time-step by time-step, with intervals specified in the input. For each time-step, it performs an optimisation that looks ahead over a certain optimisation horizon, and finds the use of resources within this horizon that gives the overall lowest *penalty* (generalised "cost") that satisfies all constraints.

The underlying system model is linear, and decision variables are continous (e.g. power output from each generator, power flow on a line) or integer (e.g. wheter a generator is online or offline). 
The simulation is therefore a rolling horizon mixed-integer linear program. 

Oogeso considers *flow* of different energy/matter carriers (electricity, heat, water, etc.) in and out of *devices* that are connected together in *networks* described through *nodes* and *edges*. 
In the simplest setup, only electricity flow and electricity network is considered.
Linear equations and ineqalities describe the physical relationships between the flows through these networks, and in and out of the devices. These are the optimisation problem *constraints*.

The *objective* of the optimisation is to minimise the overall penalty. The penalty may be costs in monetary units, fuel usage, carbon emissions or similar. The overall penalty is the sum of contributions from all devices.


## Input data

There are two main modelling alternatives with Ooges:
* Electricity only modelling (electricity supply, distribution and demand)
* Integrated multi-energy modelling (multiple energy/matter carriers)

If the electricity supply system is not closely integrated with the rest of the system an electricity-system only modelling may be sufficient. 
However, if interactions between elements of the integrated electricity, heat, processing system is important, the multi-energy capabilities of Oogeso are relevant.

Examples of input data are provided with the Oogeso test data.

Read more about the input data here:
* [Input data](userguide_inputdata.md)

## Basic usage
Preamble:

```python
import oogeso
import oogeso.utils
import oogeso.io
import oogeso.plots
```

Read input data. In this case time-series profiles are not included in the yaml input file, but read from separate CSV files:

```python
energy_system_data = oogeso.io.read_data_from_yaml('testcase2_inputdata.yaml')

# Read time-series profiles:
profiles_dfs = oogeso.io.read_profiles_from_csv(
    filename_forecasts="testcase2_profiles_forecasts.csv",
    filename_nowcasts="testcase2_profiles_nowcasts.csv",timestamp_col="timestamp",
        exclude_cols=["timestep"])
profiles = oogeso.utils.create_time_series_data(
    profiles_dfs["forecast"],profiles_dfs["nowcast"],
    time_start=None,time_end=None,timestep_minutes=15)
# Attach profiles to input data object:
energy_system_data.profiles = profiles
```

Run simulation:

```python
simulator = oogeso.Simulator(energy_system_data)
sim_result = simulator.run_simulation(solver="cbc")
```

Analyse and plot results (a few examples)

```python
# Plot electricity supply and consumption:
oogeso.plots.plot_sum_power_mix(
    res,optimisation_model=simulator.optimiser,carrier="el").show()
```

## Examples
For a quick demonstration of how Oogeso works, have a look at these
Jupyter notebooks:


* [Simple test case](https://github.com/oogeso/oogeso/blob/master/examples/test%20case2.ipynb)
* [LEOGO platform](https://github.com/oogeso/oogeso/blob/master/examples/leogo_reference_platform.ipynb)


## Read more

The Oogeso model is described in a paper to be subitted for publication in an academic journal.

Until that paper is available, this (outdated) document gives some details
about the modelling framework and the theoretical context of the Oogeso model:
* [manual](oogeso_manual.pdf)
