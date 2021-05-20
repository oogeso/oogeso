"""This module contains the MILP problem definition"""

import pyomo.environ as pyo
import pyomo.opt as pyopt
import numpy as np
import pandas as pd
import logging
from . import milp_compute
from . import electricalsystem


def check_constraints_complete(pyomo_model):
    """Check that constraints are defined for all device models

    Parameters
    ----------
    pyomo_model : Pyomo model
        Energy system model as MILP problem
    """

    logging.debug("Checking existance of constraints for all device types")
    # devmodels = Multicarrier.devicemodel_inout()
    device_models = milp_compute.devicemodel_inout()
    for i in device_models.keys():
        isdefined = hasattr(pyomo_model, "constrDevice_{}".format(i))
        if not isdefined:
            raise Exception(
                "Device model constraints for '{}' have" " not been defined".format(i)
            )
        logging.debug("....{} -> OK".format(i))


def definePyomoModel():
    """Specify the energy system model as a Pyomo optimisation model

    Returns
    -------
    model : abstract Pyomo model
        Energy system model as an MILP problem

    Model parameters are named model.paramXXX, variables are model.varXXX,
    constraints are model.constrXXX and the objective function is
    model.objective

    """
    model = pyo.AbstractModel()

    # Sets
    model.setCarrier = pyo.Set(doc="energy carrier")
    model.setNode = pyo.Set()
    model.setEdge = pyo.Set()
    model.setDevice = pyo.Set()
    model.setTerminal = pyo.Set(initialize=["in", "out"])
    model.setFlowComponent = pyo.Set(initialize=["oil", "gas", "water"])
    # time for rolling horizon optimisation:
    model.setHorizon = pyo.Set(ordered=True)
    model.setParameters = pyo.Set()
    model.setProfile = pyo.Set()

    # Parameters (input data)
    model.paramNode = pyo.Param(model.setNode, within=pyo.Any)
    model.paramEdge = pyo.Param(model.setEdge, within=pyo.Any)
    model.paramDevice = pyo.Param(model.setDevice, within=pyo.Any)
    model.paramNodeCarrierHasSerialDevice = pyo.Param(model.setNode, within=pyo.Any)
    model.paramNodeDevices = pyo.Param(model.setNode, within=pyo.Any)
    model.paramNodeEdgesFrom = pyo.Param(
        model.setCarrier, model.setNode, within=pyo.Any
    )
    model.paramNodeEdgesTo = pyo.Param(model.setCarrier, model.setNode, within=pyo.Any)
    model.paramParameters = pyo.Param(model.setParameters, within=pyo.Any)
    model.paramCarriers = pyo.Param(model.setCarrier, within=pyo.Any)

    model.paramCoeffB = pyo.Param(model.setNode, model.setNode, within=pyo.Reals)
    model.paramCoeffDA = pyo.Param(model.setEdge, model.setNode, within=pyo.Reals)
    # Mutable parameters (will be modified between successive optimisations)
    model.paramProfiles = pyo.Param(
        model.setProfile, model.setHorizon, within=pyo.Reals, mutable=True, initialize=0
    )
    model.paramDeviceIsOnInitially = pyo.Param(
        model.setDevice, mutable=True, within=pyo.Binary, initialize=0
    )
    model.paramDevicePrepTimestepsInitially = pyo.Param(
        model.setDevice, mutable=True, within=pyo.Integers, initialize=0
    )
    # needed for ramp rate limits:
    model.paramDevicePowerInitially = pyo.Param(
        model.setDevice, mutable=True, within=pyo.Reals, initialize=0
    )
    # needed for energy storage:
    model.paramDeviceEnergyInitially = pyo.Param(
        model.setDevice, mutable=True, within=pyo.Reals, initialize=0
    )
    # target energy level at end of horizon (useful for long-term storage)
    model.paramDeviceEnergyTarget = pyo.Param(
        model.setDevice, mutable=True, within=pyo.Reals, initialize=0
    )

    # Variables
    # model.varNodeVoltageAngle = pyo.Var(model.setNode,within=pyo.Reals)
    model.varEdgeFlow = pyo.Var(model.setEdge, model.setHorizon, within=pyo.Reals)
    #        model.varEdgeFlowComponent = pyo.Var(
    #                model.setEdgeWithFlowComponents,model.setFlowComponent,
    #                model.setHorizon,within=pyo.Reals)
    model.varDeviceIsPrep = pyo.Var(
        model.setDevice, model.setHorizon, within=pyo.Binary, initialize=0
    )
    model.varDeviceIsOn = pyo.Var(
        model.setDevice, model.setHorizon, within=pyo.Binary, initialize=0
    )
    model.varDeviceStarting = pyo.Var(
        model.setDevice, model.setHorizon, within=pyo.Binary
    )
    model.varDeviceStopping = pyo.Var(
        model.setDevice, model.setHorizon, within=pyo.Binary
    )
    model.varDeviceStorageEnergy = pyo.Var(
        model.setDevice, model.setHorizon, within=pyo.Reals
    )
    # available reserve power from storage (linked to power rating and storage level):
    model.varDeviceStoragePmax = pyo.Var(
        model.setDevice, model.setHorizon, within=pyo.NonNegativeReals, initialize=0
    )
    # binary variable related to available powr from storage:
    model.varStorY = pyo.Var(model.setDevice, model.setHorizon, within=pyo.Binary)
    # absolute value variable for storage with target level:
    model.varDeviceStorageDeviationFromTarget = pyo.Var(
        model.setDevice, within=pyo.NonNegativeReals, initialize=0
    )
    model.varPressure = pyo.Var(
        model.setNode,
        model.setCarrier,
        model.setTerminal,
        model.setHorizon,
        within=pyo.NonNegativeReals,
        initialize=0,
    )
    model.varElVoltageAngle = pyo.Var(model.setNode, model.setHorizon, within=pyo.Reals)
    model.varDeviceFlow = pyo.Var(
        model.setDevice,
        model.setCarrier,
        model.setTerminal,
        model.setHorizon,
        within=pyo.NonNegativeReals,
    )
    model.varTerminalFlow = pyo.Var(
        model.setNode, model.setCarrier, model.setHorizon, within=pyo.Reals
    )
    #        model.varTerminalFlowComponent = pyo.Var(
    #                model.setNode,model.setFlowComponent,model.setHorizon,
    #                within=pyo.Reals)

    # Objective

    #        def storPmaxPushup(model):
    #            '''term in objective function to push varDeviceStoragePmax up
    #            to its maximum value (to get correct calculation of reserve)'''
    #            sumStorPmax=0
    #            for dev in model.setDevice:
    #                if model.paramDevice[dev]['model'] == 'storage_el':
    #                    for t in model.setHorizon:
    #                        sumStorPmax += model.varDeviceStoragePmax[dev,t]
    #            return sumStorPmax

    def rule_objective_co2(model):
        """CO2 emissions per sec"""
        sumE = milp_compute.compute_CO2(model)  # *model.paramParameters['CO2_price']
        return sumE

    def rule_objective_co2intensity(model):
        """CO2 emission intensity (CO2 per exported oil/gas)
        DOES NOT WORK - NONLINEAR (ratio)"""
        sumE = milp_compute.compute_CO2_intensity(model)
        return sumE

    def rule_objective_costs(model):
        """costs (co2 price, operating costs, startstop, storage depletaion) per second"""
        startupCosts = milp_compute.compute_startup_costs(model)  # kr/s
        storageDepletionCosts = milp_compute.compute_costForDepletedStorage(model)
        opCosts = milp_compute.compute_operatingCosts(model)  # kr/s
        co2 = milp_compute.compute_CO2(model)  # kgCO2/s
        co2_tax = model.paramParameters["co2_tax"]  # kr/kgCO2
        co2Cost = co2 * co2_tax  # kr/s
        sumCost = co2Cost + startupCosts + storageDepletionCosts + opCosts
        return sumCost

    def rule_objective_exportRevenue(model):
        """revenue from exported oil and gas minus costs (co2 price and
        operating costs) per second"""
        sumRevenue = milp_compute.compute_exportRevenue(model)  # kr/s
        startupCosts = milp_compute.compute_startup_costs(model)  # kr/s
        co2 = milp_compute.compute_CO2(model)  # kgCO2/s
        co2_tax = model.paramParameters["co2_tax"]  # kr/kgCO2
        co2Cost = co2 * co2_tax  # kr/s
        storageDepletionCosts = milp_compute.compute_costForDepletedStorage(model)
        opCosts = milp_compute.compute_operatingCosts(model)  # kr/s
        sumCost = -sumRevenue + co2Cost + startupCosts + storageDepletionCosts + opCosts
        return sumCost

    def rule_objective(model):
        obj = model.paramParameters["objective"]
        if obj == "co2":
            rule = rule_objective_co2(model)
        elif obj == "costs":
            rule = rule_objective_costs(model)
        elif obj == "exportRevenue":
            rule = rule_objective_exportRevenue(model)
        elif obj == "co2intensity":
            rule = rule_objective_co2intensity(model)
        else:
            raise Exception("Objective '{}' has not been implemented".format(obj))
        return rule

    model.objObjective = pyo.Objective(rule=rule_objective, sense=pyo.minimize)
    #        model.objObjective = pyo.Objective(rule=rule_objective_exportRevenue,
    #                                           sense=pyo.minimize)

    def rule_devmodel_well_production(model, dev, t, i):
        if model.paramDevice[dev]["model"] != "well_production":
            return pyo.Constraint.Skip
        if i == 1:
            node = model.paramDevice[dev]["node"]
            lhs = model.varPressure[(node, "wellstream", "out", t)]
            rhs = model.paramDevice[dev]["naturalpressure"]
            return lhs == rhs

    model.constrDevice_well_production = pyo.Constraint(
        model.setDevice,
        model.setHorizon,
        pyo.RangeSet(1, 1),
        rule=rule_devmodel_well_production,
    )

    def rule_devmodel_well_gaslift(model, dev, carrier, t, i):
        if model.paramDevice[dev]["model"] != "well_gaslift":
            return pyo.Constraint.Skip

        # flow from reservoir (equals flow out minus gas injection)
        Q_reservoir = (
            sum(model.varDeviceFlow[dev, c, "out", t] for c in model.setFlowComponent)
            - model.varDeviceFlow[dev, "gas", "in", t]
        )
        node = model.paramDevice[dev]["node"]
        if i == 1:
            # output pressure is fixed
            lhs = model.varPressure[(node, carrier, "out", t)]
            rhs = model.paramDevice[dev]["separatorpressure"]
            return lhs == rhs
        elif i == 2:
            # output flow per comonent determined by GOR and WC
            GOR = model.paramDevice[dev]["gas_oil_ratio"]
            WC = model.paramDevice[dev]["water_cut"]
            comp_oil = (1 - WC) / (1 + GOR - GOR * WC)
            comp_water = WC / (1 + GOR - GOR * WC)
            comp_gas = GOR * (1 - WC) / (1 + GOR * (1 - WC))
            comp = {"oil": comp_oil, "gas": comp_gas, "water": comp_water}
            lhs = model.varDeviceFlow[dev, carrier, "out", t]
            if carrier == "gas":
                lhs -= model.varDeviceFlow[dev, carrier, "in", t]
            rhs = comp[carrier] * Q_reservoir
            return lhs == rhs
        elif i == 3:
            # gas injection rate vs proudction rate (determines input gas)
            # gas injection rate vs OIL proudction rate (determines input gas)
            if carrier == "gas":
                lhs = model.varDeviceFlow[dev, "gas", "in", t]
                # rhs = model.paramDevice[dev]['f_inj']*Q_reservoir
                rhs = (
                    model.paramDevice[dev]["f_inj"]
                    * model.varDeviceFlow[dev, "oil", "out", t]
                )
                return lhs == rhs
            else:
                return pyo.Constraint.Skip
        elif i == 4:
            # gas injection pressure is fixed
            if carrier == "gas":
                lhs = model.varPressure[(node, carrier, "in", t)]
                rhs = model.paramDevice[dev]["injectionpressure"]
                return lhs == rhs
            else:
                return pyo.Constraint.Skip

    model.constrDevice_well_gaslift = pyo.Constraint(
        model.setDevice,
        model.setFlowComponent,
        model.setHorizon,
        pyo.RangeSet(1, 4),
        rule=rule_devmodel_well_gaslift,
    )

    # TODO: separator equations
    logging.info("TODO: separator power (eta) and heat (eta2) demand")

    def rule_devmodel_separator(model, dev, t, i):
        if model.paramDevice[dev]["model"] != "separator":
            return pyo.Constraint.Skip
        wellstream_prop = model.paramCarriers["wellstream"]
        GOR = wellstream_prop["gas_oil_ratio"]
        WC = wellstream_prop["water_cut"]
        comp_oil = (1 - WC) / (1 + GOR - GOR * WC)
        comp_water = WC / (1 + GOR - GOR * WC)
        comp_gas = GOR * (1 - WC) / (1 + GOR * (1 - WC))
        flow_in = model.varDeviceFlow[dev, "wellstream", "in", t]
        if i == 1:
            lhs = model.varDeviceFlow[dev, "gas", "out", t]
            rhs = flow_in * comp_gas
            return lhs == rhs
        elif i == 2:
            lhs = model.varDeviceFlow[dev, "oil", "out", t]
            rhs = flow_in * comp_oil
            return lhs == rhs
        elif i == 3:
            # return pyo.Constraint.Skip
            lhs = model.varDeviceFlow[dev, "water", "out", t]
            rhs = flow_in * comp_water
            return lhs == rhs
        elif i == 4:
            # electricity demand
            lhs = model.varDeviceFlow[dev, "el", "in", t]
            rhs = flow_in * model.paramDevice[dev]["eta_el"]
            return lhs == rhs
        elif i == 5:
            lhs = model.varDeviceFlow[dev, "heat", "in", t]
            rhs = flow_in * model.paramDevice[dev]["eta_heat"]
            return lhs == rhs
        elif i == 6:
            """gas pressure out = nominal"""
            node = model.paramDevice[dev]["node"]
            lhs = model.varPressure[(node, "gas", "out", t)]
            rhs = model.paramNode[node]["pressure.gas.out"]
            return lhs == rhs
        elif i == 7:
            """oil pressure out = nominal"""
            node = model.paramDevice[dev]["node"]
            lhs = model.varPressure[(node, "oil", "out", t)]
            rhs = model.paramNode[node]["pressure.oil.out"]
            return lhs == rhs
        elif i == 8:
            """water pressure out = nominal"""
            node = model.paramDevice[dev]["node"]
            lhs = model.varPressure[(node, "water", "out", t)]
            rhs = model.paramNode[node]["pressure.water.out"]
            return lhs == rhs

    model.constrDevice_separator = pyo.Constraint(
        model.setDevice,
        model.setHorizon,
        pyo.RangeSet(1, 8),
        rule=rule_devmodel_separator,
    )

    # Alternative separator model - using oil/gas/water input instead of
    # wellstream
    def rule_devmodel_separator2(model, dev, fc, t, i):
        if model.paramDevice[dev]["model"] != "separator2":
            return pyo.Constraint.Skip
        # composition = model.paramCarriers['wellstream']['composition']
        wellstream_prop = model.paramCarriers["wellstream"]
        flow_in = sum(
            model.varDeviceFlow[dev, f, "in", t] for f in model.setFlowComponent
        )
        if i == 1:
            # component flow in = flow out
            lhs = model.varDeviceFlow[dev, fc, "out", t]
            rhs = model.varDeviceFlow[dev, fc, "in", t]
            return lhs == rhs
        elif i == 2:
            # pressure out is nominal
            node = model.paramDevice[dev]["node"]
            lhs = model.varPressure[(node, fc, "out", t)]
            rhs = model.paramNode[node]["pressure.{}.out".format(fc)]
            return lhs == rhs
        elif i == 3:
            # electricity demand
            if fc != model.setFlowComponent.first():
                return pyo.Constraint.Skip
            lhs = model.varDeviceFlow[dev, "el", "in", t]
            rhs = flow_in * model.paramDevice[dev]["eta_el"]
            return lhs == rhs
        elif i == 4:
            # heat demand
            if fc != model.setFlowComponent.first():
                return pyo.Constraint.Skip
            lhs = model.varDeviceFlow[dev, "heat", "in", t]
            rhs = flow_in * model.paramDevice[dev]["eta_heat"]
            return lhs == rhs

    model.constrDevice_separator2 = pyo.Constraint(
        model.setDevice,
        model.setFlowComponent,
        model.setHorizon,
        pyo.RangeSet(1, 4),
        rule=rule_devmodel_separator2,
    )

    def rule_devmodel_compressor_el(model, dev, t, i):
        if model.paramDevice[dev]["model"] != "compressor_el":
            return pyo.Constraint.Skip
        if i == 1:
            """gas flow in equals gas flow out (mass flow)"""
            lhs = model.varDeviceFlow[dev, "gas", "in", t]
            rhs = model.varDeviceFlow[dev, "gas", "out", t]
            return lhs == rhs
        elif i == 2:
            """Device el demand"""
            lhs = model.varDeviceFlow[dev, "el", "in", t]
            rhs = milp_compute.compute_compressor_demand(model, dev, linear=True, t=t)
            return lhs == rhs

    model.constrDevice_compressor_el = pyo.Constraint(
        model.setDevice,
        model.setHorizon,
        pyo.RangeSet(1, 2),
        rule=rule_devmodel_compressor_el,
    )

    def rule_devmodel_compressor_gas(model, dev, t, i):
        if model.paramDevice[dev]["model"] != "compressor_gas":
            return pyo.Constraint.Skip
        # device power is defined as gas demand in MW (rather than Sm3)
        powerdemand = milp_compute.compute_compressor_demand(
            model, dev, linear=True, t=t
        )
        if i == 1:
            # matter conservation
            gas_energy_content = model.paramCarriers["gas"]["energy_value"]  # MJ/Sm3
            lhs = model.varDeviceFlow[dev, "gas", "out", t]
            rhs = (
                model.varDeviceFlow[dev, "gas", "in", t]
                - powerdemand / gas_energy_content
            )
            return lhs == rhs

    model.constrDevice_compressor_gas = pyo.Constraint(
        model.setDevice,
        model.setHorizon,
        pyo.RangeSet(1, 1),
        rule=rule_devmodel_compressor_gas,
    )

    def rule_devmodel_gasheater(model, dev, t, i):
        if model.paramDevice[dev]["model"] != "gasheater":
            return pyo.Constraint.Skip
        if i == 1:
            # heat out = gas input * energy content * efficiency
            gas_energy_content = model.paramCarriers["gas"]["energy_value"]  # MJ/Sm3
            lhs = model.varDeviceFlow[dev, "heat", "out", t]
            rhs = (
                model.varDeviceFlow[dev, "gas", "in", t]
                * gas_energy_content
                * model.paramDevice[dev]["eta"]
            )
            return lhs == rhs

    model.constrDevice_gasheater = pyo.Constraint(
        model.setDevice,
        model.setHorizon,
        pyo.RangeSet(1, 1),
        rule=rule_devmodel_gasheater,
    )

    def rule_devmodel_heatpump(model, dev, t, i):
        if model.paramDevice[dev]["model"] != "heatpump":
            return pyo.Constraint.Skip
        if i == 1:
            # heat out = el in * efficiency
            lhs = model.varDeviceFlow[dev, "heat", "out", t]
            rhs = (
                model.varDeviceFlow[dev, "el", "in", t] * model.paramDevice[dev]["eta"]
            )
            return lhs == rhs

    model.constrDevice_heatpump = pyo.Constraint(
        model.setDevice,
        model.setHorizon,
        pyo.RangeSet(1, 1),
        rule=rule_devmodel_heatpump,
    )

    def rule_devmodel_gasturbine(model, dev, t, i):
        if model.paramDevice[dev]["model"] != "gasturbine":
            return pyo.Constraint.Skip
        elpower = model.varDeviceFlow[dev, "el", "out", t]
        gas_energy_content = model.paramCarriers["gas"]["energy_value"]  # MJ/Sm3
        if i == 1:
            """turbine el power out vs gas fuel in"""
            # fuel consumption (gas in) is a linear function of el power output
            # fuel = B + A*power
            # => efficiency = power/(A+B*power)
            A = model.paramDevice[dev]["fuelA"]
            B = model.paramDevice[dev]["fuelB"]
            Pmax = model.paramDevice[dev]["Pmax"]
            lhs = model.varDeviceFlow[dev, "gas", "in", t] * gas_energy_content / Pmax
            rhs = (
                B * (model.varDeviceIsOn[dev, t] + model.varDeviceIsPrep[dev, t])
                + A * model.varDeviceFlow[dev, "el", "out", t] / Pmax
            )
            return lhs == rhs
        elif i == 2:
            """heat output = (gas energy in - el power out)* heat efficiency"""
            lhs = model.varDeviceFlow[dev, "heat", "out", t]
            rhs = (
                model.varDeviceFlow[dev, "gas", "in", t] * gas_energy_content
                - model.varDeviceFlow[dev, "el", "out", t]
            ) * model.paramDevice[dev]["eta_heat"]
            return lhs == rhs

    model.constrDevice_gasturbine = pyo.Constraint(
        model.setDevice,
        model.setHorizon,
        pyo.RangeSet(1, 2),
        rule=rule_devmodel_gasturbine,
    )

    def rule_devmodel_source(model, dev, t, i, carrier):
        if model.paramDevice[dev]["model"] != "source_{}".format(carrier):
            return pyo.Constraint.Skip
        if i == 1:
            node = model.paramDevice[dev]["node"]
            lhs = model.varPressure[(node, carrier, "out", t)]
            rhs = model.paramDevice[dev]["naturalpressure"]
            return lhs == rhs

    def rule_devmodel_source_gas(model, dev, t, i):
        return rule_devmodel_source(model, dev, t, i, "gas")

    model.constrDevice_source_gas = pyo.Constraint(
        model.setDevice,
        model.setHorizon,
        pyo.RangeSet(1, 1),
        rule=rule_devmodel_source_gas,
    )

    def rule_devmodel_source_oil(model, dev, t, i):
        return rule_devmodel_source(model, dev, t, i, "oil")

    model.constrDevice_source_oil = pyo.Constraint(
        model.setDevice,
        model.setHorizon,
        pyo.RangeSet(1, 1),
        rule=rule_devmodel_source_oil,
    )

    def rule_devmodel_source_water(model, dev, t, i):
        return rule_devmodel_source(model, dev, t, i, "water")

    model.constrDevice_source_water = pyo.Constraint(
        model.setDevice,
        model.setHorizon,
        pyo.RangeSet(1, 1),
        rule=rule_devmodel_source_water,
    )

    # TODO: diesel gen fuel, onoff variables..
    def rule_devmodel_source_el(model, dev, t):
        if model.paramDevice[dev]["model"] != "source_el":
            return pyo.Constraint.Skip
        expr = pyo.Constraint.Skip
        return expr

    model.constrDevice_source_el = pyo.Constraint(
        model.setDevice, model.setHorizon, rule=rule_devmodel_source_el
    )

    def rule_devmodel_storage_el(model, dev, t, i):
        if model.paramDevice[dev]["model"] != "storage_el":
            return pyo.Constraint.Skip
        if i == 1:
            # energy balance
            # (el_in*eta - el_out/eta)*dt = delta storage
            # eta = efficiency charging and discharging
            delta_t = model.paramParameters["time_delta_minutes"] / 60  # hours
            lhs = (
                model.varDeviceFlow[dev, "el", "in", t] * model.paramDevice[dev]["eta"]
                - model.varDeviceFlow[dev, "el", "out", t]
                / model.paramDevice[dev]["eta"]
            ) * delta_t
            if t > 0:
                Eprev = model.varDeviceStorageEnergy[dev, t - 1]
            else:
                Eprev = model.paramDeviceEnergyInitially[dev]
            rhs = model.varDeviceStorageEnergy[dev, t] - Eprev
            return lhs == rhs
        elif i == 2:
            # energy storage limit
            ub = model.paramDevice[dev]["Emax"]
            lb = 0
            # Specifying Emin may be useful e.g. to impose that battery
            # should only be used for reserve
            # BUT - probably better to specify an energy depletion cost
            # instead (Ecost) - to allow using battery (at a cost) if required
            if "Emin" in model.paramDevice[dev]:
                lb = model.paramDevice[dev]["Emin"]
            return pyo.inequality(lb, model.varDeviceStorageEnergy[dev, t], ub)
        elif i == 3:
            # return pyo.Constraint.Skip # unnecessary -> generic Pmax/min constraints
            # discharging power limit
            ub = model.paramDevice[dev]["Pmax"]
            return model.varDeviceFlow[dev, "el", "out", t] <= ub
        elif i == 4:
            # return pyo.Constraint.Skip # unnecessary -> generic Pmax/min constraints
            # charging power limit
            # ub = model.paramDevice[dev]['Pmax']
            if "Pmin" in model.paramDevice[dev]:
                ub = -model.paramDevice[dev]["Pmin"]
            else:
                # assume max charging power is the same as discharging power (Pmax)
                ub = model.paramDevice[dev]["Pmax"]  # <- see generic Pmax/min constr
            return model.varDeviceFlow[dev, "el", "in", t] <= ub
        elif i == 5:
            # Constraint 5-8: varDeviceStoragePmax = min{Pmax,E/dt}
            # ref: https://or.stackexchange.com/a/1174
            lhs = model.varDeviceStoragePmax[dev, t]
            rhs = model.paramDevice[dev]["Pmax"]
            return lhs <= rhs
        elif i == 6:
            lhs = model.varDeviceStoragePmax[dev, t]
            # Parameter specifying for how long the power needs to be
            # sustained to count as reserve (e.g. similar to GT startup time)
            dt_hours = model.paramParameters["time_reserve_minutes"] / 60
            rhs = model.varDeviceStorageEnergy[dev, t] / dt_hours
            return lhs <= rhs
        elif i == 7:
            bigM = 10 * model.paramDevice[dev]["Pmax"]
            lhs = model.varDeviceStoragePmax[dev, t]
            rhs = model.paramDevice[dev]["Pmax"] - bigM * (1 - model.varStorY[dev, t])
            return lhs >= rhs
        elif i == 8:
            dt_hours = model.paramParameters["time_reserve_minutes"] / 60
            bigM = 10 * model.paramDevice[dev]["Emax"] / dt_hours
            lhs = model.varDeviceStoragePmax[dev, t]
            rhs = (
                model.varDeviceStorageEnergy[dev, t] / dt_hours
                - bigM * model.varStorY[dev, t]
            )
            return lhs >= rhs
        elif i == 9:
            # constraint on storage end vs start
            # Adding this does not seem to improve result (not lower CO2)
            if ("E_end" in model.paramDevice[dev]) and (t == model.setHorizon[-1]):
                lhs = model.varDeviceStorageEnergy[dev, t]
                # rhs = model.varDeviceStorageEnergy[dev,0] # end=start
                rhs = model.paramDevice[dev]["E_end"]
                return lhs == rhs
            else:
                return pyo.Constraint.Skip

    model.constrDevice_storage_el = pyo.Constraint(
        model.setDevice,
        model.setHorizon,
        pyo.RangeSet(1, 9),
        rule=rule_devmodel_storage_el,
    )

    logging.info("TODO: check liquid pump approximation")

    def rule_devmodel_pump(model, dev, t, i, carrier):
        if model.paramDevice[dev]["model"] != "pump_{}".format(carrier):
            return pyo.Constraint.Skip
        if i == 1:
            # flow out = flow in
            lhs = model.varDeviceFlow[dev, carrier, "out", t]
            rhs = model.varDeviceFlow[dev, carrier, "in", t]
            return lhs == rhs
        elif i == 2:
            # # power demand vs flow rate and pressure difference
            # # see eg. doi:10.1016/S0262-1762(07)70434-0
            # # P = Q*(p_out-p_in)/eta
            # # units: m3/s*MPa = MW
            # #
            # # Assuming incompressible fluid, so flow rate m3/s=Sm3/s
            # # (this approximation may not be very good for multiphase
            # # wellstream)
            # node = model.paramDevice[dev]['node']
            # eta = model.paramDevice[dev]['eta']
            # flowrate = model.varDeviceFlow[dev,carrier,'in',t]
            # # assume nominal pressure and keep only flow rate dependence
            # #TODO: Better linearisation?
            # p_in = model.paramNode[node]['pressure.{}.in'.format(carrier)]
            # p_out = model.paramNode[node]['pressure.{}.out'.format(carrier)]
            # delta_p = p_out - p_in
            # if self._quadraticConstraints:
            #     # Quadratic constraint...
            #     delta_p = (model.varPressure[(node,carrier,'out',t)]
            #                 -model.varPressure[(node,carrier,'in',t)])
            lhs = model.varDeviceFlow[dev, "el", "in", t]
            # rhs = flowrate*delta_p/eta
            """Device el demand"""
            rhs = milp_compute.compute_pump_demand(
                model, dev, linear=True, t=t, carrier=carrier
            )
            return lhs == rhs

    def rule_devmodel_pump_oil(model, dev, t, i):
        return rule_devmodel_pump(model, dev, t, i, carrier="oil")

    model.constrDevice_pump_oil = pyo.Constraint(
        model.setDevice,
        model.setHorizon,
        pyo.RangeSet(1, 2),
        rule=rule_devmodel_pump_oil,
    )

    def rule_devmodel_pump_water(model, dev, t, i):
        return rule_devmodel_pump(model, dev, t, i, carrier="water")

    model.constrDevice_pump_water = pyo.Constraint(
        model.setDevice,
        model.setHorizon,
        pyo.RangeSet(1, 2),
        rule=rule_devmodel_pump_water,
    )

    def rule_devmodel_pump_wellstream(model, dev, t, i):
        return rule_devmodel_pump(model, dev, t, i, carrier="wellstream")

    model.constrDevice_pump_wellstream = pyo.Constraint(
        model.setDevice,
        model.setHorizon,
        pyo.RangeSet(1, 2),
        rule=rule_devmodel_pump_wellstream,
    )

    def rule_devmodel_sink_gas(model, dev, t):
        if model.paramDevice[dev]["model"] != "sink_gas":
            return pyo.Constraint.Skip
        expr = pyo.Constraint.Skip
        return expr

    model.constrDevice_sink_gas = pyo.Constraint(
        model.setDevice, model.setHorizon, rule=rule_devmodel_sink_gas
    )

    def rule_devmodel_sink_oil(model, dev, t):
        if model.paramDevice[dev]["model"] != "sink_oil":
            return pyo.Constraint.Skip
        expr = pyo.Constraint.Skip
        return expr

    model.constrDevice_sink_oil = pyo.Constraint(
        model.setDevice, model.setHorizon, rule=rule_devmodel_sink_oil
    )

    def rule_devmodel_sink_el(model, dev, t):
        if model.paramDevice[dev]["model"] != "sink_el":
            return pyo.Constraint.Skip
        expr = pyo.Constraint.Skip
        return expr

    model.constrDevice_sink_el = pyo.Constraint(
        model.setDevice, model.setHorizon, rule=rule_devmodel_sink_el
    )

    def rule_devmodel_sink_heat(model, dev, t):
        if model.paramDevice[dev]["model"] != "sink_heat":
            return pyo.Constraint.Skip
        expr = pyo.Constraint.Skip
        return expr

    model.constrDevice_sink_heat = pyo.Constraint(
        model.setDevice, model.setHorizon, rule=rule_devmodel_sink_heat
    )

    def rule_devmodel_sink_water(model, dev, t, i):
        if model.paramDevice[dev]["model"] != "sink_water":
            return pyo.Constraint.Skip

        if "Qavg" not in model.paramDevice[dev]:
            return pyo.Constraint.Skip
        if "Vmax" not in model.paramDevice[dev]:
            return pyo.Constraint.Skip
        if model.paramDevice[dev]["Vmax"] == 0:
            return pyo.Constraint.Skip
        if i == 1:
            # FLEXIBILITY
            # (water_in-water_avg)*dt = delta buffer
            delta_t = model.paramParameters["time_delta_minutes"] / 60  # hours
            lhs = (
                model.varDeviceFlow[dev, "water", "in", t]
                - model.paramDevice[dev]["Qavg"]
            ) * delta_t
            if t > 0:
                Eprev = model.varDeviceStorageEnergy[dev, t - 1]
            else:
                Eprev = model.paramDeviceEnergyInitially[dev]
            rhs = model.varDeviceStorageEnergy[dev, t] - Eprev
            return lhs == rhs
        elif i == 2:
            # energy buffer limit
            Emax = model.paramDevice[dev]["Vmax"]
            return pyo.inequality(
                -Emax / 2, model.varDeviceStorageEnergy[dev, t], Emax / 2
            )
        return expr

    model.constrDevice_sink_water = pyo.Constraint(
        model.setDevice,
        model.setHorizon,
        pyo.RangeSet(1, 2),
        rule=rule_devmodel_sink_water,
    )

    # hydrogen flow:
    def rule_devmodel_electrolyser(model, dev, t, i):
        if model.paramDevice[dev]["model"] != "electrolyser":
            return pyo.Constraint.Skip
        energy_value = model.paramCarriers["hydrogen"]["energy_value"]  # MJ/Sm3
        efficiency = model.paramDevice[dev]["eta"]
        if i == 1:
            lhs = model.varDeviceFlow[dev, "hydrogen", "out", t] * energy_value
            rhs = model.varDeviceFlow[dev, "el", "in", t] * efficiency
            return lhs == rhs
        elif i == 2:
            """heat output = waste energy * heat recovery factor"""
            lhs = model.varDeviceFlow[dev, "heat", "out", t]
            eta_heat = model.paramDevice[dev]["eta_heat"]
            rhs = model.varDeviceFlow[dev, "el", "in", t] * (1 - efficiency) * eta_heat
            return lhs == rhs

    model.constrDevice_electrolyser = pyo.Constraint(
        model.setDevice,
        model.setHorizon,
        pyo.RangeSet(1, 2),
        rule=rule_devmodel_electrolyser,
    )

    def rule_devmodel_fuelcell(model, dev, t, i):
        if model.paramDevice[dev]["model"] != "fuelcell":
            return pyo.Constraint.Skip
        energy_value = model.paramCarriers["hydrogen"]["energy_value"]  # MJ/Sm3
        efficiency = model.paramDevice[dev]["eta"]
        if i == 1:
            """hydrogen to el"""
            lhs = model.varDeviceFlow[dev, "el", "out", t]  # MW
            rhs = (
                model.varDeviceFlow[dev, "hydrogen", "in", t]
                * energy_value
                * efficiency
            )
            return lhs == rhs
        elif i == 2:
            """heat output = waste energy * heat recovery factor"""
            lhs = model.varDeviceFlow[dev, "heat", "out", t]
            eta_heat = model.paramDevice[dev]["eta_heat"]
            rhs = (
                model.varDeviceFlow[dev, "hydrogen", "in", t]
                * energy_value
                * (1 - efficiency)
                * model.paramDevice[dev]["eta_heat"]
            )
            return lhs == rhs

    model.constrDevice_fuelcell = pyo.Constraint(
        model.setDevice,
        model.setHorizon,
        pyo.RangeSet(1, 2),
        rule=rule_devmodel_fuelcell,
    )

    def rule_devmodel_storage_hydrogen(model, dev, t, i):
        if model.paramDevice[dev]["model"] != "storage_hydrogen":
            return pyo.Constraint.Skip
        if i == 1:
            # energy balance (delta E = in - out) (energy in Sm3)
            delta_t = model.paramParameters["time_delta_minutes"] * 60  # seconds
            eta = 1
            if "eta" in model.paramDevice[dev]:
                eta = model.paramDevice[dev]["eta"]
            lhs = (
                model.varDeviceFlow[dev, "hydrogen", "in", t] * eta
                - model.varDeviceFlow[dev, "hydrogen", "out", t] / eta
            ) * delta_t
            if t > 0:
                Eprev = model.varDeviceStorageEnergy[dev, t - 1]
            else:
                Eprev = model.paramDeviceEnergyInitially[dev]
            rhs = model.varDeviceStorageEnergy[dev, t] - Eprev
            return lhs == rhs
        elif i == 2:
            # energy storage limit
            ub = model.paramDevice[dev]["Emax"]
            lb = 0
            if "Emin" in model.paramDevice[dev]:
                lb = model.paramDevice[dev]["Emin"]
            return pyo.inequality(lb, model.varDeviceStorageEnergy[dev, t], ub)
        elif i == 3:
            # Constraint 3 and 4: to represent absolute value in obj.function
            # see e.g. http://lpsolve.sourceforge.net/5.1/absolute.htm
            #
            # deviation from target and absolute value at the end of horizon
            if t != model.setHorizon[-1]:
                return pyo.Constraint.Skip
            Xprime = model.varDeviceStorageDeviationFromTarget[dev]
            # profile = model.paramDevice[dev]['target_profile']
            target_value = model.paramDeviceEnergyTarget[dev]
            deviation = model.varDeviceStorageEnergy[dev, t] - target_value
            return Xprime >= deviation
        elif i == 4:
            # deviation from target and absolute value at the end of horizon
            if t != model.setHorizon[-1]:
                return pyo.Constraint.Skip
            Xprime = model.varDeviceStorageDeviationFromTarget[dev]
            # profile = model.paramDevice[dev]['target_profile']
            target_value = model.paramDeviceEnergyTarget[dev]
            deviation = model.varDeviceStorageEnergy[dev, t] - target_value
            return Xprime >= -deviation

    model.constrDevice_storage_hydrogen = pyo.Constraint(
        model.setDevice,
        model.setHorizon,
        pyo.RangeSet(1, 4),
        rule=rule_devmodel_storage_hydrogen,
    )

    def rule_elReserveMargin(model, t):
        """Reserve margin constraint (electrical supply)
        Not used capacity by power suppliers/storage/load flexibility
        must be larger than some specified margin
        (to cope with unforeseen variations)
        """
        if ("elReserveMargin" not in model.paramParameters) or (
            model.paramParameters["elReserveMargin"] < 0
        ):
            if t == model.setHorizon.first():
                logging.info("Valid elReserveMargin not defined -> no constraint")
            return pyo.Constraint.Skip
        # exclude constraint for first timesteps since the point of the
        # dispatch margin is exactly to cope with forecast errors
        # *2 to make sure there is time to start up gt
        if t < model.paramParameters["forecast_timesteps"]:
            return pyo.Constraint.Skip

        margin = model.paramParameters["elReserveMargin"]
        capacity_unused = milp_compute.compute_elReserve(model, t)
        expr = capacity_unused >= margin
        return expr

    model.constrDevice_elReserveMargin = pyo.Constraint(
        model.setHorizon, rule=rule_elReserveMargin
    )

    def rule_elBackupMargin(model, dev, t):
        """Backup capacity constraint (electrical supply)
        Not used capacity by other online power suppliers plus sheddable
        load must be larger than power output of this device
        (to take over in case of a fault)
        """
        if ("elBackupMargin" not in model.paramParameters) or (
            model.paramParameters["elBackupMargin"] < 0
        ):
            if (dev == model.setDevice.first()) & (t == model.setHorizon.first()):
                logging.info("Valid elBackupMargin not defined -> no constraint")
            return pyo.Constraint.Skip
        margin = model.paramParameters["elBackupMargin"]
        devmodel = model.paramDevice[dev]["model"]
        if "el" not in Multicarrier.devicemodel_inout()[devmodel]["out"]:
            # this is not a power generator
            return pyo.Constraint.Skip
        res_otherdevs = milp_compute.compute_elReserve(model, t, exclude_device=dev)
        # elBackupMargin is zero or positive (if loss of load is acceptable)
        expr = res_otherdevs - model.varDeviceFlow[dev, "el", "out", t] >= -margin
        return expr

    model.constrDevice_elBackupMargin = pyo.Constraint(
        model.setDevice, model.setHorizon, rule=rule_elBackupMargin
    )

    def rule_startup_shutdown(model, dev, t):
        """startup/shutdown constraint
        connecting starting, stopping, preparation, online stages of GTs"""

        Tdelay = 0
        if "startupDelay" in model.paramDevice[dev]:
            Tdelay_min = model.paramDevice[dev]["startupDelay"]
            # Delay in timesteps, rounding down.
            # example: time_delta = 5 min, startupDelay= 8 min => Tdelay=1
            Tdelay = int(Tdelay_min / model.paramParameters["time_delta_minutes"])
        prevPart = 0
        if t >= Tdelay:
            # prevPart = sum( model.varDeviceStarting[dev,t-tau]
            #    for tau in range(0,Tdelay) )
            prevPart = model.varDeviceStarting[dev, t - Tdelay]
        else:
            # OBS: for this to work as intended, need to reconstruct constraint
            # pyo.value(...) not needed
            # prepInit = pyo.value(model.paramDevicePrepTimestepsInitially[dev])
            prepInit = model.paramDevicePrepTimestepsInitially[dev]
            if prepInit + t == Tdelay:
                prevPart = 1
        if t > 0:
            ison_prev = model.varDeviceIsOn[dev, t - 1]
        else:
            ison_prev = model.paramDeviceIsOnInitially[dev]
        lhs = model.varDeviceIsOn[dev, t] - ison_prev
        rhs = prevPart - model.varDeviceStopping[dev, t]
        return lhs == rhs

    model.constrDevice_startup_shutdown = pyo.Constraint(
        model.setDevice, model.setHorizon, rule=rule_startup_shutdown
    )

    def rule_startup_delay(model, dev, t):
        """startup delay/preparation for GTs"""
        Tdelay = 0
        if "startupDelay" in model.paramDevice[dev]:
            Tdelay_min = model.paramDevice[dev]["startupDelay"]
            # Delay in timesteps, rounding down.
            # example: time_delta = 5 min, startupDelay= 8 min => Tdelay=1
            Tdelay = int(Tdelay_min / model.paramParameters["time_delta_minutes"])
        else:
            return pyo.Constraint.Skip
        # determine if was in preparation previously
        # dependend on value - so must reconstruct constraint each time
        stepsPrevPrep = pyo.value(model.paramDevicePrepTimestepsInitially[dev])
        if stepsPrevPrep > 0:
            prevIsPrep = 1
        else:
            prevIsPrep = 0

        prevPart = 0
        if t < Tdelay - stepsPrevPrep:
            prevPart = prevIsPrep
        tau_range = range(0, min(t + 1, Tdelay))
        lhs = model.varDeviceIsPrep[dev, t]
        rhs = sum(model.varDeviceStarting[dev, t - tau] for tau in tau_range) + prevPart
        #            if (t==0) & (dev=="GT2"):
        #                print("rule startup delay (t=0)")
        #                print("prev steps=",stepsPrevPrep,"prevIsPrep=",prevIsPrep,
        #                    'prevpart=',prevPart,'Tdelay=',Tdelay)
        #                print("lhs=",lhs,"rhs=",rhs)
        return lhs == rhs

    model.constrDevice_startup_delay = pyo.Constraint(
        model.setDevice, model.setHorizon, rule=rule_startup_delay
    )

    def rule_ramprate(model, dev, t):
        """power ramp rate limit"""

        # If no ramp limits have been specified, skip constraint
        if ("maxRampUp" not in model.paramDevice[dev]) or pd.isna(
            model.paramDevice[dev]["maxRampUp"]
        ):
            return pyo.Constraint.Skip
        if t > 0:
            p_prev = milp_compute.getDevicePower(model, dev, t - 1)
        else:
            p_prev = model.paramDevicePowerInitially[dev]
        p_this = milp_compute.getDevicePower(model, dev, t)
        deltaP = milp_compute.getDevicePower(model, dev, t) - p_prev
        delta_t = model.paramParameters["time_delta_minutes"]
        maxP = model.paramDevice[dev]["Pmax"]
        max_neg = -model.paramDevice[dev]["maxRampDown"] * maxP * delta_t
        max_pos = model.paramDevice[dev]["maxRampUp"] * maxP * delta_t
        return pyo.inequality(max_neg, deltaP, max_pos)

    model.constrDevice_ramprate = pyo.Constraint(
        model.setDevice, model.setHorizon, rule=rule_ramprate
    )

    # Not so easy to formulate equal load sharing as a linear equation
    #
    # def rule_gasturbine_loadsharing(model,dev,t):
    #     '''ensure load is shared evently between online gas turbines'''
    #     if model.paramDevice[dev]['model'] != 'gasturbine':
    #         return pyo.Constraint.Skip
    #     for otherdev in model.setDevice:
    #         if (model.paramDevice[otherdev]['model'] == 'gasturbine'):
    #             rhs =
    #
    #     lhs = model.varDeviceFlow[dev,t,'el','out']
    #

    def rule_terminalEnergyBalance(model, carrier, node, terminal, t):
        """ node energy balance (at in and out terminals)
        "in" terminal: flow into terminal is positive (Pinj>0)
        "out" terminal: flow out of terminal is positive (Pinj>0)

        distinguishing between the case (1) where node/carrier has a
        single terminal or (2) with an "in" and one "out" terminal
        (with device in series) that may have different properties
        (such as gas pressure)

        edge1 \                                     / edge3
               \      devFlow_in    devFlow_out    /
                [term in]-->--device-->--[term out]
               /      \                       /    \
        edge2 /        \......termFlow......./      \ edge4


        """

        # Pinj = power injected into in terminal /out of out terminal
        Pinj = 0
        # devices:
        if node in model.paramNodeDevices:
            for dev in model.paramNodeDevices[node]:
                # Power into terminal:
                dev_model = model.paramDevice[dev]["model"]
                devmodels = milp_compute.devicemodel_inout()
                if dev_model in devmodels:
                    # print("carrier:{},node:{},terminal:{},model:{}"
                    #      .format(carrier,node,terminal,dev_model))
                    if carrier in devmodels[dev_model][terminal]:
                        Pinj -= model.varDeviceFlow[dev, carrier, terminal, t]
                else:
                    raise Exception("Undefined device model ({})".format(dev_model))

        # connect terminals (i.e. treat as one):
        if not model.paramNodeCarrierHasSerialDevice[node][carrier]:
            Pinj -= model.varTerminalFlow[node, carrier, t]

        # edges:
        if (carrier, node) in model.paramNodeEdgesTo and (terminal == "in"):
            for edg in model.paramNodeEdgesTo[(carrier, node)]:
                # power into node from edge
                Pinj += model.varEdgeFlow[edg, t]
        elif (carrier, node) in model.paramNodeEdgesFrom and (terminal == "out"):
            for edg in model.paramNodeEdgesFrom[(carrier, node)]:
                # power out of node into edge
                Pinj += model.varEdgeFlow[edg, t]

        expr = Pinj == 0
        if (type(expr) is bool) and (expr == True):
            expr = pyo.Constraint.Skip
        return expr

    model.constrTerminalEnergyBalance = pyo.Constraint(
        model.setCarrier,
        model.setNode,
        model.setTerminal,
        model.setHorizon,
        rule=rule_terminalEnergyBalance,
    )

    # # additional terminal energy/mass balance for mixed flow (wellstream)
    # def rule_terminalEnergyBalanceWithFlowComponents(
    #         model,node,terminal,fc,t):
    #     carrier="wellstream"
    #     Pinj = 0
    #     # devices:
    #     if (node in model.paramNodeDevices):
    #         for dev in model.paramNodeDevices[node]:
    #             # Power into terminal:
    #             dev_model = model.paramDevice[dev]['model']
    #             if carrier in self._devmodels[dev_model][terminal]:
    #                 GOR = model.paramDevice[dev]['gas_oil_ratio']
    #                 WC = model.paramDevice[dev]['water_cut']
    #                 comp_oil = (1-WC)/(1+GOR-GOR*WC)
    #                 comp_water = WC/(1+GOR-GOR*WC)
    #                 comp_gas = GOR*(1-WC)/(1+GOR*(1-WC))
    #                 comp={'oil':comp_oil,'gas':comp_gas,'water':comp_water}
    #                 Pinj -= comp[fc]*model.varDeviceFlow[dev,carrier,terminal,t]
    #
    #     # connect terminals (i.e. treat as one):
    #     if not model.paramNodeCarrierHasSerialDevice[node][carrier]:
    #         Pinj -= model.varTerminalFlowComponent[node,fc,t]
    #
    #     # edges:
    #     if (carrier,node) in model.paramNodeEdgesTo and (terminal=='in'):
    #         for edg in model.paramNodeEdgesTo[(carrier,node)]:
    #             # power into node from edge
    #             Pinj += (model.varEdgeFlowComponent[edg,fc,t])
    #     elif (carrier,node) in model.paramNodeEdgesFrom and (terminal=='out'):
    #         for edg in model.paramNodeEdgesFrom[(carrier,node)]:
    #             # power out of node into edge
    #             Pinj += (model.varEdgeFlowComponent[edg,fc,t])
    #
    #     expr = (Pinj==0)
    #     if ((type(expr) is bool) and (expr==True)):
    #         expr = pyo.Constraint.Skip
    #     return expr
    # model.constrTerminalEnergyBalanceWithFlowComponents = pyo.Constraint(
    #     model.setNode, model.setTerminal,model.setFlowComponent,
    #     model.setHorizon, rule=rule_terminalEnergyBalanceWithFlowComponents)

    #        logging.info("TODO: el power balance constraint redundant?")
    #        def rule_terminalElPowerBalance(model,node,t):
    #            ''' electric power balance at in and out terminals
    #            '''
    #            # Pinj = power injected into terminal
    #            Pinj = 0
    #            rhs = 0
    #            carrier='el'
    #            # devices:
    #            if (node in model.paramNodeDevices):
    #                for dev in model.paramNodeDevices[node]:
    #                    # Power into terminal:
    #                    dev_model = model.paramDevice[dev]['model']
    #                    if dev_model in self._devmodels:#model.paramDevicemodel:
    #                        if carrier in self._devmodels[dev_model]['in']:
    #                            Pinj -= model.varDeviceFlow[dev,carrier,'in',t]
    #                        if carrier in self._devmodels[dev_model]['out']:
    #                            Pinj += model.varDeviceFlow[dev,carrier,'out',t]
    #                    else:
    #                        raise Exception("Undefined device model ({})".format(dev_model))
    #
    #            # edges:
    #            # El linearised power flow equations:
    #            # Pinj = B theta
    #            n2s = [k[1]  for k in model.paramCoeffB.keys() if k[0]==node]
    #            for n2 in n2s:
    #                rhs -= model.paramCoeffB[node,n2]*(
    #                        model.varElVoltageAngle[n2,t]*self.elbase['baseAngle'])
    #            rhs = rhs*self.elbase['baseMVA']
    #
    #            expr = (Pinj==rhs)
    #            if ((type(expr) is bool) and (expr==True)):
    #                expr = pyo.Constraint.Skip
    #            return expr
    #        model.constrElPowerBalance = pyo.Constraint(model.setNode,
    #                    model.setHorizon, rule=rule_terminalElPowerBalance)

    def rule_elVoltageReference(model, t):
        n = model.paramParameters["reference_node"]
        expr = model.varElVoltageAngle[n, t] == 0
        return expr

    model.constrElVoltageReference = pyo.Constraint(
        model.setHorizon, rule=rule_elVoltageReference
    )

    logging.info("TODO: equation for flow vs pressure of liquids")

    def rule_edgeFlowEquations(model, edge, t):
        """Flow as a function of node values (voltage/pressure)"""
        carrier = model.paramEdge[edge]["type"]
        n_from = model.paramEdge[edge]["nodeFrom"]
        n_to = model.paramEdge[edge]["nodeTo"]

        if carrier == "el":
            """power flow vs voltage angle difference
            Linearised power flow equations (DC power flow)"""
            baseMVA = electricalsystem.elbase["baseMVA"]
            baseAngle = electricalsystem.elbase["baseAngle"]
            lhs = model.varEdgeFlow[edge, t]
            lhs = lhs / baseMVA
            rhs = 0
            # TODO speed up creatioin of constraints - remove for loop
            n2s = [k[1] for k in model.paramCoeffDA.keys() if k[0] == edge]
            for n2 in n2s:
                rhs += model.paramCoeffDA[edge, n2] * (
                    model.varElVoltageAngle[n2, t] * baseAngle
                )
            return lhs == rhs

        elif carrier in ["gas", "wellstream", "oil", "water"]:
            p1 = model.varPressure[(n_from, carrier, "out", t)]
            p2 = model.varPressure[(n_to, carrier, "in", t)]
            Q = model.varEdgeFlow[edge, t]
            if "num_pipes" in model.paramEdge[edge]:
                num_pipes = model.paramEdge[edge]["num_pipes"]
                logging.debug("{},{}: {} parallel pipes".format(edge, t, num_pipes))
                Q = Q / num_pipes
            p2_computed = milp_compute.compute_edge_pressuredrop(
                model, edge, p1=p1, Q=Q, linear=True
            )
            return p2 == p2_computed
        #            elif carrier in ['wellstream','water']:
        #                p_from = model.varPressure[(n_from,carrier,'out',t)]
        #                p_to = model.varPressure[(n_to,carrier,'in',t)]
        #                return (p_from==p_to)
        else:
            # Other types of edges - no constraints other than max/min flow
            return pyo.Constraint.Skip

    model.constrEdgeFlowEquations = pyo.Constraint(
        model.setEdge, model.setHorizon, rule=rule_edgeFlowEquations
    )

    def rule_pressureAtNode(model, node, carrier, t):
        if carrier in ["el", "heat"]:
            return pyo.Constraint.Skip
        elif model.paramNodeCarrierHasSerialDevice[node][carrier]:
            # pressure in and out are related via device equations for
            # device connected between in and out terminals. So no
            # extra constraint required
            return pyo.Constraint.Skip
        else:
            # single terminal. (pressure out=pressure in)
            expr = (
                model.varPressure[(node, carrier, "out", t)]
                == model.varPressure[(node, carrier, "in", t)]
            )
            return expr

    model.constrPressureAtNode = pyo.Constraint(
        model.setNode, model.setCarrier, model.setHorizon, rule=rule_pressureAtNode
    )

    def rule_pressureBounds(model, node, term, carrier, t):
        col = "pressure.{}.{}".format(carrier, term)
        if not col in model.paramNode[node]:
            # no pressure data relevant for this node/carrier
            return pyo.Constraint.Skip
        nom_p = model.paramNode[node][col]
        if pd.isna(nom_p):
            # no pressure data specified for this node/carrier
            return pyo.Constraint.Skip
        cc = "maxdeviation_pressure.{}.{}".format(carrier, term)
        if (cc in model.paramNode[node]) and (not np.isnan(model.paramNode[node][cc])):
            maxdev = model.paramNode[node][cc]
            if t == 0:
                logging.debug(
                    "Using ind. pressure limit for: {}, {}, {}".format(node, cc, maxdev)
                )
        else:
            maxdev = model.paramParameters["max_pressure_deviation"]
            if maxdev == -1:
                return pyo.Constraint.Skip
        lb = nom_p * (1 - maxdev)
        ub = nom_p * (1 + maxdev)
        return (lb, model.varPressure[(node, carrier, term, t)], ub)

    model.constrPressureBounds = pyo.Constraint(
        model.setNode,
        model.setTerminal,
        model.setCarrier,
        model.setHorizon,
        rule=rule_pressureBounds,
    )

    def rule_devicePmax(model, dev, t):
        # max/min power (zero if device is not on)
        if "Pmax" not in model.paramDevice[dev]:
            return pyo.Constraint.Skip
        maxValue = model.paramDevice[dev]["Pmax"]
        if "profile" in model.paramDevice[dev]:
            # use an availability profile if provided
            extprofile = model.paramDevice[dev]["profile"]
            maxValue = maxValue * model.paramProfiles[extprofile, t]
        ison = 1
        if model.paramDevice[dev]["model"] in ["gasturbine"]:
            ison = model.varDeviceIsOn[dev, t]
        power = milp_compute.getDevicePower(model, dev, t)
        expr = power <= ison * maxValue
        return expr

    model.constrDevicePmax = pyo.Constraint(
        model.setDevice, model.setHorizon, rule=rule_devicePmax
    )

    def rule_devicePmin(model, dev, t):
        if "Pmin" not in model.paramDevice[dev]:
            return pyo.Constraint.Skip
        minValue = model.paramDevice[dev]["Pmin"]
        if "profile" in model.paramDevice[dev]:
            # use an availability profile if provided
            extprofile = model.paramDevice[dev]["profile"]
            minValue = minValue * model.paramProfiles[extprofile, t]
        ison = 1
        if model.paramDevice[dev]["model"] in ["gasturbine"]:
            ison = model.varDeviceIsOn[dev, t]
        power = milp_compute.getDevicePower(model, dev, t)
        expr = power >= ison * minValue
        return expr

    model.constrDevicePmin = pyo.Constraint(
        model.setDevice, model.setHorizon, rule=rule_devicePmin
    )

    def rule_deviceQmax(model, dev, t):
        # max/min power (zero if device is not on)
        if "Qmax" not in model.paramDevice[dev]:
            return pyo.Constraint.Skip
        maxValue = model.paramDevice[dev]["Qmax"]
        if "profile" in model.paramDevice[dev]:
            # use an availability profile if provided
            extprofile = model.paramDevice[dev]["profile"]
            maxValue = maxValue * model.paramProfiles[extprofile, t]
        flow = milp_compute.getDeviceFlow(model, dev, t)
        expr = flow <= maxValue
        return expr

    model.constrDeviceQmax = pyo.Constraint(
        model.setDevice, model.setHorizon, rule=rule_deviceQmax
    )

    def rule_deviceQmin(model, dev, t):
        if "Qmin" not in model.paramDevice[dev]:
            return pyo.Constraint.Skip
        minValue = model.paramDevice[dev]["Qmin"]
        if "profile" in model.paramDevice[dev]:
            # use an availability profile if provided
            extprofile = model.paramDevice[dev]["profile"]
            minValue = minValue * model.paramProfiles[extprofile, t]
        flow = milp_compute.getDeviceFlow(model, dev, t)
        expr = flow >= minValue
        return expr

    model.constrDeviceQmin = pyo.Constraint(
        model.setDevice, model.setHorizon, rule=rule_deviceQmin
    )

    def rule_edgePmaxmin(model, edge, t):
        edgetype = model.paramEdge[edge]["type"]
        expr = pyo.Constraint.Skip  # default if Pmax/Qmax has not been set
        if edgetype in ["el", "heat"]:
            if "Pmax" in model.paramEdge[edge]:
                expr = pyo.inequality(
                    -model.paramEdge[edge]["Pmax"],
                    model.varEdgeFlow[edge, t],
                    model.paramEdge[edge]["Pmax"],
                )
        elif edgetype in ["wellstream", "gas", "oil", "water", "hydrogen"]:
            if "Qmax" in model.paramEdge[edge]:
                expr = model.varEdgeFlow[edge, t] <= model.paramEdge[edge]["Qmax"]
        else:
            raise Exception("Unknown edge type ({})".format(edgetype))
        return expr

    model.constrEdgeBounds = pyo.Constraint(
        model.setEdge, model.setHorizon, rule=rule_edgePmaxmin
    )

    def rule_emissionRateLimit(model, t):
        """Upper limit on CO2 emission rate"""
        emissionRateMax = model.paramParameters["emissionRateMax"]
        if emissionRateMax < 0:
            # don't include constraint if parameter is set to negative value
            return pyo.Constraint.Skip
        else:
            lhs = milp_compute.compute_CO2(model, timesteps=[t])
            rhs = emissionRateMax
            return lhs <= rhs

    model.constrEmissionRateLimit = pyo.Constraint(
        model.setHorizon, rule=rule_emissionRateLimit
    )

    def rule_emissionIntensityLimit(model, t):
        """Upper limit on CO2 emission intensity"""
        emissionIntensityMax = model.paramParameters["emissionIntensityMax"]
        if emissionIntensityMax < 0:
            # don't include constraint if parameter is set to negative value
            return pyo.Constraint.Skip
        else:
            lhs = milp_compute.compute_CO2(model, timesteps=[t])
            rhs = emissionIntensityMax * milp_compute.compute_oilgas_export(
                model, timesteps=[t]
            )
            return lhs <= rhs

    model.constrEmissionIntensityLimit = pyo.Constraint(
        model.setHorizon, rule=rule_emissionIntensityLimit
    )

    return model
    # END class init.
