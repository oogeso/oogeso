import logging
import pandas as pd
import pyomo.environ as pyo
import pyomo.opt as pyopt

from oogeso.core.devices.storage import _StorageDevice
from . import devices, networks
from .networks import electricalsystem as el_calc
from .networks.network_node import NetworkNode
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from oogeso.dto.oogeso_input_data_objects import (
        EnergySystemData,
        OptimisationParametersData,
    )
    from oogeso.core.devices.device import Device
    from oogeso.core.networks.edge import Edge
    from oogeso.core.networks.network import Network

logger = logging.getLogger(__name__)


class OptimisationModel:
    """Class for MILP optimisation model"""

    ZERO_WARNING_THRESHOLD = 1e-6

    def __init__(self, data: "EnergySystemData"):
        """Create optimisation problem formulation with supplied data"""

        # dictionaries {key:object} for all devices, nodes and edges
        self.all_devices: Dict[str, Device] = {}
        self.all_nodes: Dict[str, NetworkNode] = {}
        self.all_edges: Dict[str, Edge] = {}
        #        self.all_carriers = {}
        self.all_networks: Dict[str, Network] = {}
        self.optimisation_parameters: OptimisationParametersData = data.parameters
        self.pyomo_instance = None
        # List of constraints that need to be reconstructed for each optimisation:
        self.constraints_to_reconstruct = []
        # List of devices with storage
        self.devices_with_storage = []
        profiles_in_use = list(
            set(d.profile for d in data.devices if d.profile is not None)
        )
        logger.info("profiles in use: %s", profiles_in_use)

        self._create_network_objects_from_data(data)
        self._setNodePressureFromEdgeData()
        self.pyomo_instance = self._create_pyomo_model(profiles_in_use)

    def solve(self, solver="gurobi", write_yaml=False, timelimit=None):
        """Solve problem for planning horizon at a single timestep"""

        opt = pyo.SolverFactory(solver)
        if timelimit is not None:
            if solver == "gurobi":
                opt.options["TimeLimit"] = timelimit
            elif solver == "cbc":
                opt.options["sec"] = timelimit
            elif solver == "cplex":
                opt.options["timelimit"] = timelimit
            elif solver == "glpk":
                opt.options["tmlim"] = timelimit
        logger.debug("Solving...")
        sol = opt.solve(self.pyomo_instance)

        if write_yaml:
            sol.write_yaml()

        if (sol.solver.status == pyopt.SolverStatus.ok) and (
            sol.solver.termination_condition == pyopt.TerminationCondition.optimal
        ):
            logger.debug("Solved OK")
            pass
        elif sol.solver.termination_condition == pyopt.TerminationCondition.infeasible:
            raise Exception("Infeasible solution")
        else:
            # Something else is wrong
            logger.info("Solver Status:{}".format(sol.solver.status))
        return sol

    def _setNodePressureFromEdgeData(self):
        # Set node terminal nominal pressure based on edge from/to pressure values
        for edge in self.all_edges.values():
            edg = edge.edge_data
            carrier = edg.carrier
            # Setting nominal pressure levels at node terminals. Raise exception
            # if inconsistencies are found
            if (hasattr(edg, "pressure_from")) and (edg.pressure_from is not None):
                n_from: NetworkNode = self.all_nodes[edg.node_from]
                p_from = edg.pressure_from
                n_from.set_nominal_pressure(carrier, "out", p_from)
            if (hasattr(edg, "pressure_to")) and (edg.pressure_to is not None):
                n_to: NetworkNode = self.all_nodes[edg.node_to]
                p_to = edg.pressure_to
                n_to.set_nominal_pressure(carrier, "in", p_to)

    def _create_network_objects_from_data(self, data: "EnergySystemData"):
        """Create energy system objects, and populate local dictionaries

        self.all_devices, self.all_nodes, self.all_networks"""

        # json_str = json.dumps(data)
        # energy_system_data: inputdata.EnergySystemData = (
        #    inputdata.deserialize_oogeso_data(json_str)
        # )
        # energy_system_data = json.loads(json_str, cls=inputdata.DataclassJSONDecoder)
        energy_system_data = data

        # Create energy system network elements (devices, nodes, edges)
        for dev_data_obj in energy_system_data.devices:
            dev_id = dev_data_obj.id
            if dev_data_obj.include == False:
                # skip this edge and move to next
                logger.debug("Excluding device {}".format(dev_id))
                continue
            device_model = dev_data_obj.model
            # The class corresponding to the device type should always have a
            # name identical to the type (but capitalized):
            logger.debug("Device model={}".format(device_model))
            Devclass = getattr(devices, device_model.capitalize())
            carrier_data_dict = {carr_obj.id: carr_obj for carr_obj in data.carriers}
            new_device = Devclass(dev_data_obj, carrier_data_dict)
            if isinstance(new_device, _StorageDevice):
                # Add this device to global list of storage devices:
                self.devices_with_storage.append(new_device)

            new_device.setFlowUpperBound(data.profiles)
            self.all_devices[dev_id] = new_device

        for node_data_obj in energy_system_data.nodes:
            new_node = networks.NetworkNode(node_data_obj)
            node_id = new_node.id
            self.all_nodes[node_id] = new_node

        edges_per_type = {}
        for edge_data_obj in data.edges:
            if edge_data_obj.include == False:
                # skip this edge and move to next
                continue
            edge_id = edge_data_obj.id
            carrier = edge_data_obj.carrier
            new_edge = networks.Edge(edge_data_obj)
            if carrier not in edges_per_type:
                edges_per_type[carrier] = {}
            edges_per_type[carrier][edge_id] = new_edge
            self.all_edges[edge_id] = new_edge

        for carrier_data_obj in data.carriers:
            carrier_model = carrier_data_obj.id  # el,heat,oil,gas,water,hydrogen,
            # The network class corresponding to the carrier should always have a
            # name identical to the id (but capitalized):
            NetworkClass = getattr(networks, carrier_model.capitalize())
            if carrier_model not in edges_per_type:
                edges_per_type[carrier_model] = {}
            new_network = NetworkClass(
                carrier_data=carrier_data_obj,
                edges=edges_per_type[carrier_model],
            )
            self.all_networks[carrier_model] = new_network

        # Make network connections between nodes, edges and devices
        for dev_id, dev in self.all_devices.items():
            logger.debug("Node-device: %s,%s", dev_id, dev)
            node_id_where_connected = dev.dev_data.node_id
            node = self.all_nodes[node_id_where_connected]
            node.addDevice(dev_id, dev)
            dev.addNode(node)
        for edge_id, edge in self.all_edges.items():
            node_from = self.all_nodes[edge.edge_data.node_from]
            node_from.addEdge(edge, "from")
            node_to = self.all_nodes[edge.edge_data.node_to]
            node_to.addEdge(edge, "to")
            edge.addNodes(node_from, node_to)

    def _create_pyomo_model(self, profiles_in_use):
        """Create pyomo MILP model

        Parameters:
        -----------
            planning_horizon: dict
            input data as read from input file
        """

        model = pyo.ConcreteModel()
        model = self._specify_sets_and_parameters(model, profiles_in_use)
        model = self._specify_variables(model)
        model = self._specify_objective(model)
        model = self._specify_constraints(model)

        # Specify initial values from input data
        for dev in self.all_devices.values():
            dev.setInitValues(model)

        # Keep track of duals:
        # WARNING:
        # From Gurobi: "Shadow prices are not well-defined in mixed-integer
        # problems, so we don't provide dual values for an integer program."
        model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

        return model

    def _specify_sets_and_parameters(self, model, profiles_in_use):
        """specify pyomo model sets and parameters"""
        energycarriers = self.all_networks.keys()
        model.setCarrier = pyo.Set(initialize=energycarriers, doc="carrier")
        model.setTerminal = pyo.Set(initialize=["in", "out"], doc="terminal")
        model.setNode = pyo.Set(initialize=self.all_nodes.keys(), doc="node")
        model.setDevice = pyo.Set(initialize=self.all_devices.keys(), doc="device")
        model.setEdge = pyo.Set(initialize=self.all_edges.keys(), doc="edge")
        timerange = range(self.optimisation_parameters.planning_horizon)
        model.setHorizon = pyo.Set(ordered=True, initialize=timerange, doc="time")
        model.setProfile = pyo.Set(initialize=profiles_in_use, doc="profile")

        # specify mutable parameters:
        # (will be modified between successive optimisations)
        model.paramProfiles = pyo.Param(
            model.setProfile,
            model.setHorizon,
            within=pyo.Reals,
            mutable=True,
            initialize=1,
        )
        model.paramDeviceIsOnInitially = pyo.Param(
            model.setDevice, mutable=True, within=pyo.Binary, initialize=1
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

        # Specify handy immutable parameters (for easy access when building/updating model)
        model.paramTimestepDeltaMinutes = pyo.Param(
            within=pyo.Reals, default=self.optimisation_parameters.time_delta_minutes
        )
        model.paramTimeStorageReserveMinutes = pyo.Param(
            default=self.optimisation_parameters.time_reserve_minutes
        )
        model.paramPiecewiseRepn = pyo.Param(
            within=pyo.Any, default=self.optimisation_parameters.piecewise_repn
        )
        model.paramMaxPressureDeviation = pyo.Param(
            within=pyo.Reals,
            default=self.optimisation_parameters.max_pressure_deviation,
        )

        return model

    def _specify_variables(self, model):
        """specify pyomo model variables"""
        model.varEdgeFlow = pyo.Var(model.setEdge, model.setHorizon, within=pyo.Reals)
        model.varEdgeFlow12 = pyo.Var(
            model.setEdge, model.setHorizon, within=pyo.NonNegativeReals
        )
        model.varEdgeFlow21 = pyo.Var(
            model.setEdge, model.setHorizon, within=pyo.NonNegativeReals
        )
        model.varEdgeLoss = pyo.Var(
            model.setEdge, model.setHorizon, within=pyo.NonNegativeReals, initialize=0
        )
        model.varEdgeLoss12 = pyo.Var(
            model.setEdge, model.setHorizon, within=pyo.NonNegativeReals, initialize=0
        )
        model.varEdgeLoss21 = pyo.Var(
            model.setEdge, model.setHorizon, within=pyo.NonNegativeReals, initialize=0
        )
        model.varDeviceIsPrep = pyo.Var(
            model.setDevice, model.setHorizon, within=pyo.Binary, initialize=0
        )
        model.varDeviceIsOn = pyo.Var(
            model.setDevice, model.setHorizon, within=pyo.Binary, initialize=1
        )
        model.varDeviceStarting = pyo.Var(
            model.setDevice, model.setHorizon, within=pyo.Binary, initialize=0
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
        model.varElVoltageAngle = pyo.Var(
            model.setNode, model.setHorizon, within=pyo.Reals
        )
        model.varDeviceFlow = pyo.Var(
            model.setDevice,
            model.setCarrier,
            model.setTerminal,
            model.setHorizon,
            within=pyo.NonNegativeReals,
            initialize=0,
        )
        model.varTerminalFlow = pyo.Var(
            model.setNode, model.setCarrier, model.setHorizon, within=pyo.Reals
        )
        # this penalty variable should only require (device,time), but the
        # piecewise constraint requires the domain to be the same as for varDeviceFlow
        model.varDevicePenalty = pyo.Var(
            model.setDevice,
            model.setCarrier,
            model.setTerminal,
            model.setHorizon,
            within=pyo.Reals,
        )
        return model

    def _specify_objective(self, model):
        """specify pyomo model objective"""
        obj = self.optimisation_parameters.objective
        if obj == "penalty":
            rule = self._rule_objective_penalty
        elif obj == "co2":
            rule = self._rule_objective_co2
        elif obj == "costs":
            rule = self._rule_objective_costs
        elif obj == "exportRevenue":
            rule = self._rule_objective_exportRevenue
        elif obj == "co2intensity":
            rule = self._rule_objective_co2intensity
        else:
            raise Exception("Objective '{}' has not been implemented".format(obj))
        logger.info("Using objective function: {}".format(obj))
        # logger.info(rule)
        model.objObjective = pyo.Objective(rule=rule, sense=pyo.minimize)
        return model

    def _specify_constraints(self, model):

        # 1. Constraints associated with each device:
        for dev in self.all_devices.values():
            list_to_reconstruct = dev.defineConstraints(model)

            # Because of logic that needs to be re-evalued, these constraints need
            # to be reconstructed each optimisation:
            for constr in list_to_reconstruct:
                self.constraints_to_reconstruct.append(constr)

        # 2. Constraints associated with each node:
        for node in self.all_nodes.values():
            node.defineConstraints(model)

        # 3. Constraints associated with each network type (and its edges):
        for netw in self.all_networks.values():
            netw.defineConstraints(model)

        # 4. Global constraints:
        # 4.1 max limit emission rate:
        params_generic = self.optimisation_parameters
        if (params_generic.emission_rate_max is not None) and (
            params_generic.emission_rate_max >= 0
        ):
            model.constrO_emissionrate = pyo.Constraint(
                model.setHorizon, rule=self._rule_emissionRateLimit
            )
        else:
            logger.debug("No emission_rate_max limit specified")
        # 4.2 max limit emission intensity
        if (params_generic.emission_intensity_max is not None) and (
            params_generic.emission_intensity_max >= 0
        ):
            model.constrO_emissionintensity = pyo.Constraint(
                model.setHorizon, rule=self._rule_emissionIntensityLimit
            )
        else:
            logger.debug("No emission_intensity_max limit specified")
        # 4.3 electrical reserve margin:
        el_parameters = self.all_networks["el"].carrier_data
        el_reserve_margin = el_parameters.el_reserve_margin
        el_backup_margin = el_parameters.el_backup_margin
        if (el_reserve_margin is not None) and (el_reserve_margin >= 0):
            model.constrO_elReserveMargin = pyo.Constraint(
                model.setHorizon, rule=self._rule_elReserveMargin
            )
        else:
            logger.info("No el_reserve_margin limit specified")
        # 4.4 electrical backup power margin
        if (el_backup_margin is not None) and (el_backup_margin >= 0):
            model.constrO_elBackupMargin = pyo.Constraint(
                model.setDevice,
                model.setHorizon,
                rule=self._rule_elBackupMargin,
            )
        else:
            logger.debug("No el_backup_margin limit specified")

        return model

    def updateOptimisationModel(self, timestep, profiles, first=False):
        """Update Pyomo model instance

        Parameters
        ----------
        timestep : int
            Present timestep
        profiles : pandas dataframe
            Time series profiles
        first : booblean
            True if it is the first time the model is updated (simulation start)
        """
        opt_timesteps = self.optimisation_parameters.optimisation_timesteps
        horizon = self.optimisation_parameters.planning_horizon
        timesteps_use_nowcast = self.optimisation_parameters.forecast_timesteps

        # Update profile (using nowcast for first 4 timesteps, forecast for rest)
        # -this is because for the first timesteps, we tend to have updated
        #  and quite good forecasts - for example, it may become apparent
        #  that there will be much less wind power than previously forecasted
        for prof in self.pyomo_instance.setProfile:
            for t in range(timesteps_use_nowcast):  # 0,1,2,3
                profile_str = "nowcast"
                if prof not in profiles["nowcast"]:
                    # no nowcast, use forecast instead
                    profile_str = "forecast"
                self.pyomo_instance.paramProfiles[prof, t] = profiles[profile_str].loc[
                    timestep + t, prof
                ]
            for t in range(timesteps_use_nowcast, horizon):
                self.pyomo_instance.paramProfiles[prof, t] = profiles["forecast"].loc[
                    timestep + t, prof
                ]

        def _updateOnTimesteps(t_prev, dev):
            # sum up consequtive timesteps starting at tprev going
            # backwards, where device has been in preparation phase
            sum_on = 0
            docontinue = True
            for tt in range(t_prev, -1, -1):
                # if (self.instance.varDeviceIsOn[dev,tt]==1):
                if pyo.value(self.pyomo_instance.varDeviceIsPrep[dev, tt]) == 1:
                    sum_on = sum_on + 1
                else:
                    docontinue = False
                    break  # exit for loop
            if docontinue:
                # we got all the way back to 0, so must include initial value
                sum_on = (
                    sum_on + self.pyomo_instance.paramDevicePrepTimestepsInitially[dev]
                )
            return sum_on

        # Update startup/shutdown info
        # pick the last value from previous optimistion prior to the present time
        if not first:
            t_prev = opt_timesteps - 1
            for dev, dev_obj in self.all_devices.items():
                # On/off status: (round because solver doesn't alwasy return an integer)
                self.pyomo_instance.paramDeviceIsOnInitially[dev] = round(
                    pyo.value(self.pyomo_instance.varDeviceIsOn[dev, t_prev])
                )
                self.pyomo_instance.paramDevicePrepTimestepsInitially[
                    dev
                ] = _updateOnTimesteps(t_prev, dev)
                # Initial power output (relevant for ramp rate constraint):
                if dev_obj.dev_data.max_ramp_up is not None:
                    self.pyomo_instance.paramDevicePowerInitially[
                        dev
                    ] = dev_obj.getFlowVar(self.pyomo_instance, t_prev)
                # Energy storage:
                if dev_obj in self.devices_with_storage:
                    self.pyomo_instance.paramDeviceEnergyInitially[
                        dev
                    ] = self.pyomo_instance.varDeviceStorageEnergy[dev, t_prev]
                    # Update target profile if present:
                    if hasattr(dev_obj.dev_data, "target_profile") and (
                        dev_obj.dev_data.target_profile is not None
                    ):
                        prof = dev_obj.dev_data.target_profile
                        max_E = dev_obj.dev_data.E_max
                        self.pyomo_instance.paramDeviceEnergyTarget[dev] = (
                            max_E * profiles["forecast"].loc[timestep + horizon, prof]
                        )

        # These constraints need to be reconstructed to update properly
        # (pyo.value(...) and/or if sentences...)
        for c in self.constraints_to_reconstruct:
            # logger.debug("reconstructing {}".format(c))
            # c.reconstruct() <- removed in Pyomo v.6
            c.clear()
            c._constructed = False
            c.construct()
        return

    #        def storPmaxPushup(model):
    #            '''term in objective function to push varDeviceStoragePmax up
    #            to its maximum value (to get correct calculation of reserve)'''
    #            sumStorPmax=0
    #            for dev in model.setDevice:
    #                if model.paramDevice[dev]['model'] == 'storage_el':
    #                    for t in model.setHorizon:
    #                        sumStorPmax += model.varDeviceStoragePmax[dev,t]
    #            return sumStorPmax

    def _rule_objective_penalty(self, model):
        """'penalty' as specified through penalty functions"""
        sum_penalty = 0
        timesteps = model.setHorizon
        for d in model.setDevice:
            dev = self.all_devices[d]
            this_penalty = dev.compute_penalty(model, timesteps)
            sum_penalty = sum_penalty + this_penalty
        return sum_penalty

    def _rule_objective_co2(self, model):
        """CO2 emissions per sec"""
        sumE = self.compute_CO2(model)  # *model.paramParameters['CO2_price']
        return sumE

    def _rule_objective_co2intensity(self, model):
        """CO2 emission intensity (CO2 per exported oil/gas)
        DOES NOT WORK - NONLINEAR (ratio)"""
        sumE = self.compute_CO2_intensity(model)
        return sumE

    def _rule_objective_costs(self, model):
        """costs (co2 price, operating costs, startstop, storage depletaion)
        per second (assuming fixed oil/gas production)"""
        startupCosts = self.compute_startup_penalty(model)  # kr/s
        storageDepletionCosts = self.compute_costForDepletedStorage(model)
        opCosts = self.compute_operatingCosts(model)  # kr/s
        co2 = self.compute_CO2(model)  # kgCO2/s
        co2_tax = self.optimisation_parameters.co2_tax  # kr/kgCO2
        co2Cost = co2 * co2_tax  # kr/s
        sumCost = co2Cost + startupCosts + storageDepletionCosts + opCosts
        return sumCost

    def _rule_objective_exportRevenue(self, model):
        """revenue from exported oil and gas minus costs (co2 price and
        operating costs) per second"""
        sumRevenue = self.compute_exportRevenue(model)  # kr/s
        startupCosts = self.compute_startup_penalty(model)  # kr/s
        co2 = self.compute_CO2(model)  # kgCO2/s
        co2_tax = self.optimisation_parameters.co2_tax  # kr/kgCO2
        co2Cost = co2 * co2_tax  # kr/s
        storageDepletionCosts = self.compute_costForDepletedStorage(model)
        opCosts = self.compute_operatingCosts(model)  # kr/s
        sumCost = -sumRevenue + co2Cost + startupCosts + storageDepletionCosts + opCosts
        return sumCost

    def _rule_emissionRateLimit(self, model, t):
        """Upper limit on CO2 emission rate"""
        params_generic = self.optimisation_parameters
        emissionRateMax = params_generic.emission_rate_max
        lhs = self.compute_CO2(model, timesteps=[t])
        rhs = emissionRateMax
        return lhs <= rhs

    def _rule_emissionIntensityLimit(self, model, t):
        """Upper limit on CO2 emission intensity"""
        params_generic = self.optimisation_parameters
        emissionIntensityMax = params_generic.emission_intensity_max
        lhs = self.compute_CO2(model, timesteps=[t])
        rhs = emissionIntensityMax * self.compute_oilgas_export(model, timesteps=[t])
        return lhs <= rhs

    def _rule_elReserveMargin(self, model, t):
        """Reserve margin constraint (electrical supply)
        Not used capacity by power suppliers/storage/load flexibility
        must be larger than some specified margin
        (to cope with unforeseen variations)
        """
        # exclude constraint for first timesteps since the point of the
        # dispatch margin is exactly to cope with forecast errors
        if t < self.optimisation_parameters.forecast_timesteps:
            return pyo.Constraint.Skip

        network_el = self.all_networks["el"]
        margin = network_el.carrier_data.el_reserve_margin
        capacity_unused = network_el.compute_elReserve(model, t, self.all_devices)
        expr = capacity_unused >= margin
        return expr

    def _rule_elBackupMargin(self, model, dev, t):
        """Backup capacity constraint (electrical supply)
        Not used capacity by other online power suppliers plus sheddable
        load must be larger than power output of this device
        (to take over in case of a fault)

        elBackupMargin is zero or positive (if loss of load is acceptable)
        """
        dev_obj = self.all_devices[dev]
        network_el = self.all_networks["el"]
        margin = network_el.carrier_data.el_backup_margin
        if "el" not in dev_obj.carrier_out:
            # this is not a power generator
            return pyo.Constraint.Skip
        res_otherdevs = network_el.compute_elReserve(
            model, t, self.all_devices, exclude_device=dev
        )
        expr = res_otherdevs - model.varDeviceFlow[dev, "el", "out", t] >= -margin
        return expr

    def compute_CO2(self, model, devices=None, timesteps=None):
        """compute CO2 emissions - average per sec (kgCO2/s)"""
        if devices is None:
            devices = model.setDevice
        if timesteps is None:
            timesteps = model.setHorizon
        sumCO2 = 0
        for d in devices:
            dev = self.all_devices[d]
            thisCO2 = dev.compute_CO2(model, timesteps)
            sumCO2 = sumCO2 + thisCO2
        # Average per s
        sumCO2 = sumCO2 / len(timesteps)
        return sumCO2

    def compute_CO2_intensity(self, model, timesteps=None):
        """CO2 emission per exported oil/gas (kgCO2/Sm3oe)"""
        if timesteps is None:
            timesteps = model.setHorizon

        co2_kg_per_time = self.compute_CO2(model, devices=None, timesteps=timesteps)
        flow_oilequivalents_m3_per_time = self.compute_oilgas_export(model, timesteps)
        if pyo.value(flow_oilequivalents_m3_per_time) != 0:
            co2intensity = co2_kg_per_time / flow_oilequivalents_m3_per_time
        if pyo.value(flow_oilequivalents_m3_per_time) == 0:
            # logger.debug("zero export, so co2 intensity set to None")
            co2intensity = None
        return co2intensity

    def compute_startup_penalty(self, model, devices=None, timesteps=None):
        """startup costs (average per sec)"""
        if timesteps is None:
            timesteps = model.setHorizon
        if devices is None:
            devices = model.setDevice
        start_stop_costs = 0
        for d in devices:
            dev_obj = self.all_devices[d]
            thisCost = dev_obj.compute_startup_penalty(model, timesteps)
            start_stop_costs += thisCost
        # get average per sec:
        deltaT = self.optimisation_parameters.time_delta_minutes * 60
        sumTime = len(timesteps) * deltaT
        start_stop_costs = start_stop_costs / sumTime
        return start_stop_costs

    logger.info("TODO: operating cost for el storage - needs improvement")

    def compute_operatingCosts(self, model):
        """term in objective function to represent fuel costs or similar
        as average per sec ($/s)

        opCost = energy costs (NOK/MJ, or NOK/Sm3)
        Note: el costs per MJ not per MWh
        """
        sumCost = 0
        timesteps = model.setHorizon
        for dev in model.setDevice:
            dev_obj = self.all_devices[dev]
            thisCost = dev_obj.compute_operatingCosts(model, timesteps)
            sumCost += thisCost
        return sumCost

    def compute_costForDepletedStorage(self, model):
        """term in objective function to discourage depleting battery,
        making sure it is used only when required"""
        storCost = 0
        timesteps = model.setHorizon
        for dev in model.setDevice:
            dev_obj = self.all_devices[dev]
            thisCost = dev_obj.compute_costForDepletedStorage(model, timesteps)
            storCost += thisCost
        return storCost

    def compute_exportRevenue(self, model, carriers=None, timesteps=None):
        """revenue from exported oil and gas - average per sec ($/s)"""
        return self.compute_export(
            model, value="revenue", carriers=carriers, timesteps=timesteps
        )

    def compute_oilgas_export(self, model, timesteps=None):
        """Export volume (Sm3oe/s)"""
        return self.compute_export(
            model, value="volume", carriers=["oil", "gas"], timesteps=timesteps
        )

    def compute_export(self, model, value="revenue", carriers=None, timesteps=None):
        """Compute average export (volume or revenue)

        Parameters
        ----------
        model : oogeso model
        value : string ("revenue", "volume")
            which value to compute, revenue (€/s) or volume (Sm3oe/s)

        Computes the energy/mass flow into (sink) devices with a price.CARRIER
        parameter defined (CARRIER can be any of 'oil', 'gas', 'el')
        """
        if carriers is None:
            carriers = model.setCarrier
        if timesteps is None:
            timesteps = model.setHorizon

        sumValue = 0
        for dev in model.setDevice:
            dev_obj = self.all_devices[dev]
            thisValue = dev_obj.compute_export(model, value, carriers, timesteps)
            sumValue += thisValue
        # average per second (timedelta is not required)
        sumValue = sumValue / len(timesteps)
        return sumValue

    def getDevicesInout(self, carrier_in=None, carrier_out=None):
        """devices that have the specified connections in and out"""
        devs = []
        for d, dev_obj in self.all_devices.items():
            ok_in = (carrier_in is None) or (carrier_in in dev_obj.carrier_in)
            ok_out = (carrier_out is None) or (carrier_out in dev_obj.carrier_out)
            if ok_in and ok_out:
                devs.append(d)
        return devs

    def write(self, filename: str):
        """Export optimisation problem to MPS or LP file"""
        self.pyomo_instance.write(
            filename=filename, io_options={"symbolic_solver_labels": True}
        )

    def extract_all_variable_values(
        self, timelimit: int = None, timeshift: int = 0
    ) -> Dict[str, pd.Series]:
        """Extract variable values and return as a dictionary of pandas milti-index series"""
        ins = self.pyomo_instance
        all_vars = [
            ins.varEdgeFlow,
            ins.varEdgeLoss,
            ins.varDeviceIsPrep,
            ins.varDeviceIsOn,
            ins.varDeviceStarting,
            ins.varDeviceStopping,
            ins.varDeviceStorageEnergy,
            ins.varDeviceStoragePmax,
            ins.varPressure,
            ins.varElVoltageAngle,
            ins.varDeviceFlow,
            ins.varTerminalFlow,
            ins.varDevicePenalty,
        ]
        # all_vars = self.pyomo_instance.component_objects(pyo.Var, active=True)
        all_values = {}
        for myvar in all_vars:
            # extract the variable index names in the right order
            indices = [index_set.doc for index_set in myvar._implicit_subsets]
            var_values = myvar.get_values()
            if not var_values:
                # print("var_values=", var_values)
                # empty dictionary, so no variables to store
                all_values[myvar.name] = None
                continue
            # This creates a pandas.Series:
            df = pd.DataFrame.from_dict(var_values, orient="index", columns=["value"])[
                "value"
            ]
            df.index = pd.MultiIndex.from_tuples(df.index, names=indices)
            # check that all vales are non-negative for deviceflow and give warning otherwise
            if (myvar == ins.varDeviceFlow) and (
                df < -self.ZERO_WARNING_THRESHOLD
            ).any():
                ind = df[df < -self.ZERO_WARNING_THRESHOLD].index[0]
                logger.warning(
                    "Negative number in varDeviceFlow - set to zero ({}:{})".format(
                        ind, df[ind]
                    )
                )
                df = df.clip(lower=0)

            # ignore NA values
            df = df.dropna()
            # df = df.unstack("time").T

            if timelimit is not None:
                mask = df.index.get_level_values("time") < timelimit
                df = df[mask]
            if timeshift > 0:
                level_time = df.index.names.index("time")
                new_values = df.index.levels[level_time] + timeshift
                new_index = df.index.set_levels(new_values, level="time")
                df.index = new_index

            all_values[myvar.name] = df
        return all_values
