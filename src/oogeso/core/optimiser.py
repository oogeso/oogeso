import logging
from typing import Dict, List, Optional, Union

import pandas as pd
import pyomo.environ as pyo
import pyomo.opt as pyopt

from oogeso import dto
from oogeso.core import networks
from oogeso.core.devices.base import Device
from oogeso.core.devices.storage import StorageDevice
from oogeso.core.networks import ElNetwork, Network
from oogeso.core.networks.edge import Edge
from oogeso.core.networks.network_node import NetworkNode
from oogeso.dto.mapper import get_device_from_model_name, get_network_from_carrier_name

logger = logging.getLogger(__name__)


class OptimisationModel(pyo.ConcreteModel):
    """Class for MILP optimisation model"""

    ZERO_WARNING_THRESHOLD = 1e-6

    def __init__(self, data: dto.EnergySystemData):
        """Create optimisation problem formulation with supplied data"""

        super().__init__()

        # dictionaries {key:object} for all devices, nodes and edges
        self.all_devices: Dict[str, Device] = {}
        self.all_nodes: Dict[str, NetworkNode] = {}
        self.all_edges: Dict[str, Edge] = {}
        # self.all_carriers = {}
        self.all_networks: Dict[str, Union[Network, ElNetwork]] = {}

        # Model parameters
        self.optimisation_parameters = data.parameters

        # List of constraints that need to be reconstructed for each optimisation:
        self.constraints_to_reconstruct = []
        # List of devices with storage
        self.devices_with_storage = []
        profiles_in_use = list(set(d.profile for d in data.devices if d.profile is not None))
        logger.debug("profiles in use: %s", profiles_in_use)

        self._create_network_objects_from_data(data)
        self._set_node_pressure_from_edge_data()
        self._create_pyomo_model(profiles_in_use)

    def solve(self, solver="cbc", solver_executable=None, solver_options=None, write_yaml=False, time_limit=None):
        """Solve problem for planning horizon at a single timestep"""

        opt = pyo.SolverFactory(solver, executable=solver_executable)
        if solver_options is not None:
            for k in solver_options:
                opt.options[k] = solver_options[k]
        if time_limit is not None:
            if solver == "gurobi":
                opt.options["TimeLimit"] = time_limit
            elif solver == "cbc":
                opt.options["sec"] = time_limit
            elif solver == "cplex":
                opt.options["timelimit"] = time_limit
            elif solver == "glpk":
                opt.options["tmlim"] = time_limit
        logger.debug("Solving...")
        sol = opt.solve(self)

        if write_yaml:
            sol.write_yaml()

        if (sol.solver.status == pyopt.SolverStatus.ok) and (
            sol.solver.termination_condition == pyopt.TerminationCondition.optimal
        ):
            logger.debug("Solved OK")
        elif sol.solver.termination_condition == pyopt.TerminationCondition.infeasible:
            raise Exception("Infeasible solution")
        else:
            # Something else is wrong
            logger.warning("Solver Status:{}".format(sol.solver.status))
        return sol

    def _set_node_pressure_from_edge_data(self):
        # Set node terminal nominal pressure based on edge from/to pressure values
        for edge in self.all_edges.values():
            edg = edge.edge_data
            carrier = edg.carrier
            # Setting nominal pressure levels at node terminals. Raise exception
            # if inconsistencies are found
            if hasattr(edg, "pressure_from"):
                if edg.pressure_from is not None:
                    n_from: NetworkNode = self.all_nodes[edg.node_from]
                    p_from = edg.pressure_from
                    n_from.set_pressure_nominal(carrier, "out", p_from)
            if hasattr(edg, "pressure_to"):
                if edg.pressure_to is not None:
                    n_to: NetworkNode = self.all_nodes[edg.node_to]
                    p_to = edg.pressure_to
                    n_to.set_pressure_nominal(carrier, "in", p_to)
            # Setting max pressure deviation values at node terminals. Raise exception
            # if inconsistencies are found
            if hasattr(edg, "pressure_from_maxdeviation"):
                if edg.pressure_from_maxdeviation is not None:
                    n_from: NetworkNode = self.all_nodes[edg.node_from]
                    p_maxdev_from = edg.pressure_from_maxdeviation
                    n_from.set_pressure_maxdeviation(carrier, "out", p_maxdev_from)
            if hasattr(edg, "pressure_to_maxdeviation"):
                if edg.pressure_to_maxdeviation is not None:
                    n_to: NetworkNode = self.all_nodes[edg.node_to]
                    p_maxdev_to = edg.pressure_to_maxdeviation
                    n_to.set_pressure_maxdeviation(carrier, "in", p_maxdev_to)

    def _create_network_objects_from_data(self, data: dto.EnergySystemData):
        """Create energy system objects, and populate local dictionaries

        self.all_devices, self.all_nodes, self.all_networks"""

        energy_system_data = data

        # Create energy system network elements (devices, nodes, edges)
        for dev_data_obj in energy_system_data.devices:
            dev_id = dev_data_obj.id
            if not dev_data_obj.include:
                # skip this edge and move to next
                logger.debug("Excluding device {}".format(dev_id))
                continue
            device_model = dev_data_obj.model
            # The class corresponding to the device type should always have a
            # name identical to the type (but capitalized):
            logger.debug("Device model={}".format(device_model))
            device = get_device_from_model_name(model_name=device_model)
            carrier_data_dict = {carr_obj.id: carr_obj for carr_obj in data.carriers}
            new_device = device(dev_data_obj, carrier_data_dict)
            if isinstance(new_device, StorageDevice):
                # Add this device to global list of storage devices:
                self.devices_with_storage.append(new_device)

            new_device.set_flow_upper_bound(data.profiles)
            self.all_devices[dev_id] = new_device

        for node_data_obj in energy_system_data.nodes:
            new_node = networks.NetworkNode(node_data_obj)
            node_id = new_node.id
            self.all_nodes[node_id] = new_node

        edges_per_type = {}
        for edge_data_obj in data.edges:
            if edge_data_obj.include is False:
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
            network_class = get_network_from_carrier_name(carrier_model)
            if carrier_model not in edges_per_type:
                edges_per_type[carrier_model] = {}
            new_network = network_class(
                carrier_data=carrier_data_obj,
                edges=edges_per_type[carrier_model],
            )
            self.all_networks[carrier_model] = new_network

        # Make network connections between nodes, edges and devices
        for dev_id, dev in self.all_devices.items():
            logger.debug("Node-device: %s,%s", dev_id, dev)
            node_id_where_connected = dev.dev_data.node_id
            node = self.all_nodes[node_id_where_connected]
            node.add_device(dev_id, dev)
            dev.add_node(node)
        for edge_id, edge in self.all_edges.items():
            node_from = self.all_nodes[edge.edge_data.node_from]
            node_from.add_edge(edge, "from")
            node_to = self.all_nodes[edge.edge_data.node_to]
            node_to.add_edge(edge, "to")
            edge.add_nodes(node_from, node_to)

    def _create_pyomo_model(self, profiles_in_use):
        """Create pyomo MILP model

        Parameters:
        -----------
            planning_horizon: dict
            input data as read from input file
        """

        self._specify_sets_and_parameters(profiles_in_use)
        self._specify_variables()
        self._specify_objective()
        self._specify_constraints()

        # Specify initial values from input data
        for dev in self.all_devices.values():
            dev.set_init_values(pyomo_model=self)

        # Keep track of duals:
        # WARNING:
        # From Gurobi: "Shadow prices are not well-defined in mixed-integer
        # problems, so we don't provide dual values for an integer program."
        self.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    def _specify_sets_and_parameters(self, profiles_in_use):
        """specify pyomo model sets and parameters"""
        energycarriers = self.all_networks.keys()
        self.setCarrier = pyo.Set(initialize=energycarriers, doc="carrier")
        self.setTerminal = pyo.Set(initialize=["in", "out"], doc="terminal")
        self.setNode = pyo.Set(initialize=self.all_nodes.keys(), doc="node")
        self.setDevice = pyo.Set(initialize=self.all_devices.keys(), doc="device")
        self.setEdge = pyo.Set(initialize=self.all_edges.keys(), doc="edge")
        timerange = range(self.optimisation_parameters.planning_horizon)
        self.setHorizon = pyo.Set(ordered=True, initialize=timerange, doc="time")
        self.setProfile = pyo.Set(initialize=profiles_in_use, doc="profile")

        # specify mutable parameters:
        # (will be modified between successive optimisations)
        self.paramProfiles = pyo.Param(
            self.setProfile,
            self.setHorizon,
            within=pyo.Reals,
            mutable=True,
            initialize=1,
        )
        self.paramDeviceIsOnInitially = pyo.Param(self.setDevice, mutable=True, within=pyo.Binary, initialize=1)
        self.paramDevicePrepTimestepsInitially = pyo.Param(
            self.setDevice, mutable=True, within=pyo.Integers, initialize=0
        )
        self.paramDeviceOnlineTimestepsInitially = pyo.Param(
            self.setDevice, mutable=True, within=pyo.Integers, initialize=1000
        )
        self.paramDeviceOfflineTimestepsInitially = pyo.Param(
            self.setDevice, mutable=True, within=pyo.Integers, initialize=1000
        )
        # needed for ramp rate limits:
        self.paramDevicePowerInitially = pyo.Param(self.setDevice, mutable=True, within=pyo.Reals, initialize=0)
        # needed for energy storage:
        self.paramDeviceEnergyInitially = pyo.Param(self.setDevice, mutable=True, within=pyo.Reals, initialize=0)
        # target energy level at end of horizon (useful for long-term storage)
        self.paramDeviceEnergyTarget = pyo.Param(self.setDevice, mutable=True, within=pyo.Reals, initialize=0)

        # Specify handy immutable parameters (for easy access when building/updating model)
        self.paramTimestepDeltaMinutes = pyo.Param(
            within=pyo.Reals, default=self.optimisation_parameters.time_delta_minutes
        )

    def _specify_variables(self):
        """specify pyomo model variables"""
        self.varEdgeFlow = pyo.Var(self.setEdge, self.setHorizon, within=pyo.Reals)
        self.varEdgeFlow12 = pyo.Var(self.setEdge, self.setHorizon, within=pyo.NonNegativeReals)
        self.varEdgeFlow21 = pyo.Var(self.setEdge, self.setHorizon, within=pyo.NonNegativeReals)
        self.varEdgeLoss = pyo.Var(self.setEdge, self.setHorizon, within=pyo.NonNegativeReals, initialize=0)
        self.varEdgeLoss12 = pyo.Var(self.setEdge, self.setHorizon, within=pyo.NonNegativeReals, initialize=0)
        self.varEdgeLoss21 = pyo.Var(self.setEdge, self.setHorizon, within=pyo.NonNegativeReals, initialize=0)
        self.varDeviceIsPrep = pyo.Var(self.setDevice, self.setHorizon, within=pyo.Binary, initialize=0)
        self.varDeviceIsOn = pyo.Var(self.setDevice, self.setHorizon, within=pyo.Binary, initialize=1)
        self.varDeviceStarting = pyo.Var(self.setDevice, self.setHorizon, within=pyo.Binary, initialize=None)
        self.varDeviceStopping = pyo.Var(self.setDevice, self.setHorizon, within=pyo.Binary, initialize=None)
        self.varDeviceStorageEnergy = pyo.Var(self.setDevice, self.setHorizon, within=pyo.Reals)
        # available reserve power from storage (linked to power rating and storage level):
        self.varDeviceStoragePmax = pyo.Var(self.setDevice, self.setHorizon, within=pyo.NonNegativeReals, initialize=0)
        # binary variable related to available powr from storage:
        self.varStorY = pyo.Var(self.setDevice, self.setHorizon, within=pyo.Binary)
        # absolute value variable for storage with target level:
        self.varDeviceStorageDeviationFromTarget = pyo.Var(self.setDevice, within=pyo.NonNegativeReals, initialize=0)
        self.varPressure = pyo.Var(
            self.setNode,
            self.setCarrier,
            self.setTerminal,
            self.setHorizon,
            within=pyo.NonNegativeReals,
            initialize=0,
        )
        self.varElVoltageAngle = pyo.Var(self.setNode, self.setHorizon, within=pyo.Reals)
        self.varDeviceFlow = pyo.Var(
            self.setDevice,
            self.setCarrier,
            self.setTerminal,
            self.setHorizon,
            within=pyo.NonNegativeReals,
            initialize=0,
        )
        self.varTerminalFlow = pyo.Var(self.setNode, self.setCarrier, self.setHorizon, within=pyo.Reals)
        # this penalty variable should only require (device,time), but the
        # piecewise constraint requires the domain to be the same as for varDeviceFlow
        self.varDevicePenalty = pyo.Var(
            self.setDevice,
            self.setCarrier,
            self.setTerminal,
            self.setHorizon,
            within=pyo.Reals,
        )

    def _specify_objective(self):
        """specify pyomo model objective"""
        obj = self.optimisation_parameters.objective
        if obj == "penalty":
            rule = self._rule_objective_penalty
        elif obj == "co2":
            rule = self._rule_objective_co2
        elif obj == "costs":
            rule = self._rule_objective_costs
        elif obj == "exportRevenue":
            rule = self._rule_objective_export_revenue
        elif obj == "co2intensity":
            rule = self._rule_objective_co2intensity
        else:
            raise Exception("Objective '{}' has not been implemented".format(obj))
        logger.debug("Using objective function: {}".format(obj))
        self.objObjective = pyo.Objective(rule=rule, sense=pyo.minimize)

    def _specify_constraints(self):

        # 1. Constraints associated with each device:
        for dev in self.all_devices.values():
            list_to_reconstruct = dev.define_constraints(self)

            # Because of logic that needs to be re-evalued, these constraints need
            # to be reconstructed each optimisation:
            for constr in list_to_reconstruct:
                self.constraints_to_reconstruct.append(constr)

        # 2. Constraints associated with each node:
        for node in self.all_nodes.values():
            node.define_constraints(self)

        # 3. Constraints associated with each network type (and its edges):
        for netw in self.all_networks.values():
            netw.define_constraints(self)

        # 4. Global constraints:
        # 4.1 max limit emission rate:
        params_generic = self.optimisation_parameters
        if (params_generic.emission_rate_max is not None) and (params_generic.emission_rate_max >= 0):
            self.constr_O_emissionrate = pyo.Constraint(self.setHorizon, rule=self._rule_emission_rate_limit)
        else:
            logger.debug("No emission_rate_max limit specified")
        # 4.2 max limit emission intensity
        if (params_generic.emission_intensity_max is not None) and (params_generic.emission_intensity_max >= 0):
            self.constr_O_emissionintensity = pyo.Constraint(self.setHorizon, rule=self._rule_emission_intensity_limit)
        else:
            logger.debug("No emission_intensity_max limit specified")
        # 4.3 electrical reserve margin:
        el_parameters: dto.CarrierElData = self.all_networks["el"].carrier_data
        el_reserve_margin = el_parameters.el_reserve_margin
        el_backup_margin = el_parameters.el_backup_margin
        if (el_reserve_margin is not None) and (el_reserve_margin >= 0):
            self.constr_O_elReserveMargin = pyo.Constraint(self.setHorizon, rule=self._rule_el_reserve_margin)
        else:
            logger.debug("No el_reserve_margin limit specified")
        # 4.4 electrical backup power margin
        if (el_backup_margin is not None) and (el_backup_margin >= 0):
            self.constr_O_elBackupMargin = pyo.Constraint(
                self.setDevice,
                self.setHorizon,
                rule=self._rule_el_backup_margin,
            )
        else:
            logger.debug("No el_backup_margin limit specified")

    def update_optimisation_model(self, timestep, profiles, first=False):
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
        for prof in self.setProfile:
            for t in range(timesteps_use_nowcast):  # 0,1,2,3
                profile_str = "nowcast"
                if prof not in profiles["nowcast"]:
                    # no nowcast, use forecast instead
                    profile_str = "forecast"
                self.paramProfiles[prof, t] = profiles[profile_str].loc[timestep + t, prof]
            for t in range(timesteps_use_nowcast, horizon):
                self.paramProfiles[prof, t] = profiles["forecast"].loc[timestep + t, prof]

        def _count_consequtive_steps(_timestep, _dev, the_value, the_variable, the_parameter):
            """Count number of steps the_variable has had value=the_value"""
            count = 0
            for tt in range(_timestep, -1, -1):  # tt = [_timestep,..,2,1,0]
                if pyo.value(the_variable[_dev, tt]) == the_value:
                    count = count + 1
                else:
                    break
            if count == _timestep:
                # all timesteps back to beginning, so include count from previous optimisation
                count = count + the_parameter[_dev]
            return count

        # Update startup/shutdown info
        # pick the last value from previous optimistion prior to the present time
        if not first:
            t_prev = opt_timesteps - 1
            for dev, dev_obj in self.all_devices.items():
                # On/off status: (round because solver doesn't alwasy return an integer)
                self.paramDeviceIsOnInitially[dev] = round(pyo.value(self.varDeviceIsOn[dev, t_prev]))
                self.paramDevicePrepTimestepsInitially[dev] = _count_consequtive_steps(
                    t_prev, dev, 1, self.varDeviceIsPrep, self.paramDevicePrepTimestepsInitially
                )
                self.paramDeviceOnlineTimestepsInitially[dev] = _count_consequtive_steps(
                    t_prev, dev, 1, self.varDeviceIsOn, self.paramDeviceOnlineTimestepsInitially
                )
                self.paramDeviceOfflineTimestepsInitially[dev] = _count_consequtive_steps(
                    t_prev, dev, 0, self.varDeviceIsOn, self.paramDeviceOfflineTimestepsInitially
                )
                # Initial power output (relevant for ramp rate constraint):
                if dev_obj.dev_data.max_ramp_up is not None:
                    self.paramDevicePowerInitially[dev] = dev_obj.get_flow_var(self, t_prev)
                # Energy storage:
                if dev_obj in self.devices_with_storage:
                    self.paramDeviceEnergyInitially[dev] = self.varDeviceStorageEnergy[dev, t_prev]
                    # Update target profile if present:
                    if hasattr(dev_obj.dev_data, "target_profile") and hasattr(dev_obj.dev_data, "E_max"):
                        if dev_obj.dev_data.target_profile is not None:
                            prof = dev_obj.dev_data.target_profile
                            max_E = dev_obj.dev_data.E_max
                            self.paramDeviceEnergyTarget[dev] = (
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

    def _rule_objective_penalty(self, model: pyo.Model) -> Union[pyo.Expression, pyo.Constraint.Skip]:
        """'penalty' as specified through penalty functions"""
        sum_penalty = 0
        timesteps: List[int] = list(self.setHorizon)
        for d in self.setDevice:
            dev = self.all_devices[d]
            this_penalty = dev.compute_penalty(model, timesteps)
            sum_penalty = sum_penalty + this_penalty
        return sum_penalty

    def _rule_objective_co2(self, model: pyo.Model) -> float:
        """CO2 emissions per sec"""
        return self.compute_CO2(model)

    def _rule_objective_co2intensity(self, model: pyo.Model) -> Optional[float]:
        """CO2 emission intensity (CO2 per exported oil/gas)
        DOES NOT WORK - NONLINEAR (ratio)"""
        return self.compute_CO2_intensity(model)

    def _rule_objective_costs(self, model: pyo.Model) -> float:
        """costs (co2 price, operating costs, startstop, storage depletaion)
        per second (assuming fixed oil/gas production)"""
        startup_costs = self.compute_startup_penalty(model)  # kr/s
        storage_depletion_costs = self.compute_cost_for_depleted_storage(model)
        op_costs = self.compute_operating_costs(model)  # kr/s
        co2 = self.compute_CO2(model)  # kgCO2/s
        co2_tax = self.optimisation_parameters.co2_tax  # kr/kgCO2
        co2_cost = co2 * co2_tax  # kr/s

        return co2_cost + startup_costs + storage_depletion_costs + op_costs

    def _rule_objective_export_revenue(self, model: pyo.Model) -> float:
        """revenue from exported oil and gas minus costs (co2 price and
        operating costs) per second"""
        sum_revenue = self.compute_export_revenue(model)  # kr/s
        startup_costs = self.compute_startup_penalty(model)  # kr/s
        co2 = self.compute_CO2(model)  # kgCO2/s
        co2_tax = self.optimisation_parameters.co2_tax  # kr/kgCO2
        co2_cost = co2 * co2_tax  # kr/s
        storage_depletion_costs = self.compute_cost_for_depleted_storage(model)
        op_costs = self.compute_operating_costs(model)  # kr/s

        return -sum_revenue + co2_cost + startup_costs + storage_depletion_costs + op_costs

    def _rule_emission_rate_limit(self, model: pyo.Model, t) -> Union[pyo.Expression, pyo.Constraint.Skip]:
        """Upper limit on CO2 emission rate"""
        params_generic = self.optimisation_parameters
        lhs = self.compute_CO2(model, timesteps=[t])
        rhs = params_generic.emission_rate_max
        return pyo.Expression(lhs <= rhs)

    def _rule_emission_intensity_limit(self, model: pyo.Model, t) -> Union[pyo.Expression, pyo.Constraint.Skip]:
        """Upper limit on CO2 emission intensity"""
        params_generic = self.optimisation_parameters
        emission_intensity_max = params_generic.emission_intensity_max
        lhs = self.compute_CO2(model, timesteps=[t])
        rhs = emission_intensity_max * self.compute_oilgas_export(model, timesteps=[t])
        return pyo.Expression(lhs <= rhs)

    def _rule_el_reserve_margin(self, pyomo_model: pyo.Model, t: int) -> Union[pyo.Expression, pyo.Constraint.Skip]:
        """Reserve margin constraint (electrical supply)
        Not used capacity by power suppliers/storage/load flexibility
        must be larger than some specified margin
        (to cope with unforeseen variations)
        """
        # exclude constraint for first timesteps since the point of the
        # dispatch margin is exactly to cope with forecast errors
        if t < self.optimisation_parameters.forecast_timesteps:
            return pyo.Constraint.Skip  # noqa

        network_el = self.all_networks["el"]

        if hasattr(network_el.carrier_data, "el_reserve_margin"):
            margin = network_el.carrier_data.el_reserve_margin
        else:
            raise ValueError("network_el.carrier_data does not have the attribute el_reserve_margin")
        if hasattr(network_el, "compute_el_reserve") and callable(getattr(network_el, "compute_el_reserve")):
            capacity_unused = network_el.compute_el_reserve(pyomo_model=pyomo_model, t=t, all_devices=self.all_devices)
        else:
            raise ValueError("network_el does not have the function compute_el_reserve")

        return capacity_unused >= margin

    def _rule_el_backup_margin(self, model: pyo.Model, dev, t) -> Union[pyo.Expression, pyo.Constraint.Skip]:
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
            return pyo.Constraint.Skip  # noqa
        res_otherdevs = network_el.compute_el_reserve(model, t, self.all_devices, exclude_device=dev)
        expr = res_otherdevs - self.varDeviceFlow[dev, "el", "out", t] >= -margin
        return expr

    def compute_CO2(self, model: pyo.Model, devices=None, timesteps=None):
        """compute CO2 emissions - average per sec (kgCO2/s)"""
        if devices is None:
            devices = self.setDevice
        if timesteps is None:
            timesteps = self.setHorizon
        sum_CO2 = 0
        for d in devices:
            dev = self.all_devices[d]
            sum_CO2 += dev.compute_CO2(model, timesteps)

        # Average per s
        return sum_CO2 / len(timesteps)

    def compute_CO2_intensity(self, model: pyo.Model, timesteps=None):
        """CO2 emission per exported oil/gas (kgCO2/Sm3oe)"""
        if timesteps is None:
            timesteps = self.setHorizon

        co2_kg_per_time = self.compute_CO2(model, devices=None, timesteps=timesteps)
        flow_oil_equivalents_m3_per_time = self.compute_oilgas_export(model, timesteps)
        if pyo.value(flow_oil_equivalents_m3_per_time) != 0:
            return co2_kg_per_time / flow_oil_equivalents_m3_per_time
        elif pyo.value(flow_oil_equivalents_m3_per_time) == 0:
            return None
        else:
            return co2_kg_per_time

    def compute_startup_penalty(self, model: pyo.Model, devices=None, timesteps=None):
        """startup costs (average per sec)"""
        if timesteps is None:
            timesteps = self.setHorizon
        if devices is None:
            devices = self.setDevice
        start_stop_costs = 0
        for d in devices:
            dev_obj = self.all_devices[d]
            start_stop_costs += dev_obj.compute_startup_penalty(model, timesteps)
        # get average per sec:
        delta_T = self.optimisation_parameters.time_delta_minutes * 60
        sum_time = len(timesteps) * delta_T
        start_stop_costs = start_stop_costs / sum_time
        return start_stop_costs

    def compute_operating_costs(self, model: pyo.Model, timesteps=None):
        """term in objective function to represent fuel costs or similar
        as average per sec ($/s)

        opCost = energy costs (NOK/MJ, or NOK/Sm3)
        Note: el costs per MJ not per MWh
        """
        sum_cost = 0
        if timesteps is None:
            timesteps = self.setHorizon
        for dev in self.setDevice:
            dev_obj = self.all_devices[dev]
            sum_cost += dev_obj.compute_operating_costs(model, timesteps)
        return sum_cost

    def compute_cost_for_depleted_storage(self, model: pyo.Model, timesteps=None):
        """term in objective function to discourage depleting battery,
        making sure it is used only when required"""
        store_cost = 0
        if timesteps is None:
            timesteps = self.setHorizon
        for dev in self.setDevice:
            dev_obj = self.all_devices[dev]
            store_cost += dev_obj.compute_cost_for_depleted_storage(model, timesteps)
        return store_cost

    def compute_export_revenue(
        self, model: pyo.Model, carriers=None, timesteps: Optional[Union[List[int], pyo.Set]] = None
    ):
        """revenue from exported oil and gas - average per sec ($/s)"""
        return self.compute_export(model, value="revenue", carriers=carriers, timesteps=timesteps)

    def compute_oilgas_export(self, model: pyo.Model, timesteps: Optional[pyo.Set] = None):
        """Export volume (Sm3oe/s)"""
        return self.compute_export(model, value="volume", carriers=["oil", "gas"], timesteps=timesteps)

    def compute_export(
        self,
        model: pyo.Model,
        value="revenue",
        carriers: Optional[List[str]] = None,
        timesteps: Optional[Union[List[int], pyo.Set]] = None,
    ):
        """Compute average export (volume or revenue)

        Parameters
        ----------
        model : oogeso model
        value : string ("revenue", "volume")
            which value to compute, revenue (€/s) or volume (Sm3oe/s)
        carriers : Optional list of carrier names.
        timesteps: Optional list of timesteps

        Computes the energy/mass flow into (sink) devices with a price.CARRIER
        parameter defined (CARRIER can be any of 'oil', 'gas', 'el')
        """
        if carriers is None:
            carriers = self.setCarrier
        if timesteps is None:
            timesteps = self.setHorizon

        sum_value = 0
        for dev in self.setDevice:
            dev_obj = self.all_devices[dev]
            sum_value += dev_obj.compute_export(model, value, carriers, timesteps)
        # average per second (timedelta is not required)
        sum_value = sum_value / len(timesteps)
        return sum_value

    def get_devices_in_out(self, carrier_in=None, carrier_out=None):
        """devices that have the specified connections in and out"""
        devs = []
        for d, dev_obj in self.all_devices.items():
            ok_in = (carrier_in is None) or (carrier_in in dev_obj.carrier_in)
            ok_out = (carrier_out is None) or (carrier_out in dev_obj.carrier_out)
            if ok_in and ok_out:
                devs.append(d)
        return devs

    def extract_all_variable_values(self, timelimit: int = None, timeshift: int = 0) -> Dict[str, pd.Series]:
        """Extract variable values and return as a dictionary of pandas milti-index series"""
        all_vars = [
            self.varEdgeFlow,
            self.varEdgeLoss,
            self.varDeviceIsPrep,
            self.varDeviceIsOn,
            self.varDeviceStarting,
            self.varDeviceStopping,
            self.varDeviceStorageEnergy,
            self.varDeviceStoragePmax,
            self.varPressure,
            self.varElVoltageAngle,
            self.varDeviceFlow,
            self.varTerminalFlow,
            self.varDevicePenalty,
        ]
        all_values = {}
        for myvar in all_vars:
            # extract the variable index names in the right order
            indices = [index_set.doc for index_set in myvar._implicit_subsets]
            var_values = myvar.get_values()
            if not var_values:
                # empty dictionary, so no variables to store
                all_values[myvar.name] = None
                continue
            # This creates a pandas.Series:
            df = pd.DataFrame.from_dict(var_values, orient="index", columns=["value"])["value"]
            df.index = pd.MultiIndex.from_tuples(df.index, names=indices)
            # check that all vales are non-negative for deviceflow and give warning otherwise
            if (myvar == self.varDeviceFlow) and (df < -self.ZERO_WARNING_THRESHOLD).any():
                ind = df[df < -self.ZERO_WARNING_THRESHOLD].index[0]
                logger.warning("Negative number in varDeviceFlow - set to zero ({}:{})".format(ind, df[ind]))
                df = df.clip(lower=0)

            # ignore NA values
            df = df.dropna()
            if df.empty:
                all_values[myvar.name] = None
                continue

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
