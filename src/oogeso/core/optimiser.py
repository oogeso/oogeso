import logging
import pyomo.environ as pyo
import pyomo.opt as pyopt
import pandas as pd
from . import devices
from . import networks
from .networks import electricalsystem as el_calc


class Optimiser:
    """Class for MILP optimisation model"""

    def __init__(self, data):
        """Create optimisation problem formulation with supplied data"""

        # dictionaries {key:object} for all devices, nodes and edges
        self.all_devices = {}
        self.all_nodes = {}
        self.all_edges = {}
        self.all_carriers = {}
        self.optimisation_parameters = {}
        self.pyomo_instance = None
        # List of constraints that need to be reconstructed for each optimisation:
        self.constraints_to_reconstruct = []
        # List of devices with storage
        self.devices_with_storage = []

        # raise NotImplementedError()
        self.pyomo_instance = self.createOptimisationModel(data)
        self._setNodePressureFromEdgeData()
        self._specifyConstraints()

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
        logging.debug("Solving...")
        sol = opt.solve(self.pyomo_instance)

        if write_yaml:
            sol.write_yaml()

        if (sol.solver.status == pyopt.SolverStatus.ok) and (
            sol.solver.termination_condition == pyopt.TerminationCondition.optimal
        ):
            logging.debug("Solved OK")
            pass
        elif sol.solver.termination_condition == pyopt.TerminationCondition.infeasible:
            raise Exception("Infeasible solution")
        else:
            # Something else is wrong
            logging.info("Solver Status:{}".format(sol.solver.status))
        return sol

    def _setNodePressureFromEdgeData(self):
        # Set node terminal nominal pressure based on edge from/to pressure values

        for i, edge in self.all_edges.items():
            edg = edge.params
            if "pressure.from" not in edg:
                continue
            # if np.isnan(edg['pressure.from']):
            #    continue
            n_from = edg["nodeFrom"]
            n_to = edg["nodeTo"]
            typ = edg["type"]
            p_from = edg["pressure.from"]
            p_to = edg["pressure.to"]
            c_out = "pressure.{}.out".format(typ)
            c_in = "pressure.{}.in".format(typ)
            # Check that pressure values are consistent:
            is_consistent = True
            existing_p_from = None
            existing_p_to = None
            try:
                if c_out in self.all_nodes[n_from].params:
                    existing_p_from = self.all_nodes[n_from].params[c_out]
                logging.debug(
                    "{} p_from = {} (existing) / {} (new)".format(
                        i, existing_p_from, p_from
                    )
                )
                if (existing_p_from is not None) and (existing_p_from != p_from):
                    msg = (
                        "Input data edge pressure from values are"
                        " inconsistent (edge={}, {}!={})"
                    ).format(i, existing_p_from, p_from)
                    is_consistent = False
            except:
                pass
            try:
                if c_in in self.all_nodes[n_to].params:
                    existing_p_to = self.all_nodes[n_to].params[c_in]
                logging.debug(
                    "{} p_to = {} (existing) / {} (new)".format(i, existing_p_to, p_to)
                )
                if (existing_p_to is not None) and (existing_p_to != p_to):
                    msg = (
                        "Input data edge pressure to values are"
                        " inconsistent (edge={})"
                    ).format(i)
                    is_consistent = False
            except:
                pass
            if not is_consistent:
                # logging.warning(self.all_nodes)
                raise Exception(msg)

            self.all_nodes[n_from].params[c_out] = p_from
            self.all_nodes[n_to].params[c_in] = p_to

    def createOptimisationModel(self, data):
        """Create pyomo MILP model

        Parameters:
        -----------
            data: dict
            input data as read from input file
        """

        model = pyo.ConcreteModel()

        # Create energy system network elements (devices, nodes, edges)
        for dev_id, dev_data in data["paramDevice"].items():
            if dev_data["include"] != 0:
                device_model = dev_data["model"]
                # The class corresponding to the device type should always have a
                # name identical to the type (but capitalized):
                logging.debug("Device model={}".format(device_model))
                Devclass = getattr(devices, device_model.capitalize())
                newDevice = Devclass(model, dev_id, dev_data, self)
                self.all_devices[dev_id] = newDevice
        #            else:
        #                logging.debug("Excluding device {}".format(dev_id))

        for node_id, node_data in data["paramNode"].items():
            newNode = networks.NetworkNode(model, node_id, node_data, self)
            self.all_nodes[node_id] = newNode

        for edge_id, edge_data in data["paramEdge"].items():
            newEdge = networks.NetworkEdge(model, edge_id, edge_data, self)
            self.all_edges[edge_id] = newEdge

        for carrier_id, carrier_data in data["paramCarriers"].items():
            newCarrier = networks.EnergyCarrier(model, carrier_id, carrier_data)
            self.all_carriers[carrier_id] = newCarrier

        self.optimisation_parameters = data["paramParameters"]

        logging.debug("TODO: improve code for electric powerflow coefficients")
        nodelist = self.all_nodes.keys()
        edgelist = {
            edge_id: edge.params
            for edge_id, edge in self.all_edges.items()
            if edge.params["type"] == "el"
        }
        coeffB, coeffDA = el_calc.computePowerFlowMatrices(nodelist, edgelist, baseZ=1)
        self.elFlowCoeffB = coeffB
        self.elFlowCoeffDA = coeffDA

        # logging.debug(self.all_nodes.keys())
        # logging.debug(self.all_edges.keys())
        # logging.debug(self.all_devices.keys())

        timerange = range(data["paramParameters"]["planning_horizon"])
        profiles_in_use = list(
            set(d["profile"] for k, d in data["paramDevice"].items() if "profile" in d)
        )
        logging.info("profiles in use: {}".format(profiles_in_use))

        # Make network connections between nodes, edges and devices
        for dev_id, dev in self.all_devices.items():
            node_id_where_connected = dev.params["node"]
            node = self.all_nodes[node_id_where_connected]
            node.addDevice(dev_id, dev)
        for edge_id, edge in self.all_edges.items():
            node = self.all_nodes[edge.params["nodeFrom"]]
            node.addEdge(edge_id, edge, "from")
            node = self.all_nodes[edge.params["nodeTo"]]
            node.addEdge(edge_id, edge, "to")

        # specify sets:
        # energycarriers=['el','heat','gas','oil','water','wellstream','hydrogen']
        energycarriers = self.all_carriers.keys()
        model.setCarrier = pyo.Set(initialize=energycarriers, doc="energy carriers")
        model.setTerminal = pyo.Set(initialize=["in", "out"], doc="terminals")
        model.setNode = pyo.Set(initialize=self.all_nodes.keys(), doc="nodes")
        model.setDevice = pyo.Set(initialize=self.all_devices.keys(), doc="devices")
        model.setEdge = pyo.Set(initialize=self.all_edges.keys(), doc="edges")
        # model.setFlowComponent= pyo.Set(initialize=['oil','gas','water'])
        model.setHorizon = pyo.Set(ordered=True, initialize=timerange)
        # model.setParameters = pyo.Set()
        model.setProfile = pyo.Set(initialize=profiles_in_use)

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

        # specify variables:
        model.varEdgeFlow = pyo.Var(model.setEdge, model.setHorizon, within=pyo.Reals)
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

        # specify objective:
        obj = self.optimisation_parameters["objective"]
        if obj == "co2":
            rule = self._rule_objective_co2
        elif obj == "costs":
            rule = self._rule_objective_costs
        elif obj == "exportRevenue":
            rule = self._rule_objective_exportRevenue
        elif obj == "co2intensity":
            rule = self._rule_objective_co2intensity
        else:
            raise Exception("Objective '{}' has not been implemented".format(obj))
        logging.info("Using objective function  {}".format(obj))
        # logging.info(rule)
        model.objObjective = pyo.Objective(rule=rule, sense=pyo.minimize)

        # Specify initial values from input data
        for i, dev in self.all_devices.items():
            dev.setInitValues()

        # Keep track of duals:
        # WARNING:
        # From Gurobi: "Shadow prices are not well-defined in mixed-integer
        # problems, so we don't provide dual values for an integer program."
        model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

        return model

    def _specifyConstraints(self):
        # specify constraints:
        model = self.pyomo_instance
        for dev_id, dev in self.all_devices.items():
            dev.defineConstraints()

        for node_id, node in self.all_nodes.items():
            node.defineConstraints()

        for edge_id, edge in self.all_edges.items():
            edge.defineConstraints()

        params_generic = self.optimisation_parameters
        if ("emissionRateMax" in params_generic) and (
            params_generic["emissionRateMax"] >= 0
        ):
            model.constrO_emissionrate = pyo.Constraint(
                model.setHorizon, rule=self._rule_emissionRateLimit
            )
        else:
            logging.info("No emissionRateMax limit specified")
        if ("emissionIntensityMax" in params_generic) and (
            params_generic["emissionIntensityMax"] >= 0
        ):
            model.constrO_emissionintensity = pyo.Constraint(
                model.setHorizon, rule=self._rule_emissionIntensityLimit
            )
        else:
            logging.info("No emissionIntensityMax limit specified")

        if ("elReserveMargin" in params_generic) and (
            params_generic["elReserveMargin"] >= 0
        ):
            model.constrO_elReserveMargin = pyo.Constraint(
                model.setHorizon, rule=self._rule_elReserveMargin
            )
        else:
            logging.info("No elReserveMargin limit specified")
        if ("elBackupMargin" in params_generic) and (
            params_generic["elBackupMargin"] >= 0
        ):
            model.constrO_elBackupMargin = pyo.Constraint(
                model.setDevice, model.setHorizon, rule=self._rule_elBackupMargin
            )
        else:
            logging.info("No elBackupMargin limit specified")

    def updateOptimisationModel(self, timestep, profiles, first=False):
        """Update Pyomo model instance

        Parameters
        ----------
        timestep : int
            Present timestep
        profiles : pandas dataframe
            Time series profiles
        first : booblean
            True if it is the first timestep in rolling optimisation
        """
        opt_timesteps = self.optimisation_parameters["optimisation_timesteps"]
        horizon = self.optimisation_parameters["planning_horizon"]
        timesteps_use_actual = self.optimisation_parameters["forecast_timesteps"]
        self._timestep = timestep
        # Update profile (using actual for first 4 timesteps, forecast for rest)
        # -this is because for the first timesteps, we tend to have updated
        #  and quite good forecasts - for example, it may become apparent
        #  that there will be much less wind power than previously forecasted
        #
        for prof in self.pyomo_instance.setProfile:
            for t in range(timesteps_use_actual):  # 0,1,2,3
                self.pyomo_instance.paramProfiles[prof, t] = profiles["actual"].loc[
                    timestep + t, prof
                ]
            for t in range(timesteps_use_actual, horizon):
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
                if self.pyomo_instance.varDeviceIsPrep[dev, tt] == 1:
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
                # On/off status:
                self.pyomo_instance.paramDeviceIsOnInitially[
                    dev
                ] = self.pyomo_instance.varDeviceIsOn[dev, t_prev]
                self.pyomo_instance.paramDevicePrepTimestepsInitially[
                    dev
                ] = _updateOnTimesteps(t_prev, dev)
                # Initial power output (relevant for ramp rate constraint):
                if "maxRampUp" in dev_obj.params:
                    self.pyomo_instance.paramDevicePowerInitially[
                        dev
                    ] = dev_obj.getPowerVar(t_prev)
                # Energy storage:
                if dev_obj in self.devices_with_storage:
                    self.pyomo_instance.paramDeviceEnergyInitially[
                        dev
                    ] = self.pyomo_instance.varDeviceStorageEnergy[dev, t_prev]
                    if "target_profile" in dev_obj.params:
                        prof = dev_obj.params["target_profile"]
                        Emax = dev_obj.params["Emax"]
                        self.pyomo_instance.paramDeviceEnergyTarget[dev] = (
                            Emax
                            * self._df_profiles_forecast.loc[timestep + horizon, prof]
                        )

        # These constraints need to be reconstructed to update properly
        # (pyo.value(...) and/or if sentences...)
        for c in self.constraints_to_reconstruct:
            # logging.debug("reconstructing {}".format(c))
            c.reconstruct()
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
        startupCosts = self.compute_startup_costs(model)  # kr/s
        storageDepletionCosts = self.compute_costForDepletedStorage(model)
        opCosts = self.compute_operatingCosts(model)  # kr/s
        co2 = self.compute_CO2(model)  # kgCO2/s
        co2_tax = self.optimisation_parameters["co2_tax"]  # kr/kgCO2
        co2Cost = co2 * co2_tax  # kr/s
        sumCost = co2Cost + startupCosts + storageDepletionCosts + opCosts
        return sumCost

    def _rule_objective_exportRevenue(self, model):
        """revenue from exported oil and gas minus costs (co2 price and
        operating costs) per second"""
        sumRevenue = self.compute_exportRevenue(model)  # kr/s
        startupCosts = self.compute_startup_costs(model)  # kr/s
        co2 = self.compute_CO2(model)  # kgCO2/s
        co2_tax = self.optimisation_parameters["co2_tax"]  # kr/kgCO2
        co2Cost = co2 * co2_tax  # kr/s
        storageDepletionCosts = self.compute_costForDepletedStorage(model)
        opCosts = self.compute_operatingCosts(model)  # kr/s
        sumCost = -sumRevenue + co2Cost + startupCosts + storageDepletionCosts + opCosts
        return sumCost

    def _rule_emissionRateLimit(self, model, t):
        """Upper limit on CO2 emission rate"""
        params_generic = self.optimisation_parameters
        emissionRateMax = params_generic["emissionRateMax"]
        lhs = self.compute_CO2(model, timesteps=[t])
        rhs = emissionRateMax
        return lhs <= rhs

    def _rule_emissionIntensityLimit(self, model, t):
        """Upper limit on CO2 emission intensity"""
        params_generic = self.optimisation_parameters
        emissionIntensityMax = params_generic["emissionIntensityMax"]
        lhs = self.compute_CO2(model, timesteps=[t])
        rhs = emissionIntensityMax * self.compute_oilgas_export(model, timesteps=[t])
        return lhs <= rhs

    def _rule_elReserveMargin(self, model, t):
        """Reserve margin constraint (electrical supply)
        Not used capacity by power suppliers/storage/load flexibility
        must be larger than some specified margin
        (to cope with unforeseen variations)
        """
        params_generic = self.optimisation_parameters
        # exclude constraint for first timesteps since the point of the
        # dispatch margin is exactly to cope with forecast errors
        # *2 to make sure there is time to start up gt
        if t < params_generic["forecast_timesteps"]:
            return pyo.Constraint.Skip

        margin = params_generic["elReserveMargin"]
        capacity_unused = self.compute_elReserve(model, t)
        expr = capacity_unused >= margin
        return expr

    def _rule_elBackupMargin(self, model, dev, t):
        """Backup capacity constraint (electrical supply)
        Not used capacity by other online power suppliers plus sheddable
        load must be larger than power output of this device
        (to take over in case of a fault)

        elBackupMargin is zero or positive (if loss of load is acceptable)
        """
        params_generic = self.optimisation_parameters
        dev_obj = self.all_devices[dev]
        margin = params_generic["elBackupMargin"]
        if "el" not in dev_obj.carrier_out:
            # this is not a power generator
            return pyo.Constraint.Skip
        res_otherdevs = self.compute_elReserve(model, t, exclude_device=dev)
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
            thisCO2 = dev.compute_CO2(timesteps)
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
            # logging.debug("zero export, so co2 intensity set to None")
            co2intensity = None
        return co2intensity

    def compute_startup_costs(self, model, devices=None, timesteps=None):
        """startup costs (average per sec)"""
        if timesteps is None:
            timesteps = model.setHorizon
        if devices is None:
            devices = model.setDevice
        start_stop_costs = 0
        for d in devices:
            dev_obj = self.all_devices[d]
            thisCost = dev_obj.compute_startup_costs(timesteps)
            start_stop_costs += thisCost
        # get average per sec:
        deltaT = self.optimisation_parameters["time_delta_minutes"] * 60
        sumTime = len(timesteps) * deltaT
        start_stop_costs = start_stop_costs / sumTime
        return start_stop_costs

    logging.info("TODO: operating cost for el storage - needs improvement")

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
            thisCost = dev_obj.compute_operatingCosts(timesteps)
            sumCost += thisCost
        return sumCost

    def compute_costForDepletedStorage(self, model):
        """term in objective function to discourage depleting battery,
        making sure it is used only when required"""
        storCost = 0
        timesteps = model.setHorizon
        for dev in model.setDevice:
            dev_obj = self.all_devices[dev]
            thisCost = dev_obj.compute_costForDepletedStorage(timesteps)
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
            thisValue = dev_obj.compute_export(value, carriers, timesteps)
            sumValue += thisValue
        # average per second (timedelta is not required)
        sumValue = sumValue / len(timesteps)
        return sumValue

    def compute_elReserve(self, model, t, exclude_device=None):
        """compute non-used generator capacity (and available loadflex)
        This is reserve to cope with forecast errors, e.g. because of wind
        variation or motor start-up
        (does not include load reduction yet)

        exclue_device : str (default None)
            compute reserve by devices excluding this one
        """
        alldevs = [d for d in model.setDevice if d != exclude_device]
        # relevant devices are devices with el output or input
        cap_avail = 0
        p_generating = 0
        loadreduction = 0
        for d in alldevs:
            dev_obj = self.all_devices[d]
            reserve = dev_obj.compute_elReserve(t)
            cap_avail += reserve["capacity_available"]
            p_generating += reserve["capacity_used"]
            loadreduction += reserve["loadreduction_available"]

        res_dev = (cap_avail - p_generating) + loadreduction
        # logging.info("TODO: elReserve: Ignoring load reduction option")
        # res_dev = (cap_avail-p_generating)
        return res_dev

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
