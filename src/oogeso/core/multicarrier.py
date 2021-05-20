"""This is the main module containing the Multicarrier energy system class"""

import pyomo.environ as pyo
import pyomo.opt as pyopt
import numpy as np
import pandas as pd
import logging
from . import milp_definition
from . import milp_compute


# get progress bar:
try:
    from tqdm import trange

    has_tqdm = True
except:
    logging.info("Consider installing tqdm to get progress bar")
    trange = range
    has_tqdm = False


ZERO_WARNING_THRESHOLD = 1e-6


class Multicarrier:
    """Multicarrier energy system"""

    def __init__(self, loglevel=logging.INFO, logfile=None):
        """Create Multicarrier energy system object

        Parameters
        ----------
        loglevel : int
            logging level (default=logging.INFO)
        logfile : string
            name of log file (optional)
        """
        # logging.basicConfig(filename=logfile,level=loglevel,
        #                    format='%(asctime)s %(levelname)s: %(message)s',
        #                    datefmt='%Y-%m-%d %H:%M:%S')
        logging.basicConfig()
        logger = logging.getLogger()
        logger.setLevel(loglevel)
        logging.debug("Initialising Multicarrier")
        # Abstract model:
        # self._quadraticConstraints = quadraticConstraints
        self.model = milp_definition.definePyomoModel()
        milp_definition.check_constraints_complete(self.model)
        # Concrete model instance:
        self.instance = None
        # Dataframes keeping track of simulation results:
        self._dfDeviceFlow = None
        self._dfDeviceIsPrep = None
        self._dfDeviceIsOn = None
        self._dfDeviceStorageEnergy = None
        self._dfDeviceStoragePmax = None
        self._dfDeviceStarting = None
        self._dfDeviceStopping = None
        self._dfEdgeFlow = None
        self._dfElVoltageAngle = None
        self._dfTerminalPressure = None
        self._dfTerminalFlow = None
        self._dfCO2rate = None  # co2 emission sum per timestep
        self._dfCO2rate_per_dev = None  # co2 emission per device per timestep
        self._dfExportRevenue = None  # revenue from exported energy
        self._dfCO2intensity = None
        self._dfElReserve = None  # Reserve capacity
        self._dfElBackup = None  # Backup capacity (handling faults)
        self._dfDuals = None

    def createModelInstance(self, data, profiles, filename=None):
        """Create concrete Pyomo model instance

        data : dict of data
        filename : name of file to write model to (optional)
        """
        self._df_profiles_actual = profiles["actual"]
        self._df_profiles_forecast = profiles["forecast"]
        # self._df_profiles = self._df_profiles_forecast.loc[
        #        data['setHorizon'][None]]

        instance = self.model.create_instance(data={None: data}, name="MultiCarrier")
        instance.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

        if filename is not None:
            instance.pprint(filename=filename)
        self.instance = instance
        return instance

    def updateModelInstance(self, timestep, first=False):
        """Update Pyomo model instance

        Parameters
        ----------
        timestep : int
            Present timestep
        first : booblean
            True if it is the first timestep in rolling optimisation
        """
        opt_timesteps = self.instance.paramParameters["optimisation_timesteps"]
        horizon = self.instance.paramParameters["planning_horizon"]
        timesteps_use_actual = self.instance.paramParameters["forecast_timesteps"]
        self._timestep = timestep
        # Update profile (using actual for first 4 timesteps, forecast for rest)
        # -this is because for the first timesteps, we tend to have updated
        #  and quite good forecasts - for example, it may become apparent
        #  that there will be much less wind power than previously forecasted
        #
        for prof in self.instance.setProfile:
            for t in range(timesteps_use_actual):  # 0,1,2,3
                self.instance.paramProfiles[prof, t] = self._df_profiles_actual.loc[
                    timestep + t, prof
                ]
            for t in range(timesteps_use_actual, horizon):
                self.instance.paramProfiles[prof, t] = self._df_profiles_forecast.loc[
                    timestep + t, prof
                ]

        def _updateOnTimesteps(t_prev, dev):
            # sum up consequtive timesteps starting at tprev going
            # backwards, where device has been in preparation phase
            sum_on = 0
            docontinue = True
            for tt in range(t_prev, -1, -1):
                # if (self.instance.varDeviceIsOn[dev,tt]==1):
                if self.instance.varDeviceIsPrep[dev, tt] == 1:
                    sum_on = sum_on + 1
                else:
                    docontinue = False
                    break  # exit for loop
            if docontinue:
                # we got all the way back to 0, so must include initial value
                sum_on = sum_on + self.instance.paramDevicePrepTimestepsInitially[dev]
            return sum_on

        # Update startup/shutdown info
        # pick the last value from previous optimistion prior to the present time
        if not first:
            t_prev = opt_timesteps - 1
            for dev in self.instance.setDevice:
                # On/off status:
                self.instance.paramDeviceIsOnInitially[
                    dev
                ] = self.instance.varDeviceIsOn[dev, t_prev]
                self.instance.paramDevicePrepTimestepsInitially[
                    dev
                ] = _updateOnTimesteps(t_prev, dev)
                # Power output (relevant for ramp rate constraint):
                self.instance.paramDevicePowerInitially[
                    dev
                ] = milp_compute.getDevicePower(self.instance, dev, t_prev)
                # Energy storage:
                storagemodels = milp_compute.models_with_storage
                if self.instance.paramDevice[dev]["model"] in storagemodels:
                    self.instance.paramDeviceEnergyInitially[
                        dev
                    ] = self.instance.varDeviceStorageEnergy[dev, t_prev]
                    if "target_profile" in self.instance.paramDevice[dev]:
                        prof = self.instance.paramDevice[dev]["target_profile"]
                        Emax = self.instance.paramDevice[dev]["Emax"]
                        self.instance.paramDeviceEnergyTarget[dev] = (
                            Emax
                            * self._df_profiles_forecast.loc[timestep + horizon, prof]
                        )

        # These constraints need to be reconstructed to update properly
        # (pyo.value(...) and/or if sentences...)
        self.instance.constrDevice_startup_delay.reconstruct()
        self.instance.constrDevice_startup_shutdown.reconstruct()
        return

    def getVarValues(self, variable, names):
        """Extract MILP problem variable values as dataframe"""
        df = pd.DataFrame.from_dict(variable.get_values(), orient="index")
        df.index = pd.MultiIndex.from_tuples(df.index, names=names)
        return df[0].dropna()

    #        if unstack is None:
    #            df = df[0]
    #        else:
    #            df = df[0].unstack(level=unstack)
    #        df = df.dropna()
    #        return df

    def _keep_decision(self, df, timelimit, timeshift):
        """extract decision variables (first timesteps) from dataframe"""
        level = df.index.names.index("time")
        df = df[df.index.get_level_values(level) < timelimit]
        df.index.set_levels(
            df.index.levels[level] + timeshift, level=level, inplace=True
        )
        return df

    def saveOptimisationResult(self, timestep, store_duals=None):
        """save results of optimisation for later analysis"""

        # TODO: Implement result storage
        # hdf5? https://stackoverflow.com/questions/47072859/how-to-append-data-to-one-specific-dataset-in-a-hdf5-file-with-h5py)

        timelimit = self.instance.paramParameters["optimisation_timesteps"]
        timeshift = timestep

        # Retrieve variable values
        varDeviceFlow = self.getVarValues(
            self.instance.varDeviceFlow, names=("device", "carrier", "terminal", "time")
        )
        if (varDeviceFlow < -ZERO_WARNING_THRESHOLD).any():
            # get first index where this is true
            ind = varDeviceFlow[varDeviceFlow < -ZERO_WARNING_THRESHOLD].index[0]
            logging.warning(
                "Negative number in varDeviceFlow - set to zero ({}:{})".format(
                    ind, varDeviceFlow[ind]
                )
            )
            varDeviceFlow = varDeviceFlow.clip(lower=0)
        varDeviceIsPrep = self.getVarValues(
            self.instance.varDeviceIsPrep, names=("device", "time")
        )
        varDeviceIsOn = self.getVarValues(
            self.instance.varDeviceIsOn, names=("device", "time")
        )
        varDeviceStorageEnergy = self.getVarValues(
            self.instance.varDeviceStorageEnergy, names=("device", "time")
        )
        varDeviceStoragePmax = self.getVarValues(
            self.instance.varDeviceStoragePmax, names=("device", "time")
        )
        varDeviceStarting = self.getVarValues(
            self.instance.varDeviceStarting, names=("device", "time")
        )
        varDeviceStopping = self.getVarValues(
            self.instance.varDeviceStopping, names=("device", "time")
        )
        varEdgeFlow = self.getVarValues(
            self.instance.varEdgeFlow, names=("edge", "time")
        )
        varElVoltageAngle = self.getVarValues(
            self.instance.varElVoltageAngle, names=("node", "time")
        )
        varPressure = self.getVarValues(
            self.instance.varPressure, names=("node", "carrier", "terminal", "time")
        )
        varTerminalFlow = self.getVarValues(
            self.instance.varTerminalFlow, names=("node", "carrier", "time")
        )

        if store_duals is not None:
            # Save dual values
            # store_duals = {
            #   'elcost': {'constr':'constrDevicePmin','indx':('util',None)}
            #   }
            horizon_steps = self.instance.paramParameters["planning_horizon"]
            if self._dfDuals is None:
                self._dfDuals = pd.DataFrame(columns=store_duals.keys())
            for key, val in store_duals.items():
                # vrs=('util',None)
                vrs = val["indx"]
                constr = getattr(self.instance, val["constr"])
                sumduals = 0
                for t in range(timelimit):
                    # Replace None by the timestep, ('util',None) -> ('util',t)
                    vrs1 = tuple(x if x is not None else t for x in vrs)
                    dual = self.instance.dual[constr[vrs1]]
                    # The dual gives the improvement in the objective function
                    # if the constraint is relaxed by one unit.
                    # The units of the dual prices are the units of the
                    # objective function divided by the units of the constraint.
                    #
                    # A constraint is for a single timestep, whereas the
                    # objective function averages over all timesteps in the
                    # optimisation horizon. To get the improvement of relaxing
                    # the constraint not just in the single timestep, but in
                    # all timesteps we therefore scale up the dual value
                    dual = dual * horizon_steps
                    self._dfDuals.loc[timeshift + t, key] = dual

        # CO2 emission rate (sum all emissions)
        co2 = [
            pyo.value(milp_compute.compute_CO2(self.instance, timesteps=[t]))
            for t in range(timelimit)
        ]
        self._dfCO2rate = pd.concat(
            [
                self._dfCO2rate,
                pd.Series(data=co2, index=range(timestep, timestep + timelimit)),
            ]
        )

        # CO2 emission rate per device:
        df_co2 = pd.DataFrame()
        for d in self.instance.setDevice:
            for t in range(timelimit):
                co2_dev = milp_compute.compute_CO2(
                    self.instance, devices=[d], timesteps=[t]
                )
                df_co2.loc[t + timestep, d] = pyo.value(co2_dev)
        self._dfCO2rate_per_dev = pd.concat([self._dfCO2rate_per_dev, df_co2])
        # self._dfCO2dev.sort_index(inplace=True)

        # CO2 emission intensity (sum)
        df_df_co2intensity = [
            pyo.value(milp_compute.compute_CO2_intensity(self.instance, timesteps=[t]))
            for t in range(timelimit)
        ]
        self._dfCO2intensity = pd.concat(
            [
                self._dfCO2intensity,
                pd.Series(
                    data=df_df_co2intensity, index=range(timestep, timestep + timelimit)
                ),
            ]
        )

        # Revenue from exported energy
        df_exportRevenue = pd.DataFrame()
        for c in self.instance.setCarrier:
            for t in range(timelimit):
                exportRevenue_dev = milp_compute.compute_exportRevenue(
                    self.instance, carriers=[c], timesteps=[t]
                )
                df_exportRevenue.loc[t + timestep, c] = pyo.value(exportRevenue_dev)
        self._dfExportRevenue = pd.concat([self._dfExportRevenue, df_exportRevenue])

        # Reserve capacity
        # for all generators, should have reserve_excl_generator>p_out
        df_reserve = pd.DataFrame()
        devs_elout = self.getDevicesInout(carrier_out="el")
        for t in range(timelimit):
            rescap = pyo.value(milp_compute.compute_elReserve(self.instance, t))
            df_reserve.loc[t + timestep, "reserve"] = rescap
        self._dfElReserve = pd.concat([self._dfElReserve, df_reserve])

        # Backup capacity
        # for all generators, should have reserve_excl_generator>p_out
        df_backup = pd.DataFrame()
        devs_elout = self.getDevicesInout(carrier_out="el")
        for t in range(timelimit):
            for d in devs_elout:
                rescap = pyo.value(
                    milp_compute.compute_elReserve(self.instance, t, exclude_device=d)
                )
                df_backup.loc[t + timestep, d] = rescap
        self._dfElBackup = pd.concat([self._dfElBackup, df_backup])

        # Add to dataframes storing results (only the decision variables)
        def _addToDf(df_prev, df_new):
            level = df_new.index.names.index("time")
            df_new = df_new[df_new.index.get_level_values(level) < timelimit]
            df_new.index.set_levels(
                df_new.index.levels[level] + timeshift, level=level, inplace=True
            )
            df = pd.concat([df_prev, df_new])
            df.sort_index(inplace=True)
            return df

        self._dfDeviceFlow = _addToDf(self._dfDeviceFlow, varDeviceFlow)
        self._dfDeviceIsOn = _addToDf(self._dfDeviceIsOn, varDeviceIsOn)
        self._dfDeviceIsPrep = _addToDf(self._dfDeviceIsPrep, varDeviceIsPrep)
        self._dfDeviceStorageEnergy = _addToDf(
            self._dfDeviceStorageEnergy, varDeviceStorageEnergy
        )
        self._dfDeviceStoragePmax = _addToDf(
            self._dfDeviceStoragePmax, varDeviceStoragePmax
        )
        self._dfDeviceStarting = _addToDf(self._dfDeviceStarting, varDeviceStarting)
        self._dfDeviceStopping = _addToDf(self._dfDeviceStopping, varDeviceStopping)
        self._dfEdgeFlow = _addToDf(self._dfEdgeFlow, varEdgeFlow)
        self._dfElVoltageAngle = _addToDf(self._dfElVoltageAngle, varElVoltageAngle)
        self._dfTerminalPressure = _addToDf(self._dfTerminalPressure, varPressure)
        self._dfTerminalFlow = _addToDf(self._dfTerminalFlow, varTerminalFlow)
        return

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
        sol = opt.solve(self.instance)

        if write_yaml:
            sol.write_yaml()

        if (sol.solver.status == pyopt.SolverStatus.ok) and (
            sol.solver.termination_condition == pyopt.TerminationCondition.optimal
        ):
            logging.debug("Solved OK")
        elif sol.solver.termination_condition == pyopt.TerminationCondition.infeasible:
            raise Exception("Infeasible solution")
        else:
            # Something else is wrong
            logging.info("Solver Status:{}".format(sol.solver.status))
        return sol

    def solveMany(
        self,
        solver="gurobi",
        timerange=None,
        write_yaml=False,
        timelimit=None,
        store_duals=None,
    ):
        """Solve problem over many timesteps - rolling horizon

        Parameters
        ----------
        solver : string
            Name of solver ("cbc", "gurobi")
        timerange : list = [start,end]
            List of two elments giving the timestep start and end of
            the time window to investigate - timesteps are defined by
            the timeseries profiles used (timestep=row)
        write_yaml : boolean
            Whether to save problem to yaml file (for debugging)
        timelimit : int
            Time limit spent on each optimisation (sec)
        store_duals : dict
            Store dual values of constraints. The dictionary contains a
            key:value list where value is a new dictionary specifying the
            constraint and indices. None in the index is replaced by time
            Example:
            store_duals = {'elcost':{constr=constrDevicePmin, indx:('util',None)}
        """

        steps = self.instance.paramParameters["optimisation_timesteps"]
        horizon = self.instance.paramParameters["planning_horizon"]
        if timelimit is not None:
            logging.debug("Using solver timelimit={}".format(timelimit))
        if timerange is None:
            time_start = 0
            time_end = self._df_profiles_forecast.index.max() + 1 - horizon
        else:
            time_start = timerange[0]
            time_end = timerange[1]

        first = True
        for step in trange(time_start, time_end, steps):
            if not has_tqdm:
                # no progress bar
                logging.info("Solving timestep={}".format(step))
            # 1. Update problem formulation
            self.updateModelInstance(step, first=first)
            # 2. Solve for planning horizon
            self.solve(solver=solver, write_yaml=write_yaml, timelimit=timelimit)
            # 3. Save results (for later analysis)
            self.saveOptimisationResult(step, store_duals)
            first = False

    def printSolution(self, instance):
        """Print solution (edge flow and pressure variables) to screen"""
        print("\nSOLUTION - edgeFlow:")
        for k in instance.varEdgeFlow.keys():
            power = instance.varEdgeFlow[k].value
            print("  {}: {}".format(k, power))

        print("\nSOLUTION - Pressure:")
        for k, v in instance.varPressure.get_values().items():
            pressure = v  # pyo.value(v)
            print("  {}: {}".format(k, pressure))
            # df_edge.loc[k,'gasPressure'] = pressure

        # display all duals
        print("Duals")
        for c in instance.component_objects(pyo.Constraint, active=True):
            print("   Constraint", c)
            for index in c:
                print("      ", index, instance.dual[c[index]])

    def nodeIsNonTrivial(self, node, carrier):
        """returns True if edges or devices are connected to node for this carrier"""
        model = self.instance
        isNontrivial = False
        # edges connected?
        if ((carrier, node) in model.paramNodeEdgesFrom) or (
            (carrier, node) in model.paramNodeEdgesTo
        ):
            isNontrivial = True
            return isNontrivial
        # devices connected?
        all_devmodels = milp_compute.devicemodel_inout()
        if node in model.paramNodeDevices:
            mydevs = model.paramNodeDevices[node]
            devmodels = [model.paramDevice[d]["model"] for d in mydevs]
            for dev_model in devmodels:
                carriers_used = [
                    item
                    for sublist in list(all_devmodels[dev_model].values())
                    for item in sublist
                ]
                if carrier in carriers_used:
                    isNontrivial = True
                    return isNontrivial
        return isNontrivial

    def OBSOLETE_getProfiles(self, names):
        """Extract timeseries profiles from MILP problem"""
        if not type(names) is list:
            names = [names]
        df = pd.DataFrame.from_dict(
            self.instance.paramProfiles.extract_values(), orient="index"
        )
        df.index = pd.MultiIndex.from_tuples(df.index)
        df = df[0].unstack(level=0)
        return df

    def getDevicesInout(self, carrier_in=None, carrier_out=None):
        """devices that have the specified connections in and out"""
        model = self.instance
        inout = milp_compute.devicemodel_inout()
        devs = []
        for d in model.setDevice:
            devmodel = model.paramDevice[d]["model"]
            ok_in = (carrier_in is None) or (carrier_in in inout[devmodel]["in"])
            ok_out = (carrier_out is None) or (carrier_out in inout[devmodel]["out"])
            if ok_in and ok_out:
                devs.append(d)
        return devs

    def checkEdgePressureDrop(self, timestep=0, var="outer"):
        """Compute and display pressure drop along edges"""
        model = self.instance
        for k, edge in model.paramEdge.items():
            carrier = edge["type"]
            if carrier in ["gas", "oil", "wellstream", "water"]:
                node_from = edge["nodeFrom"]
                node_to = edge["nodeTo"]
                print(
                    "{} edge {}:{}-{}".format(carrier, k, node_from, node_to), end=" "
                )
                if not (
                    ("pressure.{}.out".format(carrier) in model.paramNode[node_from])
                    and ("pressure.{}.in".format(carrier) in model.paramNode[node_to])
                ):
                    print("--")
                    continue
                p_in = model.paramNode[node_from]["pressure.{}.out".format(carrier)]
                p_out = model.paramNode[node_to]["pressure.{}.in".format(carrier)]
                if p_out == p_in:
                    # no nominal pressure drop - i.e. pressure drop is not
                    # modelled, so skip to next
                    print("---")
                    continue
                print("{} edge {}:{}-{}".format(carrier, k, node_from, node_to))
                if var == "inner":
                    # get value from inner optimisation (over rolling horizon)
                    Q = model.varEdgeFlow[(k, timestep)]
                    var_p_in = model.varPressure[(node_from, carrier, "out", timestep)]
                    var_p_out = model.varPressure[(node_to, carrier, "in", timestep)]
                elif var == "outer":
                    # get value from outer loop (decisions)
                    Q = self._dfEdgeFlow[(k, timestep)]
                    var_p_in = self._dfTerminalPressure[
                        (node_from, carrier, "out", timestep)
                    ]
                    var_p_out = self._dfTerminalPressure[
                        (node_to, carrier, "in", timestep)
                    ]

                print(
                    "{} edge {}:{}-{} (Q={} m3/s)".format(
                        carrier, k, node_from, node_to, Q
                    )
                )
                diameter = model.paramEdge[k]["diameter_mm"] / 1000

                if "num_pipes" in edge:
                    Q = Q / edge["num_pipes"]

                if var in ["inner", "outer"]:
                    p_out_comp = milp_compute.compute_edge_pressuredrop(
                        model, edge=k, p1=p_in, Q=Q, linear=False
                    )
                    p_out_comp_linear = pyo.value(
                        milp_compute.compute_edge_pressuredrop(
                            model, edge=k, p1=p_in, Q=Q, linear=True
                        )
                    )
                    p_out_comp2 = milp_compute.compute_edge_pressuredrop(
                        model, edge=k, p1=var_p_in, Q=Q, linear=False
                    )
                    p_out_comp2_linear = pyo.value(
                        milp_compute.compute_edge_pressuredrop(
                            model, edge=k, p1=var_p_in, Q=Q, linear=True
                        )
                    )

                pressure0 = 0.1  # MPa, standard condition (Sm3)
                velocity = 4 * Q / (np.pi * diameter ** 2)
                if carrier == "gas":
                    # convert flow rate from Sm3/s to m3/s at the actual pressure:
                    # ideal gas pV=const => pQ=const => Q1=Q0*(p0/p1)
                    pressure1 = (var_p_in + var_p_out) / 2
                    Q1 = Q * (pressure0 / pressure1)
                    velocity = 4 * Q1 / (np.pi * diameter ** 2)
                print(
                    (
                        "\tNOMINAL:    pin={}  pout={}  pout_computed={:3.5g}"
                        " pout_linear={:3.5g}"
                    ).format(p_in, p_out, p_out_comp, p_out_comp_linear)
                )
                if var in ["inner", "outer"]:
                    print(
                        (
                            "\tSIMULATION: pin={}  pout={}"
                            "  pout_computed={:3.5g} pout_linear={:3.5g}"
                        ).format(var_p_in, var_p_out, p_out_comp2, p_out_comp2_linear)
                    )
                print("\tflow velocity = {:3.5g} m/s".format(velocity))

    def exportSimulationResult(self, filename):
        """Write saved simulation results to file"""

        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter(filename, engine="xlsxwriter")

        if not self._dfCO2intensity.empty:
            self._dfCO2intensity.to_excel(writer, sheet_name="CO2intensity")
        if not self._dfCO2rate.empty:
            self._dfCO2rate.to_excel(writer, sheet_name="CO2rate")
        if not self._dfCO2rate_per_dev.empty:
            self._dfCO2rate_per_dev.to_excel(writer, sheet_name="CO2rate_per_dev")
        if not self._dfDeviceStorageEnergy.empty:
            self._dfDeviceStorageEnergy.to_excel(
                writer, sheet_name="DeviceStorageEnergy"
            )
        if not self._dfDeviceFlow.empty:
            self._dfDeviceFlow.to_excel(writer, sheet_name="DeviceFlow")
        if not self._dfDeviceIsOn.empty:
            self._dfDeviceIsOn.to_excel(writer, sheet_name="DeviceIsOn")
        # if not self._dfDevicePower.empty:
        #    self._dfDevicePower.to_excel(writer,sheet_name="DevicePower")
        #        if not self._dfDeviceStarting.empty:
        #            self._dfDeviceStarting.to_excel(writer,sheet_name="DeviceStarting")
        #        if not self._dfDeviceStopping.empty:
        #            self._dfDeviceStopping.to_excel(writer,sheet_name="DeviceStopping")
        #        if not self._dfEdgeFlow.empty:
        #            self._dfEdgeFlow.to_excel(writer,sheet_name="EdgeFlow")
        #        if not self._dfElVoltageAngle.empty:
        #            self._dfElVoltageAngle.to_excel(writer,sheet_name="ElVoltageAngle")
        if not self._dfExportRevenue.empty:
            self._dfExportRevenue.to_excel(writer, sheet_name="ExportRevenue")
        #        if not self._dfTerminalFlow.empty:
        #            self._dfTerminalFlow.to_excel(writer,sheet_name="TerminalFlow")
        if not self._dfTerminalPressure.empty:
            self._dfTerminalPressure.to_excel(writer, sheet_name="TerminalPressure")

        # Close the Pandas Excel writer and output the Excel file.
        writer.save()

    def compute_kpis(self, windturbines=[]):
        """Compute key indicators of simulation results"""
        hour_per_year = 8760
        sec_per_year = 3600 * hour_per_year
        kpi = {}
        mc = self

        num_sim_timesteps = mc._dfCO2rate.shape[0]
        timesteps = mc._dfCO2rate.index
        td_min = mc.instance.paramParameters["time_delta_minutes"]
        kpi["hours_simulated"] = num_sim_timesteps * td_min / 60

        # CO2 emissions
        kpi["kgCO2_per_year"] = mc._dfCO2rate.mean() * sec_per_year
        kpi["kgCO2_per_Sm3oe"] = mc._dfCO2intensity.mean()

        # hours with reduced load
        #        kpi['reducedload_hours_per_year'] = None

        # hours with load shedding
        #        kpi['loadshed_hours_per_year'] = None

        # fuel consumption
        gasturbines = [
            i for i, g in mc.instance.paramDevice.items() if g["model"] == "gasturbine"
        ]
        mask_gt = mc._dfDeviceFlow.index.get_level_values("device").isin(gasturbines)
        gtflow = mc._dfDeviceFlow[mask_gt]
        fuel = (
            gtflow.unstack("carrier")["gas"]
            .unstack("terminal")["in"]
            .unstack()
            .mean(axis=1)
        )
        kpi["gt_fuel_sm3_per_year"] = fuel.sum() * sec_per_year

        # electric power consumption
        el_dem = (
            mc._dfDeviceFlow.unstack("carrier")["el"]
            .unstack("terminal")["in"]
            .dropna()
            .unstack()
            .mean(axis=1)
        )
        kpi["elconsumption_mwh_per_year"] = el_dem.sum() * hour_per_year
        kpi["elconsumption_avg_mw"] = el_dem.sum()

        # number of generator starts
        gt_starts = mc._dfDeviceStarting.unstack().sum(axis=1)[gasturbines].sum()
        kpi["gt_starts_per_year"] = gt_starts * hour_per_year / kpi["hours_simulated"]

        # number of generator stops
        gt_stops = mc._dfDeviceStopping.unstack().sum(axis=1)[gasturbines].sum()
        kpi["gt_stops_per_year"] = gt_stops * hour_per_year / kpi["hours_simulated"]

        # running hours of generators
        gt_ison_tsteps = mc._dfDeviceIsOn.unstack().sum(axis=1)[gasturbines].sum()
        gt_ison = gt_ison_tsteps * td_min / 60
        kpi["gt_hoursrunning_per_year"] = (
            gt_ison * hour_per_year / kpi["hours_simulated"]
        )

        # wind power output
        el_sup = (
            mc._dfDeviceFlow.unstack("carrier")["el"]
            .unstack("terminal")["out"]
            .dropna()
            .unstack()
        )
        p_wind = el_sup.T[windturbines]
        kpi["wind_output_mwh_per_year"] = p_wind.sum(axis=1).mean() * hour_per_year

        # curtailed wind energy
        p_avail = pd.DataFrame(index=timesteps)
        for d in windturbines:
            devparam = mc.instance.paramDevice[d]
            Pmax = devparam["Pmax"]
            p_avail[d] = Pmax
            if "profile" in devparam:
                profile_ref = devparam["profile"]
                p_avail[d] = Pmax * mc._df_profiles_actual.loc[timesteps, profile_ref]
        p_curtailed = (p_avail - p_wind).sum(axis=1)
        kpi["wind_curtailed_mwh_per_year"] = p_curtailed.mean() * hour_per_year
        return kpi
