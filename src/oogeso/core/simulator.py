import pandas as pd
import pyomo.environ as pyo

# from . import file_io
from .optimiser import Optimiser
import logging

# get progress bar:
try:
    from tqdm import trange

    has_tqdm = True
except:
    logging.info("Consider installing tqdm to get progress bar")
    trange = range
    has_tqdm = False


ZERO_WARNING_THRESHOLD = 1e-6


class Simulator:
    """Main class for Oogeso energy system simulations"""

    def __init__(self, optimiser):
        """Create Simulator object

        """

        # Abstract pyomo model formulation
        self.optimiser = optimiser

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

    def setOptimiser(self, optimiser):
        self.optimiser = optimiser

    def runSimulation(
        self,
        solver,
        profiles,
        timerange=None,
        timelimit=None,
        store_duals=None,
        write_yaml=False,
    ):
        """Solve problem over many timesteps - rolling horizon

        Parameters
        ----------
        solver : string
            Name of solver ("cbc", "gurobi")
        profiles : {'actual': dataframe1, 'forecast':dataframe2}
            Dataframes with actual and forecast timeseries
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

        self._df_profiles_forecast = profiles["actual"]

        steps = self.optimiser.optimisation_parameters["optimisation_timesteps"]
        horizon = self.optimiser.optimisation_parameters["planning_horizon"]
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
            self.optimiser.updateOptimisationModel(step, first=first, profiles=profiles)
            # 2. Solve for planning horizon
            self.optimiser.solve(
                solver=solver, write_yaml=write_yaml, timelimit=timelimit
            )
            # 3. Save results (for later analysis)
            self._saveOptimisationResult(step, store_duals)
            first = False

    def _getVarValues(self, variable, names):
        """Extract MILP problem variable values as dataframe"""
        df = pd.DataFrame.from_dict(variable.get_values(), orient="index")
        df.index = pd.MultiIndex.from_tuples(df.index, names=names)
        return df[0].dropna()

    def _addToDf(self, df_prev, df_new, timelimit, timeshift):
        """Add to dataframes storing results (only the decision variables)"""
        level = df_new.index.names.index("time")
        df_new = df_new[df_new.index.get_level_values(level) < timelimit]
        df_new.index.set_levels(
            df_new.index.levels[level] + timeshift, level=level, inplace=True
        )
        df = pd.concat([df_prev, df_new])
        df.sort_index(inplace=True)
        return df

    def _saveOptimisationResult(self, timestep, store_duals=None):
        """save results of optimisation for later analysis"""

        # TODO: Implement result storage
        # hdf5? https://stackoverflow.com/questions/47072859/how-to-append-data-to-one-specific-dataset-in-a-hdf5-file-with-h5py)
        pyomo_instance = self.optimiser.pyomo_instance
        timelimit = self.optimiser.optimisation_parameters["optimisation_timesteps"]
        timeshift = timestep

        # Retrieve variable values
        varDeviceFlow = self._getVarValues(
            pyomo_instance.varDeviceFlow,
            names=("device", "carrier", "terminal", "time"),
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
        varDeviceIsPrep = self._getVarValues(
            pyomo_instance.varDeviceIsPrep, names=("device", "time")
        )
        varDeviceIsOn = self._getVarValues(
            pyomo_instance.varDeviceIsOn, names=("device", "time")
        )
        varDeviceStorageEnergy = self._getVarValues(
            pyomo_instance.varDeviceStorageEnergy, names=("device", "time")
        )
        varDeviceStoragePmax = self._getVarValues(
            pyomo_instance.varDeviceStoragePmax, names=("device", "time")
        )
        varDeviceStarting = self._getVarValues(
            pyomo_instance.varDeviceStarting, names=("device", "time")
        )
        varDeviceStopping = self._getVarValues(
            pyomo_instance.varDeviceStopping, names=("device", "time")
        )
        varEdgeFlow = self._getVarValues(
            pyomo_instance.varEdgeFlow, names=("edge", "time")
        )
        varElVoltageAngle = self._getVarValues(
            pyomo_instance.varElVoltageAngle, names=("node", "time")
        )
        varPressure = self._getVarValues(
            pyomo_instance.varPressure, names=("node", "carrier", "terminal", "time")
        )
        varTerminalFlow = self._getVarValues(
            pyomo_instance.varTerminalFlow, names=("node", "carrier", "time")
        )

        if store_duals is not None:
            # Save dual values
            # store_duals = {
            #   'elcost': {'constr':'constrDevicePmin','indx':('util',None)}
            #   }
            horizon_steps = pyomo_instance.paramParameters["planning_horizon"]
            if self._dfDuals is None:
                self._dfDuals = pd.DataFrame(columns=store_duals.keys())
            for key, val in store_duals.items():
                # vrs=('util',None)
                vrs = val["indx"]
                constr = getattr(pyomo_instance, val["constr"])
                sumduals = 0
                for t in range(timelimit):
                    # Replace None by the timestep, ('util',None) -> ('util',t)
                    vrs1 = tuple(x if x is not None else t for x in vrs)
                    dual = pyomo_instance.dual[constr[vrs1]]
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
            pyo.value(self.optimiser.compute_CO2(pyomo_instance, timesteps=[t]))
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
        for d in pyomo_instance.setDevice:
            for t in range(timelimit):
                co2_dev = self.optimiser.compute_CO2(
                    pyomo_instance, devices=[d], timesteps=[t]
                )
                df_co2.loc[t + timestep, d] = pyo.value(co2_dev)
        self._dfCO2rate_per_dev = pd.concat([self._dfCO2rate_per_dev, df_co2])
        # self._dfCO2dev.sort_index(inplace=True)

        # CO2 emission intensity (sum)
        df_df_co2intensity = [
            pyo.value(
                self.optimiser.compute_CO2_intensity(pyomo_instance, timesteps=[t])
            )
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
        for c in pyomo_instance.setCarrier:
            for t in range(timelimit):
                exportRevenue_dev = self.optimiser.compute_exportRevenue(
                    pyomo_instance, carriers=[c], timesteps=[t]
                )
                df_exportRevenue.loc[t + timestep, c] = pyo.value(exportRevenue_dev)
        self._dfExportRevenue = pd.concat([self._dfExportRevenue, df_exportRevenue])

        # Reserve capacity
        # for all generators, should have reserve_excl_generator>p_out
        df_reserve = pd.DataFrame()
        for t in range(timelimit):
            rescap = pyo.value(self.optimiser.compute_elReserve(pyomo_instance, t))
            df_reserve.loc[t + timestep, "reserve"] = rescap
        self._dfElReserve = pd.concat([self._dfElReserve, df_reserve])

        # Backup capacity
        # for all generators, should have reserve_excl_generator>p_out
        df_backup = pd.DataFrame()
        # devs_elout = self.getDevicesInout(carrier_out='el')
        devs_elout = []
        for dev_id, dev_obj in self.optimiser.all_devices.items():
            if "el" in dev_obj.carrier_out:
                devs_elout.append(dev_obj.dev_id)
        for t in range(timelimit):
            for d in devs_elout:
                rescap = pyo.value(
                    self.optimiser.compute_elReserve(
                        pyomo_instance, t, exclude_device=d
                    )
                )
                df_backup.loc[t + timestep, d] = rescap
        self._dfElBackup = pd.concat([self._dfElBackup, df_backup])

        self._dfDeviceFlow = self._addToDf(
            self._dfDeviceFlow, varDeviceFlow, timelimit, timeshift
        )
        self._dfDeviceIsOn = self._addToDf(
            self._dfDeviceIsOn, varDeviceIsOn, timelimit, timeshift
        )
        self._dfDeviceIsPrep = self._addToDf(
            self._dfDeviceIsPrep, varDeviceIsPrep, timelimit, timeshift
        )
        self._dfDeviceStorageEnergy = self._addToDf(
            self._dfDeviceStorageEnergy, varDeviceStorageEnergy, timelimit, timeshift
        )
        self._dfDeviceStoragePmax = self._addToDf(
            self._dfDeviceStoragePmax, varDeviceStoragePmax, timelimit, timeshift
        )
        self._dfDeviceStarting = self._addToDf(
            self._dfDeviceStarting, varDeviceStarting, timelimit, timeshift
        )
        self._dfDeviceStopping = self._addToDf(
            self._dfDeviceStopping, varDeviceStopping, timelimit, timeshift
        )
        self._dfEdgeFlow = self._addToDf(
            self._dfEdgeFlow, varEdgeFlow, timelimit, timeshift
        )
        self._dfElVoltageAngle = self._addToDf(
            self._dfElVoltageAngle, varElVoltageAngle, timelimit, timeshift
        )
        self._dfTerminalPressure = self._addToDf(
            self._dfTerminalPressure, varPressure, timelimit, timeshift
        )
        self._dfTerminalFlow = self._addToDf(
            self._dfTerminalFlow, varTerminalFlow, timelimit, timeshift
        )
        return