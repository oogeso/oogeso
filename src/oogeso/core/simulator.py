import logging
from typing import Tuple, Sequence
from numpy import float64
import pandas as pd
import pyomo.environ as pyo
from oogeso.dto import EnergySystemData
from oogeso.dto import SimulationResult
from .optimiser import OptimisationModel


logger = logging.getLogger(__name__)

# get progress bar:
try:
    from tqdm import trange

    HAS_TQDM = True
except ImportError as err:
    logger.info("Consider installing tqdm to get progress bar")
    trange = range
    HAS_TQDM = False


class Simulator:
    """Main class for Oogeso energy system simulations"""

    def __init__(self, data: EnergySystemData):
        """Create Simulator object

        Parameters
        ----------
        data : EnergySystemData
            Data object holding information about the system (nodes, edges, devices, profiles)
            and parameter settings
        """

        # Optimisation model object:
        self.optimiser = OptimisationModel(data)

        self.result_object = None

        # Storing time-series profiles:
        _df_profiles_forecast = pd.DataFrame()
        _df_profiles_nowcast = pd.DataFrame()
        for prof in data.profiles:
            _df_profiles_forecast[prof.id] = prof.data
            if prof.data_nowcast is not None:
                _df_profiles_nowcast[prof.id] = prof.data_nowcast
        self.profiles = {
            "nowcast": _df_profiles_nowcast,
            "forecast": _df_profiles_forecast,
        }

    #    def setOptimiser(self, optimiser):
    #        self.optimiser = optimiser

    def runSimulation(
        self,
        solver: str,
        timerange: Tuple[int, int] = None,
        timelimit: int = None,
        return_variables: Sequence[str] = None,
        store_duals: dict = None,
        write_yaml: bool = False,
    ) -> SimulationResult:
        """Solve problem over many timesteps

        Parameters
        ----------
        solver : string
            Name of solver ("cbc", "gurobi")
        timerange : [int,int]
            Limit to this number of timesteps
        timelimit : int
            Time limit spent on each optimisation (sec)
        return_variables : list
            List of variables to return. Default (None) returns all variables
        store_duals : dict
            Store dual values of constraints. The dictionary contains a
            key:value list where value is a new dictionary specifying the
            constraint and indices. None in the index is replaced by time
            Example:
            store_duals = {'elcost':{constr=constrDevicePmin, indx:('util',None)}
        write_yaml : boolean
            Whether to save problem to yaml file (for debugging)

        Available return variables are:
            "dfDeviceFlow",
            "dfDeviceIsPrep",
            "dfDeviceIsOn",
            "dfDeviceStarting",
            "dfDeviceStopping",
            "dfDeviceStorageEnergy",
            "dfDeviceStoragePmax",
            "dfEdgeFlow",
            "dfEdgeLoss",
            "dfTerminalFlow",
            "dfTerminalPressure",
            "dfElVoltageAngle",
            "dfPenalty",
            "dfElReserve",
            "dfElBackup",
            "dfExportRevenue",
            "dfCO2rate",
            "dfCO2intensity",
            "dfCO2rate_per_dev",
            "dfDuals",

        """

        steps = self.optimiser.optimisation_parameters.optimisation_timesteps
        horizon = self.optimiser.optimisation_parameters.planning_horizon
        if timelimit is not None:
            logger.debug("Using solver timelimit=%s", timelimit)
        if timerange is None:
            # use the entire timeseries
            time_start = 0
            time_end = self.profiles["forecast"].index.max() + 1 - horizon
        else:
            time_start = timerange[0]
            time_end = timerange[1]

        result_object = SimulationResult(
            df_profiles_nowcast=self.profiles["nowcast"],
            df_profiles_forecast=self.profiles["forecast"],
        )
        self.result_object = result_object

        first = True
        for step in trange(time_start, time_end, steps):
            if not HAS_TQDM:
                # no progress bar
                logger.info("Solving timestep=%s", step)
            # 1. Update problem formulation
            self.optimiser.updateOptimisationModel(
                step, first=first, profiles=self.profiles
            )
            # 2. Solve for planning horizon
            self.optimiser.solve(
                solver=solver, write_yaml=write_yaml, timelimit=timelimit
            )
            # 3. Save results (for later analysis)
            new_results = self._saveOptimisationResult(
                step, return_variables, store_duals
            )
            result_object.append_results(new_results)
            first = False

        return result_object

    def _saveOptimisationResult(
        self, timestep, return_variables, store_duals=None
    ) -> SimulationResult:
        """extract results of optimisation for later analysis"""

        # TODO: Implement result storage
        # hdf5? https://stackoverflow.com/questions/47072859/how-to-append-data-to-one-specific-dataset-in-a-hdf5-file-with-h5py)
        pyomo_instance = self.optimiser.pyomo_instance
        timelimit = self.optimiser.optimisation_parameters.optimisation_timesteps

        all_available_variables = [
            "dfDeviceFlow",
            "dfDeviceIsPrep",
            "dfDeviceIsOn",
            "dfDeviceStarting",
            "dfDeviceStopping",
            "dfDeviceStorageEnergy",
            "dfDeviceStoragePmax",
            "dfEdgeFlow",
            "dfEdgeLoss",
            "dfTerminalFlow",
            "dfTerminalPressure",
            "dfElVoltageAngle",
            "dfPenalty",
            "dfElReserve",
            "dfElBackup",
            "dfExportRevenue",
            "dfCO2rate",
            "dfCO2intensity",
            "dfCO2rate_per_dev",
            "dfDuals",
            # "df_profiles_forecast",
            # "df_profiles_nowcast",
        ]
        if return_variables is None:
            return_variables = all_available_variables
        else:
            logger.debug("Storing only a subset of the data generated.")

        # Retrieve variable values as dictionary with pandas series
        res = self.optimiser.extract_all_variable_values(timelimit, timestep)

        if ("dfDuals" in return_variables) and (store_duals is not None):
            # Save dual values
            # store_duals = {
            #   'elcost': {'constr':'constrDevicePmin','indx':('util',None)}
            #   }
            horizon_steps = self.optimiser.optimisation_parameters.planning_horizon
            df_duals = pd.DataFrame(
                columns=store_duals.keys(), index=range(timestep, timestep + timelimit)
            )
            for key, val in store_duals.items():
                # vrs=('util',None)
                vrs = val["indx"]
                constr = getattr(pyomo_instance, val["constr"])
                logger.info(constr)
                # sumduals = 0
                for t in range(timelimit):
                    # Replace None by the timestep, ('util',None) -> ('util',t)
                    vrs1 = tuple(x if x is not None else t for x in vrs)
                    logger.info(vrs1)
                    logger.info(constr[vrs1])
                    dual = pyomo_instance.dual[constr[vrs1]]
                    # The dual gives the improvement in the objective function
                    # if the constraint is relaxed by one unit.
                    # The units of the dual values are the units of the
                    # objective function divided by the units of the constraint.
                    #
                    # A constraint is for a single timestep, whereas the
                    # objective function averages over all timesteps in the
                    # optimisation horizon. To get the improvement of relaxing
                    # the constraint not just in the single timestep, but in
                    # all timesteps we therefore scale up the dual value
                    dual = dual * horizon_steps
                    df_duals.loc[timestep + t, key] = dual
        else:
            df_duals = None

        # CO2 emission rate per device:
        if "dfCO2rate_per_dev" in return_variables:
            df_co2_rate_dev = pd.DataFrame(
                index=range(timestep, timestep + timelimit),
                columns=pyomo_instance.setDevice,
            )
            for d in pyomo_instance.setDevice:
                for t in range(timelimit):
                    co2_dev = self.optimiser.compute_CO2(
                        pyomo_instance, devices=[d], timesteps=[t]
                    )
                    df_co2_rate_dev.loc[t + timestep, d] = pyo.value(co2_dev)
            # change to multi-index series
            df_co2_rate_dev = df_co2_rate_dev.stack()
            df_co2_rate_dev.index.rename(["time", "device"], inplace=True)
        else:
            df_co2_rate_dev = None

        # CO2 emission rate (sum)
        if "dfCO2rate" in return_variables:
            df_co2_rate_sum = pd.Series(
                dtype=float64, index=range(timestep, timestep + timelimit)
            )
            for t in range(timelimit):
                df_co2_rate_sum.loc[t + timestep] = pyo.value(
                    self.optimiser.compute_CO2(pyomo_instance, timesteps=[t])
                )
            df_co2_rate_sum.index.rename("time", inplace=True)
        else:
            df_co2_rate_sum = None

        # CO2 emission intensity (sum)
        if "dfCO2intensity" in return_variables:
            df_co2intensity = pd.Series(
                dtype=float64, index=range(timestep, timestep + timelimit)
            )
            for t in range(timelimit):
                df_co2intensity.loc[t + timestep] = pyo.value(
                    self.optimiser.compute_CO2_intensity(pyomo_instance, timesteps=[t])
                )
            df_co2intensity.index.rename("time", inplace=True)
        else:
            df_co2_rate_sum = None

        # Penalty values per device
        # df_penalty=res["varDevicePenalty"], # this does not include start/stop penalty
        if "dfPenalty" in return_variables:
            df_penalty = pd.DataFrame(
                dtype=float64,
                index=range(timestep, timestep + timelimit),
                columns=pyomo_instance.setDevice,
            )
            for d, dev in self.optimiser.all_devices.items():
                for t in range(timelimit):
                    this_penalty = dev.compute_penalty(pyomo_instance, [t])
                    df_penalty.loc[t + timestep, d] = pyo.value(this_penalty)
            # change to multi-index series:
            df_penalty = df_penalty.stack()
            df_penalty.index.rename(["time", "device"], inplace=True)
        else:
            df_penalty = None

        # Revenue from exported energy (per carrier)
        if "dfExportRevenue" in return_variables:
            df_exportRevenue = pd.DataFrame(
                dtype=float64,
                index=range(timestep, timestep + timelimit),
                columns=pyomo_instance.setCarrier,
            )
            for c in pyomo_instance.setCarrier:
                for t in range(timelimit):
                    exportRevenue_dev = self.optimiser.compute_exportRevenue(
                        pyomo_instance, carriers=[c], timesteps=[t]
                    )
                    df_exportRevenue.loc[t + timestep, c] = pyo.value(exportRevenue_dev)
            # change to multi-index series:
            df_exportRevenue = df_exportRevenue.stack()
            df_exportRevenue.index.rename(["time", "carrier"], inplace=True)
        else:
            df_exportRevenue = None

        # Reserve capacity
        if "dfElReserve" in return_variables:
            df_reserve = pd.Series(
                dtype=float64, index=range(timestep, timestep + timelimit)
            )
            for t in range(timelimit):
                rescap = pyo.value(
                    self.optimiser.all_networks["el"].compute_elReserve(
                        pyomo_instance, t, self.optimiser.all_devices
                    )
                )
                df_reserve.loc[t + timestep] = rescap
            df_reserve.index.rename("time", inplace=True)
        else:
            df_reserve = None

        # Backup capacity
        if "dfElBackup" in return_variables:
            devs_elout = []
            for dev_obj in self.optimiser.all_devices.values():
                if "el" in dev_obj.carrier_out:
                    devs_elout.append(dev_obj.id)
            df_backup = pd.DataFrame(
                dtype=float64,
                index=range(timestep, timestep + timelimit),
                columns=devs_elout,
            )
            for t in range(timelimit):
                for d in devs_elout:
                    rescap = pyo.value(
                        self.optimiser.all_networks["el"].compute_elReserve(
                            pyomo_instance,
                            t,
                            self.optimiser.all_devices,
                            exclude_device=d,
                        )
                    )
                    df_backup.loc[t + timestep, d] = rescap
            df_backup = df_backup.stack()
            df_backup.index.rename(["time", "device"], inplace=True)
        else:
            df_backup = None

        dfDeviceFlow = (
            res["varDeviceFlow"] if "dfDeviceFlow" in return_variables else None
        )
        dfDeviceIsOn = (
            res["varDeviceIsOn"] if "dfDeviceIsOn" in return_variables else None
        )
        dfDeviceIsPrep = (
            res["varDeviceIsPrep"] if "dfDeviceIsPrep" in return_variables else None
        )
        dfDeviceStarting = (
            res["varDeviceStarting"] if "dfDeviceStarting" in return_variables else None
        )
        dfDeviceStopping = (
            res["varDeviceStopping"] if "dfDeviceStopping" in return_variables else None
        )
        dfDeviceStorageEnergy = (
            res["varDeviceStorageEnergy"]
            if "dfDeviceStorageEnergy" in return_variables
            else None
        )
        dfDeviceStoragePmax = (
            res["varDeviceStoragePmax"]
            if "dfDeviceStoragePmax" in return_variables
            else None
        )
        dfEdgeFlow = res["varEdgeFlow"] if "dfEdgeFlow" in return_variables else None
        dfEdgeLoss = res["varEdgeLoss"] if "dfEdgeLoss" in return_variables else None
        dfTerminalFlow = (
            res["varTerminalFlow"] if "dfTerminalFlow" in return_variables else None
        )
        dfTerminalPressure = (
            res["varPressure"] if "dfTerminalPressure" in return_variables else None
        )
        dfElVoltageAngle = (
            res["varElVoltageAngle"] if "dfElVoltageAngle" in return_variables else None
        )

        result_object = SimulationResult(
            dfDeviceFlow=dfDeviceFlow,
            dfDeviceIsPrep=dfDeviceIsPrep,
            dfDeviceIsOn=dfDeviceIsOn,
            dfDeviceStarting=dfDeviceStarting,
            dfDeviceStopping=dfDeviceStopping,
            dfDeviceStorageEnergy=dfDeviceStorageEnergy,
            dfDeviceStoragePmax=dfDeviceStoragePmax,
            dfEdgeFlow=dfEdgeFlow,
            dfEdgeLoss=dfEdgeLoss,
            dfTerminalFlow=dfTerminalFlow,
            dfTerminalPressure=dfTerminalPressure,
            dfElVoltageAngle=dfElVoltageAngle,
            dfPenalty=df_penalty,
            dfElReserve=df_reserve,
            dfElBackup=df_backup,
            dfExportRevenue=df_exportRevenue,
            dfCO2rate=df_co2_rate_sum,
            dfCO2intensity=df_co2intensity,
            dfCO2rate_per_dev=df_co2_rate_dev,
            dfDuals=df_duals,
            df_profiles_forecast=None,
            df_profiles_nowcast=None,
        )
        return result_object
