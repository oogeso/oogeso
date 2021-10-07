from dataclasses import dataclass, fields
import logging
import pandas as pd
import pyomo.environ as pyo
from oogeso.dto.oogeso_input_data_objects import EnergySystemData
from .optimiser import Optimiser

# get progress bar:
try:
    from tqdm import trange

    has_tqdm = True
except:
    logging.info("Consider installing tqdm to get progress bar")
    trange = range
    has_tqdm = False


@dataclass
class SimulationResult:
    """Results from oogeso simulation"""

    # Dataframes keeping track of simulation results:
    dfDeviceFlow: pd.DataFrame = None
    dfDeviceIsPrep: pd.DataFrame = None
    dfDeviceIsOn: pd.DataFrame = None
    dfDeviceStorageEnergy: pd.DataFrame = None
    dfDeviceStoragePmax: pd.DataFrame = None
    dfDeviceStarting: pd.DataFrame = None
    dfDeviceStopping: pd.DataFrame = None
    dfPenalty: pd.DataFrame = None
    dfEdgeFlow: pd.DataFrame = None
    dfEdgeLoss: pd.DataFrame = None
    dfElVoltageAngle: pd.DataFrame = None
    dfTerminalPressure: pd.DataFrame = None
    dfTerminalFlow: pd.DataFrame = None
    dfCO2rate: pd.DataFrame = None  # co2 emission sum per timestep
    dfCO2rate_per_dev: pd.DataFrame = None  # co2 emission per device per timestep
    dfExportRevenue: pd.DataFrame = None  # revenue from exported energy
    dfCO2intensity: pd.DataFrame = None
    dfElReserve: pd.DataFrame = None  # Reserve capacity
    dfElBackup: pd.DataFrame = None  # Backup capacity (handling faults)
    dfDuals: pd.DataFrame = None
    # Time-series profiles used in simulatin:
    df_profiles_forecast: pd.DataFrame = None
    df_profiles_nowcast: pd.DataFrame = None
    optimiser: Optimiser = None  # reference to optimiser object

    def append_results(self, sim_res):
        exclude_list = ["optimiser", "df_profiles_forecast", "df_profiles_forecast"]
        for my_field in fields(self):
            field_name = my_field.name
            if field_name not in exclude_list:
                my_df = getattr(self, field_name)
                other_df = getattr(sim_res, field_name)
                if other_df is not None:
                    setattr(self, field_name, pd.concat([my_df, other_df]).sort_index())


class Simulator:
    """Main class for Oogeso energy system simulations"""

    def __init__(self, data: EnergySystemData):
        """Create Simulator object

        Parameters
        ----------
        data : EnergySystemData
            Data object (nodes, edges, devices, profiles)
        """

        # Abstract pyomo model formulation
        self.optimiser = Optimiser(data)

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

    def setOptimiser(self, optimiser):
        self.optimiser = optimiser

    def runSimulation(
        self,
        solver,
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
        write_yaml : boolean
            Whether to save problem to yaml file (for debugging)
        timerange : [int,int]
            Limit to this number of timesteps
        timelimit : int
            Time limit spent on each optimisation (sec)
        store_duals : dict
            Store dual values of constraints. The dictionary contains a
            key:value list where value is a new dictionary specifying the
            constraint and indices. None in the index is replaced by time
            Example:
            store_duals = {'elcost':{constr=constrDevicePmin, indx:('util',None)}
        """

        steps = self.optimiser.optimisation_parameters.optimisation_timesteps
        horizon = self.optimiser.optimisation_parameters.planning_horizon
        if timelimit is not None:
            logging.debug("Using solver timelimit={}".format(timelimit))
        if timerange is None:
            # use the entire timeseries
            time_start = 0
            time_end = self.profiles["forecast"].index.max() + 1 - horizon
        else:
            time_start = timerange[0]
            time_end = timerange[1]

        result_object = SimulationResult(
            optimiser=self.optimiser,
            df_profiles_nowcast=self.profiles["nowcast"],
            df_profiles_forecast=self.profiles["forecast"],
        )
        self.result_object = result_object

        first = True
        for step in trange(time_start, time_end, steps):
            if not has_tqdm:
                # no progress bar
                logging.info("Solving timestep={}".format(step))
            # 1. Update problem formulation
            self.optimiser.updateOptimisationModel(
                step, first=first, profiles=self.profiles
            )
            # 2. Solve for planning horizon
            self.optimiser.solve(
                solver=solver, write_yaml=write_yaml, timelimit=timelimit
            )
            # 3. Save results (for later analysis)
            new_results = self._saveOptimisationResult(step, store_duals)
            result_object.append_results(new_results)
            first = False

        return result_object

    def _saveOptimisationResult(self, timestep, store_duals=None):
        """save results of optimisation for later analysis"""

        # TODO: Implement result storage
        # hdf5? https://stackoverflow.com/questions/47072859/how-to-append-data-to-one-specific-dataset-in-a-hdf5-file-with-h5py)
        pyomo_instance = self.optimiser.pyomo_instance
        timelimit = self.optimiser.optimisation_parameters.optimisation_timesteps
        timeshift = timestep

        # Retrieve variable values as dictionary with dataframes
        res = self.optimiser.extract_all_variable_values(timelimit, timeshift)

        if store_duals is not None:
            # Save dual values
            # store_duals = {
            #   'elcost': {'constr':'constrDevicePmin','indx':('util',None)}
            #   }
            horizon_steps = self.optimiser.optimisation_parameters.planning_horizon
            df_duals = pd.DataFrame(columns=store_duals.keys())
            for key, val in store_duals.items():
                # vrs=('util',None)
                vrs = val["indx"]
                constr = getattr(pyomo_instance, val["constr"])
                logging.info(constr)
                sumduals = 0
                for t in range(timelimit):
                    # Replace None by the timestep, ('util',None) -> ('util',t)
                    vrs1 = tuple(x if x is not None else t for x in vrs)
                    logging.info(vrs1)
                    logging.info(constr[vrs1])
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
                    df_duals.loc[timeshift + t, key] = dual

        # CO2 emission rate per device:
        df_co2_rate_dev = pd.DataFrame()
        for d in pyomo_instance.setDevice:
            for t in range(timelimit):
                co2_dev = self.optimiser.compute_CO2(
                    pyomo_instance, devices=[d], timesteps=[t]
                )
                df_co2_rate_dev.loc[t + timestep, d] = pyo.value(co2_dev)

        # CO2 emission intensity (sum) and emission rate
        df_co2intensity = pd.Series()
        df_co2_rate_sum = pd.Series()
        for t in range(timelimit):
            df_co2intensity.loc[t + timestep] = pyo.value(
                self.optimiser.compute_CO2_intensity(pyomo_instance, timesteps=[t])
            )
            df_co2_rate_sum.loc[t + timestep] = pyo.value(
                self.optimiser.compute_CO2(pyomo_instance, timesteps=[t])
            )

        # Penalty values per device
        df_penalty = pd.DataFrame()
        for d, dev in self.optimiser.all_devices.items():
            for t in range(timelimit):
                this_penalty = dev.compute_penalty([t])
                df_penalty.loc[t + timestep, d] = pyo.value(this_penalty)

        # Revenue from exported energy
        df_exportRevenue = pd.DataFrame()
        for c in pyomo_instance.setCarrier:
            for t in range(timelimit):
                exportRevenue_dev = self.optimiser.compute_exportRevenue(
                    pyomo_instance, carriers=[c], timesteps=[t]
                )
                df_exportRevenue.loc[t + timestep, c] = pyo.value(exportRevenue_dev)

        # Reserve capacity
        df_reserve = pd.DataFrame()
        for t in range(timelimit):
            rescap = pyo.value(
                self.optimiser.all_networks["el"].compute_elReserve(pyomo_instance, t)
            )
            df_reserve.loc[t + timestep, "reserve"] = rescap

        # Backup capacity
        df_backup = pd.DataFrame()
        devs_elout = []
        for dev_id, dev_obj in self.optimiser.all_devices.items():
            if "el" in dev_obj.carrier_out:
                devs_elout.append(dev_obj.id)
        for t in range(timelimit):
            for d in devs_elout:
                rescap = pyo.value(
                    self.optimiser.all_networks["el"].compute_elReserve(
                        pyomo_instance, t, exclude_device=d
                    )
                )
                df_backup.loc[t + timestep, d] = rescap

        result_object = SimulationResult(
            dfDeviceFlow=res["varDeviceFlow"],
            dfDeviceIsPrep=res["varDeviceIsPrep"],
            dfDeviceIsOn=res["varDeviceIsOn"],
            dfDeviceStarting=res["varDeviceStarting"],
            dfDeviceStopping=res["varDeviceStopping"],
            dfDeviceStorageEnergy=res["varDeviceStorageEnergy"],
            dfDeviceStoragePmax=res["varDeviceStoragePmax"],
            dfEdgeFlow=res["varEdgeFlow"],
            dfEdgeLoss=res["varEdgeLoss"],
            dfTerminalFlow=res["varTerminalFlow"],
            dfTerminalPressure=res["varPressure"],
            dfElVoltageAngle=res["varElVoltageAngle"],
            # dfPenalty=res["varDevicePenalty"], # this does not include start/stop penalty
            dfPenalty=df_penalty,
            dfElReserve=df_reserve,
            dfElBackup=df_backup,
            dfExportRevenue=df_exportRevenue,
            dfCO2rate=df_co2_rate_sum,
            dfCO2intensity=df_co2intensity,
            dfCO2rate_per_dev=df_co2_rate_dev,
            dfDuals=None,
            df_profiles_forecast=None,
            df_profiles_nowcast=None,
        )
        return result_object
