import logging
from typing import Optional, Sequence, Tuple

import pandas as pd
import pyomo.environ as pyo
from numpy import float64

from oogeso import dto

from .optimiser import OptimisationModel

logger = logging.getLogger(__name__)

# get progress bar:
try:
    from tqdm import trange

    HAS_TQDM = True
except ImportError:
    logger.debug("Consider installing tqdm to get progress bar")
    trange = range
    HAS_TQDM = False


class Simulator:
    """Main class for Oogeso energy system simulations"""

    def __init__(self, data: dto.EnergySystemData):
        """Create Simulator object

        Parameters
        ----------
        data : EnergySystemData
            Data object holding information about the system (nodes, edges, devices, profiles)
            and parameter settings
        """

        # Optimisation model object:
        self.optimiser = OptimisationModel(data=data)

        self.result_object: Optional[dto.SimulationResult] = None

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

    def run_simulation(
        self,
        solver: str,
        solver_executable: Optional[str] = None,
        solver_options: Optional[dict] = None,
        time_range: Tuple[int, int] = None,
        time_limit: Optional[int] = None,
        return_variables: Optional[Sequence[str]] = None,
        store_duals: Optional[dict] = None,
        write_yaml: bool = False,
    ) -> dto.SimulationResult:
        """Solve problem over many timesteps

        Parameters
        ----------
        solver : string
            Name of solver ("cbc", "gurobi", or others)
        solver_executable : string
            Path to executable
        solver_options : dict
            Solver-specific options passed on to solver (for fine-tuning performance)
        time_range : [int,int]
            Limit to this number of timesteps
        time_limit : int
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
            "device_flow",
            "device_is_prep",
            "device_is_on",
            "device_starting",
            "device_stopping",
            "device_storage_energy",
            "device_storage_pmax",
            "edge_flow",
            "edge_loss",
            "terminal_flow",
            "terminal_pressure",
            "el_voltage_angle",
            "penalty",
            "el_reserve",
            "el_backup",
            "export_revenue",
            "co2_rate",
            "co2_intensity",
            "co2_rate_per_dev",
            "duals",

        """

        steps = self.optimiser.optimisation_parameters.optimisation_timesteps
        horizon = self.optimiser.optimisation_parameters.planning_horizon
        if time_limit is not None:
            logger.debug("Using solver timelimit=%s", time_limit)
        if time_range is None:
            # use the entire timeseries
            time_start = 0
            time_end = self.profiles["forecast"].index.max() + 1 - horizon
        else:
            time_start = time_range[0]
            time_end = time_range[1]

        result_object = dto.SimulationResult(
            profiles_nowcast=self.profiles["nowcast"],
            profiles_forecast=self.profiles["forecast"],
        )
        self.result_object = result_object

        first = True
        for step in trange(time_start, time_end, steps):
            if not HAS_TQDM:
                # no progress bar
                logger.debug("Solving timestep=%s", step)
            # 1. Update problem formulation
            self.optimiser.update_optimisation_model(step, first=first, profiles=self.profiles)
            # 2. Solve for planning horizon
            self.optimiser.solve(
                solver=solver,
                solver_executable=solver_executable,
                solver_options=solver_options,
                write_yaml=write_yaml,
                time_limit=time_limit,
            )
            # 3. Save results (for later analysis)
            new_results = self._save_optimisation_result(step, return_variables, store_duals)
            result_object.append_results(new_results)
            first = False

        return result_object

    def _save_optimisation_result(
        self,
        timestep: int,
        return_variables: Optional[Sequence[str]] = None,
        store_duals: Optional[dict] = None,
    ) -> dto.SimulationResult:
        """extract results of optimisation for later analysis"""

        pyomo_instance = self.optimiser
        timelimit = self.optimiser.optimisation_parameters.optimisation_timesteps

        return_all = False
        if not return_variables:
            return_variables = []
            return_all = True
        else:
            logger.debug("Storing only a subset of the data generated.")

        # Retrieve variable values as dictionary with pandas series
        res = self.optimiser.extract_all_variable_values(timelimit, timestep)

        if (return_all or ("dfDuals" in return_variables)) and (store_duals is not None):
            # Save dual values
            # store_duals = {
            #   'elcost': {'constr':'constrDevicePmin','indx':('util',None)}
            #   }
            horizon_steps = self.optimiser.optimisation_parameters.planning_horizon
            df_duals = pd.DataFrame(columns=store_duals.keys(), index=range(timestep, timestep + timelimit))
            for key, val in store_duals.items():
                # vrs=('util',None)
                vrs = val["indx"]
                constr = getattr(pyomo_instance, val["constr"])
                logger.debug(constr)
                # sumduals = 0
                for t in range(timelimit):
                    # Replace None by the timestep, ('util',None) -> ('util',t)
                    vrs1 = tuple(x if x is not None else t for x in vrs)
                    logger.debug(vrs1)
                    logger.debug(constr[vrs1])
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
        if return_all or "co2_rate_per_dev" in return_variables:
            df_co2_rate_dev = pd.DataFrame(
                index=range(timestep, timestep + timelimit),
                columns=pyomo_instance.setDevice,
            )
            for d in pyomo_instance.setDevice:
                for t in range(timelimit):
                    co2_dev = self.optimiser.compute_CO2(pyomo_instance, devices=[d], timesteps=[t])
                    df_co2_rate_dev.loc[t + timestep, d] = pyo.value(co2_dev)
            # change to multi-index series
            df_co2_rate_dev = df_co2_rate_dev.stack()
            df_co2_rate_dev.index.rename(["time", "device"], inplace=True)
            df_co2_rate_dev = df_co2_rate_dev.reorder_levels(["device", "time"])
        else:
            df_co2_rate_dev = None

        # CO2 emission rate (sum)
        if return_all or "co2_rate" in return_variables:
            df_co2_rate_sum = pd.Series(dtype=float64, index=range(timestep, timestep + timelimit))
            for t in range(timelimit):
                df_co2_rate_sum.loc[t + timestep] = pyo.value(self.optimiser.compute_CO2(pyomo_instance, timesteps=[t]))
            df_co2_rate_sum.index.rename("time", inplace=True)
        else:
            df_co2_rate_sum = None

        # CO2 emission intensity (sum)
        if return_all or "co2_intensity" in return_variables:
            df_co2intensity = pd.Series(dtype=float64, index=range(timestep, timestep + timelimit))
            for t in range(timelimit):
                df_co2intensity.loc[t + timestep] = pyo.value(
                    self.optimiser.compute_CO2_intensity(pyomo_instance, timesteps=[t])
                )
            df_co2intensity.index.rename("time", inplace=True)
        else:
            df_co2intensity = None

        # Penalty values per device
        if return_all or "penalty" in return_variables:
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
            df_penalty = df_penalty.reorder_levels(["device", "time"])
        else:
            df_penalty = None

        # Revenue from exported energy (per carrier)
        if return_all or "export_revenue" in return_variables:
            df_export_revenue = pd.DataFrame(
                dtype=float64,
                index=range(timestep, timestep + timelimit),
                columns=pyomo_instance.setCarrier,
            )
            for c in pyomo_instance.setCarrier:
                for t in range(timelimit):
                    export_revenue_device = self.optimiser.compute_export_revenue(
                        pyomo_instance, carriers=[c], timesteps=[t]
                    )
                    df_export_revenue.loc[t + timestep, c] = pyo.value(export_revenue_device)
            # change to multi-index series:
            df_export_revenue = df_export_revenue.stack()
            df_export_revenue.index.rename(["time", "carrier"], inplace=True)
            df_export_revenue = df_export_revenue.reorder_levels(["carrier", "time"])
        else:
            df_export_revenue = None

        # Reserve capacity
        if return_all or "el_reserve" in return_variables:
            df_reserve = pd.Series(dtype=float64, index=range(timestep, timestep + timelimit))
            for t in range(timelimit):
                rescap = pyo.value(
                    self.optimiser.all_networks["el"].compute_el_reserve(pyomo_instance, t, self.optimiser.all_devices)
                )
                df_reserve.loc[t + timestep] = rescap
            df_reserve.index.rename("time", inplace=True)
        else:
            df_reserve = None

        # Backup capacity
        if return_all or "el_backup" in return_variables:
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
                        self.optimiser.all_networks["el"].compute_el_reserve(
                            pyomo_instance,
                            t,
                            self.optimiser.all_devices,
                            exclude_device=d,
                        )
                    )
                    df_backup.loc[t + timestep, d] = rescap
            df_backup = df_backup.stack()
            df_backup.index.rename(["time", "device"], inplace=True)
            df_backup = df_backup.reorder_levels(["device", "time"])
        else:
            df_backup = None

        df_device_flow = res["varDeviceFlow"] if (return_all or "device_flow" in return_variables) else None
        df_device_is_on = res["varDeviceIsOn"] if (return_all or "device_is_on" in return_variables) else None
        df_device_is_prep = res["varDeviceIsPrep"] if (return_all or "device_is_prep" in return_variables) else None
        df_device_starting = res["varDeviceStarting"] if (return_all or "device_starting" in return_variables) else None
        df_device_stopping = res["varDeviceStopping"] if (return_all or "device_stopping" in return_variables) else None
        df_device_storage_energy = (
            res["varDeviceStorageEnergy"] if (return_all or "device_storage_energy" in return_variables) else None
        )
        df_device_storage_P_max = (
            res["varDeviceStoragePmax"] if (return_all or "device_storage_pmax" in return_variables) else None
        )
        df_edge_flow = res["varEdgeFlow"] if (return_all or "edge_flow" in return_variables) else None
        df_edge_loss = res["varEdgeLoss"] if (return_all or "edge_loss" in return_variables) else None
        df_terminal_flow = res["varTerminalFlow"] if (return_all or "terminal_flow" in return_variables) else None
        df_terminal_pressure = res["varPressure"] if (return_all or "terminal_pressure" in return_variables) else None
        df_el_voltage_angle = (
            res["varElVoltageAngle"] if (return_all or "el_voltage_angle" in return_variables) else None
        )

        result_object = dto.SimulationResult(
            device_flow=df_device_flow,
            device_is_prep=df_device_is_prep,
            device_is_on=df_device_is_on,
            device_starting=df_device_starting,
            device_stopping=df_device_stopping,
            device_storage_energy=df_device_storage_energy,
            device_storage_pmax=df_device_storage_P_max,
            edge_flow=df_edge_flow,
            edge_loss=df_edge_loss,
            terminal_flow=df_terminal_flow,
            terminal_pressure=df_terminal_pressure,
            el_voltage_angle=df_el_voltage_angle,
            penalty=df_penalty,
            el_reserve=df_reserve,
            el_backup=df_backup,
            export_revenue=df_export_revenue,
            co2_rate=df_co2_rate_sum,
            co2_intensity=df_co2intensity,
            co2_rate_per_dev=df_co2_rate_dev,
            duals=df_duals,
            profiles_forecast=None,
            profiles_nowcast=None,
        )
        return result_object
