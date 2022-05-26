from lib2to3.pgen2.token import OP
from typing import Optional

import pandas as pd
from pyparsing import Opt

from oogeso import dto


def compute_kpis(
    sim_result: dto.SimulationResult,
    sim_data: dto.EnergySystemData,
    fuel_carrier: Optional[str] = "gas",
    dump_devs: Optional[dto.DevicePowerSinkData] = None,
    pv: Optional[dto.DevicePowerSourceData] = None,
    wind_turbines: Optional[dto.DevicePowerSourceData] = None,
    h2_storage: Optional[dto.DeviceStorageHydrogenData] = None,
):
    """Compute key indicators of simulation results

    sim_result :

    """
    hour_per_year = 8760
    sec_per_year = 3600 * hour_per_year
    kpi = {}
    res = sim_result
    if dump_devs is None:
        dump_devs = list()
    if pv is None:
        pv = list()
    if wind_turbines is None:
        wind_turbines = list()
    if h2_storage is None:
        h2_storage = list()

    num_sim_timesteps = res.co2_rate.shape[0]
    timesteps = res.co2_rate.index
    td_min = sim_data.parameters.time_delta_minutes
    kpi["hours_simulated"] = num_sim_timesteps * td_min / 60

    # CO2 emissions
    kpi["kgCO2_per_year"] = res.co2_rate.mean() * sec_per_year
    kpi["kgCO2_per_Sm3oe"] = res.co2_intensity.mean()

    # fuel consumption
    if fuel_carrier is not None:
        if fuel_carrier == "gas":
            generators = [
                g.id
                for g in sim_data.devices
                # if isinstance(g, Gasturbine) # why doesn't isinstance work?
                if g.model == "gasturbine"
            ]
            heaters = [
                g.id
                for g in sim_data.devices
                if g.model == "gasheater"
            ]
        elif fuel_carrier == "diesel":
            generators = [g.id for g in sim_data.devices if g.model == "dieselgenerator"]
            heaters = [g.id for g in sim_data.devices if g.model == "dieselheater"]

        mask_gt = res.device_flow.index.get_level_values("device").isin(generators)
        gtflow = res.device_flow[mask_gt]
        fuel = gtflow.unstack("carrier")[fuel_carrier].unstack("terminal")["in"].unstack().mean(axis=1)
        kpi["fuel_nm3_per_year"] = fuel.sum() * sec_per_year
    
    else:
        generators = list()
        heaters = list()
        kpi["fuel_nm3_per_year"] = 0


    # electric power consumption
    el_dem = res.device_flow.unstack("carrier")["el"].unstack("terminal")["in"].dropna().unstack().mean(axis=1)
    el_dem = el_dem[~el_dem.index.isin(dump_devs)]
    kpi["elconsumption_mwh_per_year"] = el_dem.sum() * hour_per_year
    kpi["elconsumption_avg_mw"] = el_dem.sum()

    # heat power consumption
    heat_dem = res.device_flow.unstack("carrier")["heat"].unstack("terminal")["in"].dropna().unstack().mean(axis=1)
    heat_dem = heat_dem[~heat_dem.index.isin(dump_devs)]
    kpi["heatconsumption_mwh_per_year"] = heat_dem.sum() * hour_per_year
    kpi["heatconsumption_avg_mw"] = heat_dem.sum()

    # electric dump
    el_dem = res.device_flow.unstack("carrier")["el"].unstack("terminal")["in"].dropna().unstack().mean(axis=1)[dump_devs]
    kpi["eldump_mwh_per_year"] = el_dem.sum() * hour_per_year
    kpi["eldump_avg_mw"] = el_dem.sum()

    # heat dump
    heat_dem = res.device_flow.unstack("carrier")["heat"].unstack("terminal")["in"].dropna().unstack().mean(axis=1)[dump_devs]
    kpi["heatdump_mwh_per_year"] = heat_dem.sum() * hour_per_year
    kpi["heatdump_avg_mw"] = heat_dem.sum()

    # number of generator starts
    gt_starts = res.device_starting.unstack().sum(axis=1)[generators].sum()
    kpi["fossil_generator_starts_per_year"] = gt_starts * hour_per_year / kpi["hours_simulated"]

    # number of generator stops
    gt_stops = res.device_stopping.unstack().sum(axis=1)[generators].sum()
    kpi["fossil_generator_stops_per_year"] = gt_stops * hour_per_year / kpi["hours_simulated"]

    # running hours of generators
    gt_ison_tsteps = res.device_is_on.unstack().sum(axis=1)[generators].sum()
    gt_ison = gt_ison_tsteps * td_min / 60
    kpi["fossil_generator_hoursrunning_per_year"] = gt_ison * hour_per_year / kpi["hours_simulated"]

    # energy output of generators
    kpi["fossil_generator_el_mwh_per_year"] = res.device_flow.unstack("carrier")["el"].unstack("terminal")["out"][generators].dropna().unstack().mean(axis=1).sum()*hour_per_year
    kpi["fossil_generator_heat_mwh_per_year"] = res.device_flow.unstack("carrier")["heat"].unstack("terminal")["out"][generators].dropna().unstack().mean(axis=1).sum()*hour_per_year

    # number of (fossil fuel) heater starts
    heater_starts = res.device_starting.unstack().sum(axis=1)[heaters].sum()
    kpi["fossil_boiler_starts_per_year"] = heater_starts * hour_per_year / kpi["hours_simulated"]

    # number of (fossil fuel) heater stops
    heater_stops = res.device_stopping.unstack().sum(axis=1)[heaters].sum()
    kpi["fossil_boiler_stops_per_year"] = heater_stops * hour_per_year / kpi["hours_simulated"]

    # running hours of (fossil fuel) heaters
    heater_ison_tsteps = res.device_is_on.unstack().sum(axis=1)[heaters].sum()
    heater_ison = heater_ison_tsteps * td_min / 60
    kpi["fossil_boiler_hoursrunning_per_year"] = heater_ison * hour_per_year / kpi["hours_simulated"]

    # energy output of (fossil fuel) heaters
    kpi["fossil_boiler_heat_mwh_per_year"] = res.device_flow.unstack("carrier")["heat"].unstack("terminal")["out"][heaters].dropna().unstack().mean(axis=1).sum()*hour_per_year

    # pv power output
    el_sup = res.device_flow.unstack("carrier")["el"].unstack("terminal")["out"].dropna().unstack()
    p_pv = el_sup.T[pv]
    kpi["pv_output_mwh_per_year"] = p_pv.sum(axis=1).mean() * hour_per_year

    # curtailed pv energy
    p_avail = pd.DataFrame(index=timesteps)
    for device_data in sim_data.devices:
        d = device_data.id
        if d in pv:
            P_max = device_data.flow_max
            p_avail[d] = P_max
            if device_data.profile is not None:
                profile_ref = device_data.profile
                p_avail[d] = P_max * res.profiles_nowcast.loc[timesteps, profile_ref]
    p_curtailed = (p_avail - p_pv).sum(axis=1)
    kpi["pv_curtailed_mwh_per_year"] = p_curtailed.mean() * hour_per_year

    # wind power output
    el_sup = res.device_flow.unstack("carrier")["el"].unstack("terminal")["out"].dropna().unstack()
    p_wind = el_sup.T[wind_turbines]
    kpi["wind_output_mwh_per_year"] = p_wind.sum(axis=1).mean() * hour_per_year

    # curtailed wind energy
    p_avail = pd.DataFrame(index=timesteps)
    for device_data in sim_data.devices:
        d = device_data.id
        if d in wind_turbines:
            P_max = device_data.flow_max
            p_avail[d] = P_max
            if device_data.profile is not None:
                profile_ref = device_data.profile
                p_avail[d] = P_max * res.profiles_nowcast.loc[timesteps, profile_ref]
    p_curtailed = (p_avail - p_wind).sum(axis=1)
    kpi["wind_curtailed_mwh_per_year"] = p_curtailed.mean() * hour_per_year

    # Hydrogen produced, stored, and used
    h2_energy_value = [c.energy_value for c in sim_data.carriers if c.id == "hydrogen"][0]
    h2_prod = res.device_flow.unstack("carrier")["hydrogen"].unstack("terminal")["in"].dropna().unstack()
    h2_prod_stored = h2_prod.T[h2_storage]
    kpi["hydrogen_production_nm3_per_year"] = h2_prod_stored.sum(axis=1).mean() * sec_per_year
    kpi["hydrogen_production_mwh_per_year"] = h2_prod_stored.sum(axis=1).mean() * h2_energy_value * hour_per_year
    kpi["hydrogen_stored_avg_nm3"] = res.device_storage_energy[h2_storage].mean()
    kpi["hydrogen_stored_avg_mwh"] = res.device_storage_energy[h2_storage].mean() * h2_energy_value / 3600
    h2_used = res.device_flow.unstack("carrier")["hydrogen"].unstack("terminal")["out"].dropna().unstack()
    h2_used_stored = h2_used.T[h2_storage]
    kpi["hydrogen_usage_nm3_per_year"] = h2_used_stored.sum(axis=1).mean() * sec_per_year
    kpi["hydrogen_usage_mwh_per_year"] = h2_used_stored.sum(axis=1).mean() * h2_energy_value * hour_per_year


    return kpi
