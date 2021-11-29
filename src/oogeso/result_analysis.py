import pandas as pd

from oogeso.dto import EnergySystemData, SimulationResult


def compute_kpis(
    sim_result: SimulationResult,
    sim_data: EnergySystemData,
    windturbines=None,
):
    """Compute key indicators of simulation results

    sim_result :

    """
    hour_per_year = 8760
    sec_per_year = 3600 * hour_per_year
    kpi = {}
    res = sim_result
    if windturbines is None:
        windturbines = []

    num_sim_timesteps = res.dfCO2rate.shape[0]
    timesteps = res.dfCO2rate.index
    td_min = sim_data.parameters.time_delta_minutes
    kpi["hours_simulated"] = num_sim_timesteps * td_min / 60

    # CO2 emissions
    kpi["kgCO2_per_year"] = res.dfCO2rate.mean() * sec_per_year
    kpi["kgCO2_per_Sm3oe"] = res.dfCO2intensity.mean()

    # hours with reduced load
    #        kpi['reducedload_hours_per_year'] = None

    # hours with load shedding
    #        kpi['loadshed_hours_per_year'] = None

    # fuel consumption
    gasturbines = [
        g.id
        for g in sim_data.devices
        # if isinstance(g, Gasturbine) # why doesn't isinstance work?
        if g.model == "gasturbine"
    ]
    mask_gt = res.dfDeviceFlow.index.get_level_values("device").isin(gasturbines)
    gtflow = res.dfDeviceFlow[mask_gt]
    fuel = gtflow.unstack("carrier")["gas"].unstack("terminal")["in"].unstack().mean(axis=1)
    kpi["gt_fuel_sm3_per_year"] = fuel.sum() * sec_per_year

    # electric power consumption
    el_dem = res.dfDeviceFlow.unstack("carrier")["el"].unstack("terminal")["in"].dropna().unstack().mean(axis=1)
    kpi["elconsumption_mwh_per_year"] = el_dem.sum() * hour_per_year
    kpi["elconsumption_avg_mw"] = el_dem.sum()

    # number of generator starts
    gt_starts = res.dfDeviceStarting.unstack().sum(axis=1)[gasturbines].sum()
    kpi["gt_starts_per_year"] = gt_starts * hour_per_year / kpi["hours_simulated"]

    # number of generator stops
    gt_stops = res.dfDeviceStopping.unstack().sum(axis=1)[gasturbines].sum()
    kpi["gt_stops_per_year"] = gt_stops * hour_per_year / kpi["hours_simulated"]

    # running hours of generators
    gt_ison_tsteps = res.dfDeviceIsOn.unstack().sum(axis=1)[gasturbines].sum()
    gt_ison = gt_ison_tsteps * td_min / 60
    kpi["gt_hoursrunning_per_year"] = gt_ison * hour_per_year / kpi["hours_simulated"]

    # wind power output
    el_sup = res.dfDeviceFlow.unstack("carrier")["el"].unstack("terminal")["out"].dropna().unstack()
    p_wind = el_sup.T[windturbines]
    kpi["wind_output_mwh_per_year"] = p_wind.sum(axis=1).mean() * hour_per_year

    # curtailed wind energy
    p_avail = pd.DataFrame(index=timesteps)
    for devdata in sim_data.devices:
        if devdata in windturbines:
            d = devdata.id
            Pmax = devdata.flow_max
            p_avail[d] = Pmax
            if devdata.profile is not None:
                profile_ref = devdata.profile
                p_avail[d] = Pmax * res.df_profiles_nowcast.loc[timesteps, profile_ref]
    p_curtailed = (p_avail - p_wind).sum(axis=1)
    kpi["wind_curtailed_mwh_per_year"] = p_curtailed.mean() * hour_per_year
    return kpi
