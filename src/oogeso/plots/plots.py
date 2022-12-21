import logging

import numpy as np
import pandas as pd
import pyomo.environ as pyo

try:
    import matplotlib.pyplot as plt
    import plotly
    import plotly.express as px
    import plotly.subplots
    import plotly.io as pio
    import seaborn as sns
except ImportError:
    raise ImportError("In order to run this plotting module you need to install matplotlib, plotly and seaborn.")

sns.set_style("whitegrid")  # Optional: sns.set_palette("dark")

plotter = "plotly"  # matplotlib

logger = logging.getLogger(__name__)


def plot_df(df, id_var, filename=None, title=None, ylabel="value"):
    """Plot dataframe using plotly (saved to file)"""

    df_tidy = df.reset_index()
    df_tidy.rename(columns={0: ylabel}, inplace=True)
    fig = px.line(df_tidy, x="time", y=ylabel, color=id_var, title=title)

    # fig = px.line()
    # for cols in df:
    #    fig.add_scatter(df[cols])
    # fig = px.line(df,x=df.index,y=df.values,title=title)
    # py.iplot('data':[{'x':df.index,'y':df.values,'type':'line'}],
    #        'layout': {'title':'title'},
    #        filename=filename)
    if filename is not None:
        plotly.offline.plot(fig, filename=filename)

    return fig


def plot_device_profile(
    sim_result,
    optimisation_model,
    devs,
    filename=None,
    reverse_legend=True,
    include_forecasts=False,
    include_on_off=False,
    include_prep=False,
    devs_shareload=None,
):
    """plot forecast and actual profile (available power), and device output

    Parameters
    ==========
    sim_result : SimulationResults object
    optimisation_model : OptimisationModel object
    devs : list
        which devices to include
    devs_shareload : list ([]=ignore, None=do it for gas turbines)
        list of devices for which displayed power should be shared evenly
        (typically gas turbines)
        The optimision returns somewhat random distribution of load per device,
        in reality they will share load more or less evenly due to their
        frequency droop settings. Rather than imposing this in the optimisation,
        this is included in the plots. Default: gas turbine
    filename : Filename
    reverse_legend : Use reverse legend
    include_forecasts : Include forecasts
    include_on_off : To include on and off
    include_prep : To include prep

    """
    res = sim_result
    optimiser = optimisation_model
    if type(devs) is not list:
        devs = [devs]
    if include_forecasts & (len(devs) > 1):
        raise ValueError("Can only plot a single device when showing forecasts")
    df = res.device_flow.unstack(["carrier", "terminal"])[("el", "out")].unstack("device")
    if devs_shareload is None:
        # gas turbines:
        devs_shareload = [d for d, d_obj in optimiser.all_devices.items() if d_obj.dev_data.model == "gasturbine"]
    if devs_shareload:  # list is non-empty
        devs_online = (df[devs_shareload] > 0).sum(axis=1)
        devs_sum = df[devs_shareload].sum(axis=1)
        devs_mean = devs_sum / devs_online
        for c in devs_shareload:
            mask = df[c] > 0
            df.loc[mask, c] = devs_mean[mask]
    df.columns.name = "devices"
    df = df[devs]
    nrows = 1
    if include_on_off:
        nrows = nrows + 1
    if include_prep:
        nrows = nrows + 1
    df2 = res.device_is_on.unstack("device")[devs]
    df_prep = res.device_is_prep.unstack("device")[devs]
    timerange = list(res.device_is_on.index.get_level_values("time"))
    if plotter == "plotly":
        fig = plotly.subplots.make_subplots(rows=nrows, cols=1, shared_xaxes=True)
        colour = pio.templates[pio.templates.default].layout.colorway
        k = -1
        row_on_off = 2
        row_prep = 2
        if include_on_off:
            row_prep = 3
        for col in df:
            dev = col
            dev_data = optimiser.all_devices[dev].dev_data
            k = (k + 1) % len(colour)  # repeat colour cycle if necessary
            fig.add_scatter(
                x=df.index,
                y=df[col],
                line_shape="hv",
                name=col,
                line=dict(color=colour[k]),
                stackgroup="P",
                legendgroup=col,
                row=1,
                col=1,
            )
            if include_on_off & (dev_data.start_stop is not None):
                fig.add_scatter(
                    x=df2.index,
                    y=df2[col],
                    line_shape="hv",
                    name=col,
                    line=dict(color=colour[k], dash="dash"),
                    stackgroup="ison",
                    legendgroup=col,
                    row=row_on_off,
                    col=1,
                    showlegend=False,
                )
            if include_prep & (dev_data.start_stop is not None):
                fig.add_scatter(
                    x=df_prep.index,
                    y=df_prep[col],
                    line_shape="hv",
                    name=col,
                    line=dict(color=colour[k], dash="dash"),
                    stackgroup="ison",
                    legendgroup=col,
                    row=row_prep,
                    col=1,
                    showlegend=False,
                )
            if include_forecasts & (dev_data.profile is not None):
                curve = dev_data.profile
                device_P_max = dev_data.flow_max
                if curve in res.profiles_nowcast:
                    fig.add_scatter(
                        x=timerange,
                        y=res.profiles_nowcast.loc[timerange, curve] * device_P_max,
                        line_shape="hv",
                        line=dict(color=colour[k + 1]),
                        name="--nowcast",
                        legendgroup=col,
                        row=1,
                        col=1,
                    )
                fig.add_scatter(
                    x=timerange,
                    y=res.profiles_forecast.loc[timerange, curve] * device_P_max,
                    line_shape="hv",
                    line=dict(color=colour[k + 2]),
                    name="--forecast",
                    legendgroup=col,
                    row=1,
                    col=1,
                )
        fig.update_xaxes(row=1, col=1, title_text="")
        fig.update_xaxes(row=nrows, col=1, title_text="Timestep")
        fig.update_yaxes(row=1, col=1, title_text="Power supply (MW)")
        if include_on_off:
            fig.update_yaxes(row=row_on_off, col=1, title_text="On/off status")
        if include_prep:
            fig.update_yaxes(row=row_prep, col=1, title_text="Startup", nticks=2)
        if reverse_legend:
            fig.update_layout(legend_traceorder="reversed")
        fig.update_layout(height=600)
        # fig.show()
    elif plotter == "matplotlib":
        fig, axs = plt.subplots(nrows=nrows, ncols=1, shared_xaxes=True, figsize=(12, 1 + 3 * nrows))
        ax = axs[0]
        labels = []
        offset_online = 0
        # df.plot(ax=ax)
        for dev in devs:
            dev_data = optimiser.all_devices[dev].dev_data
            devname = "{}:{}".format(dev, dev_data.name)
            device_P_max = dev_data.flow_max
            df[dev].plot(ax=ax)
            # get the color of the last plotted line (the one just plotted)
            col = ax.get_lines()[-1].get_color()
            labels = labels + [devname]
            if include_forecasts & (dev_data.profile is not None):
                curve = dev_data.profile
                (res.profiles_nowcast.loc[timerange, curve] * device_P_max).plot(ax=ax, linestyle="--")
                # ax.set_prop_cycle(None)
                (res.profiles_forecast.loc[timerange, curve] * device_P_max).plot(ax=ax, linestyle=":")
                labels = labels + ["--nowcast", "--forecast"]
            if include_on_off & (dev_data.start_stop is not None):
                offset_online += 0.1
                df2[dev].plot(ax=ax, linestyle="--", color=col)
                labels = labels + ["--online"]
        plt.xlim(df.index.min(), df.index.max())
        ax.legend(labels, loc="lower left", bbox_to_anchor=(1.01, 0), frameon=False)
        if filename is not None:
            plt.savefig(filename, bbox_inches="tight")
    else:
        raise ValueError(f"Plotter: {plotter} has not been implemented for plot device profile.")
    return fig


def plot_device_power_energy(sim_result, optimisation_model: pyo.Model, dev, filename=None, energy_fill_opacity=None):
    """Plot power in/out of device and storage level (if any)"""
    res = sim_result
    optimiser = optimisation_model
    dev_data = optimiser.all_devices[dev].dev_data
    device_name = "{}:{}".format(dev, dev_data.name)

    if dev_data.model == "storagehydrogen":  # Fixme: replace with isinstance class type
        # isinstance(dev_data.model,oogeso.dto.DeviceStorageHydrogenData)
        carrier = "hydrogen"
        flow_title = "Flow (Sm3/s)"
        energy_storage_title = "Energy storage( Sm3)"
    else:
        carrier = "el"
        flow_title = "Power (MW)"
        energy_storage_title = "Energy storage (MWh)"
    # Power flow in/out
    df_flow = res.device_flow[dev, carrier].unstack("terminal")
    if res.device_storage_energy is None:
        df_storage_energy = pd.DataFrame()
    else:
        df_storage_energy = res.device_storage_energy.unstack("device")
    if dev in df_storage_energy:
        df_storage_energy = df_storage_energy[dev]
        df_storage_energy.index = df_storage_energy.index + 1

    if plotter == "plotly":
        fig = plotly.subplots.make_subplots(specs=[[{"secondary_y": True}]])
        for col in df_flow.columns:
            fig.add_scatter(
                x=df_flow.index,
                y=df_flow[col],
                line_shape="hv",
                name=col,
                secondary_y=True,
                fill="tozeroy",
            )
        if not df_storage_energy.empty:
            fig.add_scatter(
                x=df_storage_energy.index,
                y=df_storage_energy,
                name="storage",
                secondary_y=False,
                fill="tozeroy",
            )  # ,line=dict(dash='dot'))
            if energy_fill_opacity is not None:
                k = len(fig["data"]) - 1
                linecol = plotly.colors.DEFAULT_PLOTLY_COLORS[k]
                opacity = energy_fill_opacity
                fillcol = "rgba({}, {})".format(linecol[4:][:-1], opacity)
                fig["data"][k]["fillcolor"] = fillcol
                fig["data"][k]["fill"] = "tozeroy"
            fig.update_yaxes(title_text=energy_storage_title, secondary_y=False, side="right")
        fig.update_xaxes(title_text="Timestep")
        fig.update_yaxes(title_text=flow_title, secondary_y=True, side="left")

    elif plotter == "matplotlib":
        fig = plt.figure(figsize=(12, 4))
        plt.title(device_name)
        ax = plt.gca()
        df_flow.plot(ax=ax, drawstyle="steps-post", marker=".")
        ax.set_xlabel("Timestep")
        ax.set_ylabel(flow_title)
        tmin = df_flow.index.get_level_values("time").min()
        tmax = df_flow.index.get_level_values("time").max() + 1
        ax.set_ylim(0, dev_data.flow_max)
        ax.legend(loc="upper left")  # , bbox_to_anchor =(1.01,0),frameon=False)

        if not df_storage_energy.empty:
            ax2 = ax.twinx()
            ax2.grid(None)
            df_storage_energy.plot(ax=ax2, linestyle=":", color="black")
            ax2.set_ylabel("Energy (MWh)")  # ,color="red")
            if dev_data.model in ["storage_el"]:
                ax2.set_ylim(0, dev_data.max_E)
            elif dev_data.model in ["well_injection"]:
                ax2.set_ylim(-dev_data.max_E / 2, dev_data.max_E / 2)
            # ax2.tick_params(axis='y', labelcolor="red")
            ax2.legend(loc="upper right")
        ax.set_xlim(tmin, tmax)
        if filename is not None:
            plt.savefig(filename, bbox_inches="tight")
    else:
        raise ValueError(f"Plotter: {plotter} has not been implemented for plot device power energy.")
    return fig


def plot_sum_power_mix(
    sim_result,
    optimisation_model,
    carrier,
    filename=None,
    reverse_legend=True,
    exclude_zero=False,
    devs_shareload=None,
):
    """
    Plot power mix

    Parameters
    ==========
    sim_result : SimulationResult object
    optimisation_model : Oogeso optimisation model
    carrier : string
    devs_shareload : list ([]=ignore, None=do it for gas turbines)
        list of devices for which power should be shared evenly (typically gas turbines)
        The optimision returns somewhat random distribution of load per device,
        in reality they will share load more or less evenly due to their
        frequency droop settings. Rather than imposing this in the optimisation,
        this is included in the plots.
    filename : Name of file
    reverse_legend : Use reverse legend
    exclude_zero : To exclude zero values
    """
    # optimiser = simulator.optimiser
    res = sim_result
    optimiser = optimisation_model
    # Power flow in/out
    df_flow = res.device_flow
    t_min = df_flow.index.get_level_values("time").min()
    t_max = df_flow.index.get_level_values("time").max() + 1
    mask_carrier = df_flow.index.get_level_values("carrier") == carrier
    mask_in = df_flow.index.get_level_values("terminal") == "in"
    mask_out = df_flow.index.get_level_values("terminal") == "out"
    df_flow_out = df_flow[mask_carrier & mask_out]
    df_flow_out.index = df_flow_out.index.droplevel(level=("carrier", "terminal"))
    df_flow_out = df_flow_out.unstack(0)
    columns_to_keep = optimiser.get_devices_in_out(carrier_out=carrier)
    df_flow_out = df_flow_out[columns_to_keep]
    df_flow_in = df_flow[mask_carrier & mask_in]
    df_flow_in.index = df_flow_in.index.droplevel(level=("carrier", "terminal"))
    df_flow_in = df_flow_in.unstack(0)
    columns_to_keep = optimiser.get_devices_in_out(carrier_in=carrier)
    df_flow_in = df_flow_in[columns_to_keep]

    if (devs_shareload is None) and (carrier in ["el", "heat"]):
        # gas turbines:
        devs_shareload = [d for d, d_obj in optimiser.all_devices.items() if d_obj.dev_data.model == "gasturbine"]
        logger.debug("Shared load=%s", devs_shareload)
    if devs_shareload:  # list is non-empty
        devs_online = (df_flow_out[devs_shareload] > 0).sum(axis=1)
        devs_sum = df_flow_out[devs_shareload].sum(axis=1)
        devs_mean = devs_sum / devs_online
        for c in devs_shareload:
            mask = df_flow_out[c] > 0
            df_flow_out.loc[mask, c] = devs_mean[mask]

    if exclude_zero:
        df_flow_in = df_flow_in.loc[:, df_flow_in.sum() != 0]
        df_flow_out = df_flow_out.loc[:, df_flow_out.sum() != 0]

    if plotter == "plotly":
        fig = plotly.subplots.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
        for col in df_flow_in:
            fig.add_scatter(
                x=df_flow_in.index,
                y=df_flow_in[col],
                line_shape="hv",
                name="in:" + col,
                stackgroup="in",
                legendgroup=col,
                row=2,
                col=1,
            )
        for col in df_flow_out:
            fig.add_scatter(
                x=df_flow_out.index,
                y=df_flow_out[col],
                line_shape="hv",
                name="out:" + col,
                stackgroup="out",
                legendgroup=col,
                row=1,
                col=1,
            )
        fig.update_xaxes(row=1, col=1, title_text="")
        fig.update_xaxes(row=2, col=1, title_text="Timestep")
        fig.update_yaxes(row=1, col=1, title_text="Power supply (MW)")
        fig.update_yaxes(row=2, col=1, title_text="Power consumption (MW)")
        if reverse_legend:
            fig.update_layout(legend_traceorder="reversed")
        fig.update_layout(height=600)
        # fig.show()
    elif plotter == "matplotlib":
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
        plt.suptitle("Sum power ({})".format(carrier))
        df_flow_out.plot.area(ax=axes[0], linewidth=0)
        df_flow_in.plot.area(ax=axes[1], linewidth=0)
        axes[0].set_ylabel("Power supply (MW)")
        axes[1].set_ylabel("Power consumption (MW)")
        axes[0].set_xlabel("")
        axes[1].set_xlabel("Timestep")
        for ax in axes:
            if reverse_legend:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(
                    handles[::-1],
                    labels[::-1],
                    loc="lower left",
                    bbox_to_anchor=(1.01, 0),
                    frameon=False,
                )
            else:
                ax.legend(loc="lower left", bbox_to_anchor=(1.01, 0), frameon=False)
            ax.set_xlim(t_min, t_max)

        if filename is not None:
            plt.savefig(filename, bbox_inches="tight")
    else:
        raise ValueError(f"Plotter: {plotter} has not been implemented for plot Power Mix.")
    return fig


def plot_export_revenue(sim_result, filename=None, currency="$"):
    export_revenue = sim_result.export_revenue.unstack("carrier")
    if plotter == "plotly":
        dfplot = export_revenue.loc[:, export_revenue.sum() > 0]
        fig = px.area(dfplot)
        fig.update_xaxes(title_text="Timestep")
        fig.update_yaxes(title_text="Revenue ({}/s)".format(currency))
    elif plotter == "matplotlib":
        fig = plt.figure(figsize=(12, 4))
        plt.title("Export revenue ($/s)")
        ax = plt.gca()
        ax.set_ylabel("{}/s".format(currency))
        ax.set_xlabel("Timestep")
        (export_revenue.loc[:, export_revenue.sum() > 0]).plot.area(ax=ax, linewidth=0)
        if filename is not None:
            plt.savefig(filename, bbox_inches="tight")
    else:
        raise ValueError(f"Plotter: {plotter} has not been implemented for plot export revenue.")
    return fig


def plot_CO2_rate(sim_result, filename=None):
    plt.figure(figsize=(12, 4))
    plt.title("CO2 emission rate (kgCO2/s)")
    ax = plt.gca()
    ax.set_ylabel("kgCO2/s")
    ax.set_xlabel("Timestep")
    sim_result.co2_rate.plot()
    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")


def plot_CO2_rate_per_device(
    sim_result,
    optimisation_model,
    filename=None,
    reverse_legend=False,
    device_shareload=None,
):

    dfco2rate = sim_result.co2_rate_per_dev.unstack("device")
    all_devices = optimisation_model.all_devices
    dfplot = dfco2rate.loc[:, ~(dfco2rate == 0).all()].copy()

    if device_shareload is None:
        # gas turbines:
        device_shareload = [d for d, d_obj in all_devices.items() if d_obj.dev_data.model == "gasturbine"]

    if device_shareload:  # list is non-empty
        device_shareload = [d for d in device_shareload if d in dfplot]
        devs_online = (dfplot[device_shareload] > 0).sum(axis=1)
        devs_sum = dfplot[device_shareload].sum(axis=1)
        devs_mean = devs_sum / devs_online
        for c in device_shareload:
            mask = dfplot[c] > 0
            dfplot.loc[mask, c] = devs_mean[mask]

    if plotter == "plotly":
        fig = plotly.subplots.make_subplots(rows=1, cols=1)
        for col in dfplot:
            fig.add_scatter(
                x=dfplot.index,
                y=dfplot[col],
                line_shape="hv",
                name=col,
                stackgroup="one",
                row=1,
                col=1,
            )
        fig.update_xaxes(title_text="Timestep")
        fig.update_yaxes(title_text="Emission rate (kgCO2/s)")
        if reverse_legend:
            fig.update_layout(legend_traceorder="reversed")
        fig.update_layout(height=600)
    else:
        fig = plt.figure(figsize=(12, 4))
        ax = plt.gca()
        ax.set_ylabel("Emission rate (kgCO2/s)")
        ax.set_xlabel("Timestep")
        dfplot.plot.area(ax=ax, linewidth=0)
        if reverse_legend:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(
                handles[::-1],
                labels[::-1],
                loc="lower left",
                bbox_to_anchor=(1.01, 0),
                frameon=False,
            )
        else:
            ax.legend(loc="lower left", bbox_to_anchor=(1.01, 0), frameon=False)

        if filename is not None:
            plt.savefig(filename, bbox_inches="tight")
    return fig


def plot_CO2_intensity(sim_result, filename=None):
    title = "CO2 intensity (kgCO2/Sm3oe)"
    x_label = "Timestep"
    y_label = "CO2 intensity (kgCO2/Sm3oe)"
    df_plot = sim_result.co2_intensity
    if plotter == "plotly":
        fig = px.line(df_plot, x=df_plot.index, y=df_plot.values)  # ,title=title)
        fig.update_xaxes(title_text=x_label)
        fig.update_yaxes(title_text=y_label)
    else:
        fig = plt.figure(figsize=(12, 4))
        plt.title(title)
        ax = plt.gca()
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
        df_plot.plot()
        if filename is not None:
            plt.savefig(filename, bbox_inches="tight")
    return fig


def plot_profiles(profiles, filename=None):
    """Plot profiles (forecast and actual)"""
    fig = None
    if isinstance(profiles, list):
        # list of TimeSeriesData objects
        df = pd.DataFrame()
        for d in profiles:
            df[(d.id, "forecast")] = d.data
            if d.data_nowcast is not None:
                df[(d.id, "nowcast")] = d.data_nowcast
        df.index.name = "timestep"
        df.columns = pd.MultiIndex.from_tuples(df.columns, names=("variable", "type"))
    elif isinstance(profiles, dict):
        df = pd.concat(
            {
                "nowcast": profiles["actual"],
                "forecast": profiles["forecast"],
            }
        )
        df.index.names = ["type", "timestep"]
    else:
        raise Exception("profiles must be input data format or internal format")
    if plotter == "plotly":
        df = df.reset_index()
        df = df.melt(id_vars=["timestep"])
        fig = px.line(
            df,
            x="timestep",
            y="value",
            line_group="type",
            color="variable",
            line_dash="type",
            line_shape="hv",
        )
    elif plotter == "matplotlib":
        plt.figure(figsize=(12, 4))
        ax = plt.gca()
        df.loc[:, df.columns.get_level_values("type") == "forecast"].plot(ax=ax)
        # reset color cycle (so using the same as for the actual plot):
        ax.set_prop_cycle(None)
        df.loc[:, df.columns.get_level_values("type") == "nowcast"].plot(ax=ax, linestyle=":")
        ax.legend(loc="lower left", bbox_to_anchor=(1.01, 0), frameon=False)
        plt.xlabel("Timestep")
        plt.ylabel("Relative value")
        plt.title("Forecast and nowcast profile (forecast=sold line)")
        if filename is not None:
            plt.savefig(filename, bbox_inches="tight")
    return fig


def plot_device_power_flow_pressure(sim_result, optimisation_model: pyo.Model, dev, carriers_inout=None, filename=None):
    res = sim_result
    all_devices = optimisation_model.all_devices
    dev_obj = all_devices[dev]
    dev_data = dev_obj.dev_data
    node = dev_data.node_id
    devname = "{}:{}".format(dev, dev_data.name)
    if carriers_inout is None:
        carriers_inout = {"in": dev_obj.carrier_in, "out": dev_obj.carrier_out}
    if "serial" in carriers_inout:
        del carriers_inout["serial"]

    plt.figure(figsize=(12, 4))
    ax = plt.gca()
    for inout, carriers in carriers_inout.items():
        if inout == "in":
            ls = "--"
        else:
            ls = ":"
        for carr in carriers:
            ax.plot(
                res.device_flow.unstack([0, 1, 2])[(dev, carr, inout)],
                ls,
                label="DeviceFlow ({},{})".format(carr, inout),
            )
    # Pressure
    for inout, carriers in carriers_inout.items():
        for carr in carriers:
            if carr != "el":
                ax.plot(
                    res.terminal_pressure.unstack(0)[node].unstack([0, 1])[(carr, inout)],
                    label="TerminalPressure ({},{})".format(carr, inout),
                )
    plt.title(devname)
    plt.xlabel("Timestep")
    plt.ylabel("Value")
    plt.legend(loc="lower left", bbox_to_anchor=(1.01, 0), frameon=False)
    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")


def plot_gas_turbine_efficiency(
    fuel_A=2.35, fuel_B=0.53, energy_content=40, co2_content=2.34, filename=None, P_max=None
):
    """
    co2content : CO2 content, kgCO2/Sm3gas
    energycontent: energy content, MJ/Sm3gas
    A,B : linear parameters
    """

    x_pow = np.linspace(0, 1, 50)
    y_fuel = fuel_B + fuel_A * x_pow

    nplots = 3 if P_max is None else 4
    plt.figure(figsize=(4 * nplots, 4))

    if P_max is not None:
        y_fuel_sm3 = y_fuel * P_max / energy_content  # Sm3/s
        plt.subplot(1, nplots, 1)
        plt.title("Fuel usage (Sm3/h)")
        plt.xlabel("Electric power output (MW)")
        plt.plot(x_pow * P_max, y_fuel_sm3 * 3600)  # per hour
        plt.ylim(bottom=0)

    plt.subplot(1, nplots, nplots - 2)
    plt.title("Fuel usage ($P_{gas}/P_{el}^{max}$)")
    plt.xlabel("Electric power output ($P_{el}/P_{el}^{max}$)")
    # plt.ylabel("Gas power input ($P_{gas}/P_{el}^{max}$)")
    plt.plot(x_pow, y_fuel)
    plt.ylim(bottom=0)

    plt.subplot(1, nplots, nplots - 1)
    plt.title("Efficiency ($P_{el}/P_{gas}$)")
    plt.xlabel("Electric power output ($P_{el}/P_{el}^{max}$)")
    plt.plot(x_pow, x_pow / y_fuel)

    #    plt.subplot(1,3,3)
    #    plt.title("Specific fuel usage ($P_{gas}/P_{el}$)")
    #    plt.xlabel("Electric power output ($P_{el}/P_{el}^{max}$)")
    #    plt.plot(x_pow,y_fuel/x_pow)
    #    plt.ylim(top=30)

    # 1 MJ = 3600 MWh
    with np.errstate(divide="ignore", invalid="ignore"):
        emissions = 3600 * co2_content / energy_content * y_fuel / x_pow
    plt.subplot(1, nplots, nplots)
    plt.title("Emission intensity (kgCO2/MWh)")
    plt.xlabel("Electric power output ($P_{el}/P_{el}^{max}$)")
    plt.plot(x_pow, emissions)
    plt.ylim(top=2000)

    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")


def plot_reserve(
    sim_result,
    optimisation_model,
    include_margin=True,
    dynamic_margin=True,
    use_forecast=False,
    include_sum=True,
    devs_shareload=None,
):
    """Plot unused online capacity by all el devices
    devs_shareload : list ([]=ignore, None=do it for gas turbines)
        list of devices for which power should be shared evenly (typically gas turbines)
        The optimision returns somewhat random distribution of load per device,
        in reality they will share load more or less evenly due to their
        frequency droop settings. Rather than imposing this in the optimisation,
        this is included in the plots.
    """
    df_devs = pd.DataFrame()
    res = sim_result
    optimiser = optimisation_model
    timerange = list(res.el_reserve.index)
    margin_incr = pd.DataFrame(0, index=timerange, columns=["margin"])
    if devs_shareload is None:
        # split load evently amongst gas turbines:
        devs_shareload = [d for d, d_obj in optimiser.all_devices.items() if d_obj.dev_data.model == "gasturbine"]
        logger.debug("Shared load=%s", devs_shareload)
    if devs_shareload:  # list is non-empty
        devs_online = res.device_is_on[devs_shareload].unstack("device").sum(axis=1)
        devs_sum = res.device_flow.unstack("device").loc[("el", "out")][devs_shareload].sum(axis=1)
        devs_mean = devs_sum / devs_online
    for d, dev_obj in optimiser.all_devices.items():
        dev_data = dev_obj.dev_data
        device_model = dev_data.model
        rf = 1
        if "el" in dev_obj.carrier_out:
            # Generators and storage
            max_value = dev_data.flow_max
            if dev_data.profile is not None:
                ext_profile = dev_data.profile
                if use_forecast or (ext_profile not in res.profiles_nowcast):
                    max_value = max_value * res.profiles_forecast.loc[timerange, ext_profile]
                else:
                    max_value = max_value * res.profiles_nowcast.loc[timerange, ext_profile]
            if dev_data.start_stop is not None:  # device_model in ["gasturbine"]:
                is_on = res.device_is_on[d]
                max_value = is_on * max_value
            elif device_model in ["storage_el"]:
                max_value = res.dfDeviceStoragePmax[d] + res.device_flow[d, "el", "in"]
            if dev_data.reserve_factor is not None:
                reserve_factor = dev_data.reserve_factor
                if reserve_factor == 0:
                    # device does not count towards reserve
                    rf = 0
                if dynamic_margin:
                    # instead of reducing reserve, increase the margin instead
                    # R*0.8-M = R - (M+0.2R) - this shows better in the plot what
                    margin_incr["margin"] += rf * max_value * (1 - reserve_factor)
                else:
                    max_value = max_value * reserve_factor
            cap_avail = rf * max_value
            p_generating = rf * res.device_flow[d, "el", "out"]
            if d in devs_shareload:
                # plot evenly distributed output (devs_mean)
                p_generating = rf * res.device_is_on[d] * devs_mean
            reserv = cap_avail - p_generating
            df_devs[d] = reserv
    df_devs.columns.name = "device"
    if plotter == "plotly":
        fig = px.area(df_devs, line_shape="hv")
        if include_sum:
            fig.add_scatter(
                x=df_devs.index,
                y=df_devs.sum(axis=1),
                name="SUM",
                line=dict(dash="dot", color="black"),
                line_shape="hv",
            )
        if include_margin:
            margin = optimiser.all_networks["el"].carrier_data.el_reserve_margin
            # wind contribution (cf compute reserve)
            margin_incr["margin"] = margin_incr["margin"] + margin
            fig.add_scatter(
                x=margin_incr.index,
                y=margin_incr["margin"],
                name="Margin",
                line=dict(dash="dot", color="red"),
                mode="lines",
            )
        fig.update_xaxes(title_text="Timestep")
        fig.update_yaxes(title_text="Reserve (MW)")
        #    return df_devs,margin_incr
    elif plotter == "matplotlib":
        ax = df_devs.plot.area()
        if include_sum:
            df_devs.sum(axis=1).plot(style=":", color="black", drawstyle="steps-post")
        if include_margin:
            margin = optimiser.all_networks["el"].carrier_data.el_reserve_margin
            # wind contribution (cf compute reserve)
            margin_incr["margin"] = margin_incr["margin"] + margin
            margin_incr[["margin"]].plot(style=":", color="red", ax=ax)

        plt.xlabel("Timestep")
        plt.ylabel("Reserve (MW)")
        fig = plt.gcf()
    else:
        raise ValueError(f"Plotter: {plotter} has not been implemented for plot reserve.")
    return fig


def plot_el_backup(sim_result, filename=None, show_margin=False, return_margin=False):
    """plot reserve capacity vs device power output"""
    res = sim_result
    res_dev = res.el_backup.unstack("device")
    df_device_flow = res.device_flow.copy()
    carrier = "el"
    mask_carrier = df_device_flow.index.get_level_values("carrier") == carrier
    mask_out = df_device_flow.index.get_level_values("terminal") == "out"
    df_device_flow.index = df_device_flow.index.droplevel(level=("carrier", "terminal"))
    df_device_flow = df_device_flow[mask_carrier & mask_out].unstack(0)
    df_device_flow = df_device_flow[res_dev.columns]
    df_margin = (res_dev - df_device_flow).min(axis=1)
    if plotter == "plotly":
        fig = px.line()  # title="Online backup capacity (solid lines) vs device output (dotted lines)")
        colour = plotly.colors.DEFAULT_PLOTLY_COLORS
        k = 0
        for col in res_dev:
            fig.add_scatter(
                x=res_dev.index,
                y=res_dev[col],
                mode="lines",
                legendgroup=col,
                name="{} R_other".format(col),
                line_shape="hv",
                line=dict(color=colour[k]),
            )
            fig.add_scatter(
                x=df_device_flow.index,
                y=df_device_flow[col],
                legendgroup=col,
                name="{} P_out".format(col),
                line_shape="hv",
                line=dict(color=colour[k], dash="dot"),
            )
            k = k + 1
        if show_margin:
            fig.add_scatter(
                x=df_margin.index,
                y=df_margin,
                name="MARGIN",
                line=dict(color="black"),
                line_shape="hv",
            )
        fig.update_xaxes(title_text="Timestep")
        fig.update_yaxes(title_text="Power (MW)")
    elif plotter == "matplotlib":
        fig = plt.figure(figsize=(12, 4))
        ax = plt.gca()
        res_dev.plot(ax=ax, legend=True, alpha=1, linestyle="-")
        labels = list(res_dev.columns)
        if show_margin:
            df_margin.plot(ax=ax, linestyle="-", linewidth=3, color="black", label="MARGIN")
            labels = labels + ["MARGIN"]
        plt.gca().set_prop_cycle(None)
        df_device_flow.plot(ax=ax, linestyle=":", legend=False, alpha=1)
        plt.title("Online backup capacity (solid lines) vs device output (dotted lines)")
        ax.legend(labels, loc="lower left", bbox_to_anchor=(1.01, 0), frameon=False)
        if filename is not None:
            plt.savefig(filename, bbox_inches="tight")
    else:
        raise ValueError(f"Plotter: {plotter} has not been implemented for plot el backup.")
    if return_margin:
        return df_margin

    return fig
