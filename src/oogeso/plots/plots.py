import logging
from typing import Optional
from xml.etree.ElementInclude import include

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyparsing import col

import oogeso
from oogeso.core import optimiser

try:
    import matplotlib.pyplot as plt
    import plotly
    import plotly.express as px
    import plotly.io as pio
    import seaborn as sns
except ImportError:
    raise ImportError("In order to run this plotting module you need to install matplotlib, plotly and seaborn.")

sns.set_style("whitegrid")  # Optional: sns.set_palette("dark")
pio.templates.default = "plotly_white"

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
    carrier="el",
    filename=None,
    reverse_legend=True,
    include_forecasts=False,
    include_on_off=False,
    include_prep=False,
    devs_shareload=None,
    names=None,
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
    df = res.device_flow.unstack(["carrier", "terminal"])[(carrier, "out")].unstack("device")
    if devs_shareload is None:
        # gas turbines:
        devs_shareload = [d for d, d_obj in optimiser.all_devices.items() if d_obj.dev_data.model in ["gasturbine", "dieselgenerator", "dieselheater"]]
    if devs_shareload:  # list is non-empty
        devs_online = (df[devs_shareload] > 0).sum(axis=1)
        devs_sum = df[devs_shareload].sum(axis=1)
        devs_mean = devs_sum / devs_online
        for c in devs_shareload:
            mask = df[c] > 0
            df.loc[mask, c] = devs_mean[mask]
    df.columns.name = "devices"
    df = df[devs]
    if "pv" in devs:
        pv = df.pop("pv")
        df.insert(0,"pv",pv)
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
        colour = plotly.colors.DEFAULT_PLOTLY_COLORS
        k = 0
        row_on_off = 2
        row_prep = 2
        if include_on_off:
            row_prep = 3
        for col in df:
            dev = col
            dev_data = optimiser.all_devices[dev].dev_data
            k = (k + 1) % len(colour)  # repeat colour cycle if necessary
            if names is not None:
                name = names[col]
            else:
                name = col
            fig.add_scatter(
                x=df.index,
                y=df[col],
                line_shape="hv",
                name=name,
                line=dict(color=colour[k]),
                stackgroup="P",
                legendgroup=col,
                row=1,
                col=1#,
                #mode ='none'
            )
            if include_on_off & (dev_data.start_stop is not None):
                fig.add_scatter(
                    x=df2.index,
                    y=df2[col],
                    line_shape="hv",
                    name=name,
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
                    name=name,
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


def plot_device_power_energy(sim_result, optimisation_model: pyo.Model, dev, include_flow=True, filename=None, energy_fill_opacity=None, dev_color_dict=None, names=None, sy=False):
    """Plot power in/out of device and storage level (if any)"""
    res = sim_result
    optimiser = optimisation_model
    dev_data = optimiser.all_devices[dev].dev_data
    device_name = "{}:{}".format(dev, dev_data.name)

    if include_flow or sy:
        sy = True
    else:
        sy = False

    if dev_data.model in ["storagehydrogen", "storagehydrogencompressor"]:
        carrier = "hydrogen"
        flow_title = "Flow (Nm3/s)"
        energy_storage_title = "Energy storage [Nm<sup>3</sup>]"
    else:
        carrier = "el"
        flow_title = "Power (MW)"
        energy_storage_title = "Energy storage [MWh]"
    # Power flow in/out
    df_flow = res.device_flow[dev, carrier].unstack("terminal")
    if res.device_storage_energy is None:
        df_storage_energy = pd.DataFrame()
    else:
        df_storage_energy = res.device_storage_energy.unstack("device")
    if dev in df_storage_energy:
        df_storage_energy = df_storage_energy[dev]
        df_storage_energy.index = df_storage_energy.index + 1
    if names is not None:
        names["in"] = "In"
        names["out"] = "Out"
        for y in df_flow:
            if y in names.keys():
                df_flow.rename(columns={y:names[y]}, inplace=True)

    if plotter == "plotly":
        fig = plotly.subplots.make_subplots(specs=[[{"secondary_y": sy}]])
        if include_flow:
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
            if dev_color_dict is not None:
                fig.add_scatter(
                    x=df_storage_energy.index,
                    y=df_storage_energy,
                    name="Storage",
                    secondary_y=False,
                    fill="tozeroy",
                    line_color = dev_color_dict[dev]
                )  # ,line=dict(dash='dot'))
            else:
                fig.add_scatter(
                    x=df_storage_energy.index,
                    y=df_storage_energy,
                    name="Storage",
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
            fig.update_yaxes(title_text=energy_storage_title, secondary_y=False, side="left")
        fig.update_xaxes(title_text="Timestep")
        if include_flow:
            fig.update_yaxes(title_text=flow_title, secondary_y=True, side="right")

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
    devs_combine=None,
    devs_exclude=None,
    dev_color_dict=None,
    names=None,
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
    devs_combine : list of lists of devices to combine and the name of the combined device
    devs_exclude : list of devices to exclude from the plotting
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
    if devs_combine:
        for i in range(len(devs_combine[0])):
            combine_names = []
            for name in devs_combine[1][i]:
                if name in df_flow_out:
                    combine_names.append(name)
            devs_sum = df_flow_out[combine_names].sum(axis=1)
            df_flow_out[devs_combine[0][i]] = devs_sum
            if dev_color_dict is not None:
                dev_color_dict[devs_combine[0][i]] = dev_color_dict[devs_combine[1][i][0]]
            df_flow_out.drop(columns=combine_names, inplace=True)
    if exclude_zero:
        df_flow_in = df_flow_in.loc[:, df_flow_in.sum() != 0]
        df_flow_out = df_flow_out.loc[:, df_flow_out.sum() != 0]
    if devs_exclude is None:
        devs_exclude = []
    if names is not None:
        for y in df_flow_in:
            if y in names.keys():
                df_flow_in.rename(columns={y:names[y]}, inplace=True)
        for y in df_flow_out:
            if y in names.keys():
                df_flow_out.rename(columns={y:names[y]}, inplace=True)
        for d in devs_exclude:
            if d in names.keys():
                devs_exclude.append(names[d])

    if plotter == "plotly":
        fig = plotly.subplots.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
        for col in df_flow_in:
            if col not in devs_exclude:
                if dev_color_dict is not None:
                    fig.add_scatter(
                        x=df_flow_in.index,
                        y=df_flow_in[col],
                        line_shape="hv",
                        line_width=1,
                        line_color=dev_color_dict[col],
                        name="In: " + col,
                        stackgroup="in",
                        legendgroup=col,
                        row=2,
                        col=1,
                    )
                else:
                    fig.add_scatter(
                        x=df_flow_in.index,
                        y=df_flow_in[col],
                        line_shape="hv",
                        line_width=1,
                        name="In: " + col,
                        stackgroup="in",
                        legendgroup=col,
                        row=2,
                        col=1,
                    )
        for col in df_flow_out:
            if col not in devs_exclude:
                if dev_color_dict is not None:
                    fig.add_scatter(
                        x=df_flow_out.index,
                        y=df_flow_out[col],
                        line_shape="hv",
                        line_width=1,
                        line_color=dev_color_dict[col],
                        name="Out: " + col,
                        stackgroup="out",
                        legendgroup=col,
                        row=1,
                        col=1,
                    )
                else:
                    fig.add_scatter(
                        x=df_flow_out.index,
                        y=df_flow_out[col],
                        line_shape="hv",
                        line_width=1,
                        name="Out: " + col,
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
    if plotter == "plotly":
        fig = plotly.subplots.make_subplots(rows=1, cols=1)
        fig.add_scatter(
            x=sim_result.co2_rate.index,
            y=sim_result.co2_rate.values,
            line_shape="hv",
            line_width=1,
            line_color="black",
            stackgroup="one",
            row=1,
            col=1,
        )
        fig.update_xaxes(title_text="Timestep")
        fig.update_yaxes(title_text="CO<sub>2</sub> emission rate [kg/s]")
        fig.update_layout(height=600)
        return fig
    else:
        plt.figure(figsize=(24, 8))
        plt.title("CO<sub>2</sub> emission rate [kg/s]")
        ax = plt.gca()
        ax.set_ylabel("kg/s")
        ax.set_xlabel("Timestep")
        sim_result.co2_rate.plot()
        if filename is not None:
            plt.savefig(filename, bbox_inches="tight")


def plot_CO2_rate_per_device(
    sim_result,
    optimisation_model,
    filename=None,
    reverse_legend=False,
    devs_shareload=None,
    devs_combine=None,
    devs_exclude=None,
    dev_color_dict=None,
    names=None,
):

    dfco2rate = sim_result.co2_rate_per_dev.unstack("device")
    all_devices = optimisation_model.all_devices
    dfplot = dfco2rate.loc[:, ~(dfco2rate == 0).all()].copy()

    if devs_shareload is None:
        # gas turbines:
        devs_shareload = [d for d, d_obj in all_devices.items() if d_obj.dev_data.model == "gasturbine"]

    if devs_shareload:  # list is non-empty
        devs_shareload = [d for d in devs_shareload if d in dfplot]
        devs_online = (dfplot[devs_shareload] > 0).sum(axis=1)
        devs_sum = dfplot[devs_shareload].sum(axis=1)
        devs_mean = devs_sum / devs_online
        for c in devs_shareload:
            mask = dfplot[c] > 0
            dfplot.loc[mask, c] = devs_mean[mask]

    if devs_combine:
        for i in range(len(devs_combine[0])):
            devs_sum = dfplot[[x for x in devs_combine[1][i] if x in dfplot.columns]].sum(axis=1)
            dfplot[devs_combine[0][i]] = devs_sum
            if dev_color_dict is not None:
                dev_color_dict[devs_combine[0][i]] = dev_color_dict[devs_combine[1][i][0]]
            dfplot.drop(dfplot.filter(devs_combine[1][i]), axis=1, inplace=True)
    if devs_exclude is None:
        devs_exclude = []
    if names is not None:
        for y in dfplot:
            if y in names.keys():
                dfplot.rename(columns={y:names[y]}, inplace=True)
        for d in devs_exclude:
            if d in names.keys():
                devs_exclude.append(names[d])

    if plotter == "plotly":
        fig = plotly.subplots.make_subplots(rows=1, cols=1)
        for col in dfplot:
            if col not in devs_exclude:
                if dev_color_dict is not None:
                    fig.add_scatter(
                        x=dfplot.index,
                        y=dfplot[col],
                        line_shape="hv",
                        line_width=1,
                        line_color=dev_color_dict[col],
                        name=col,
                        stackgroup="one",
                        row=1,
                        col=1,
                    )
                else:
                    fig.add_scatter(
                        x=dfplot.index,
                        y=dfplot[col],
                        line_shape="hv",
                        line_width=1,
                        name=col,
                        stackgroup="one",
                        row=1,
                        col=1,
                    )
        fig.update_xaxes(title_text="Timestep")
        fig.update_yaxes(title_text="CO<sub>2</sub> emission rate [kg/s]")
        if reverse_legend:
            fig.update_layout(legend_traceorder="reversed")
        fig.update_layout(height=600)
    else:
        fig = plt.figure(figsize=(12, 4))
        ax = plt.gca()
        ax.set_ylabel("CO<sub>2</sub> emission rate [kg/s]")
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

def plot_op_cost(sim_result, filename=None):
    if plotter == "plotly":
        fig = plotly.subplots.make_subplots(rows=1, cols=1)
        fig.add_scatter(
            x=sim_result.op_cost.index,
            y=sim_result.op_cost.values,
            line_shape="hv",
            line_width=1,
            stackgroup="one",
            row=1,
            col=1,
        )
        fig.update_xaxes(title_text="Timestep")
        fig.update_yaxes(title_text="Operational cost [NOK/s]")
        fig.update_layout(height=600)
        return fig
    else:
        plt.figure(figsize=(24, 8))
        plt.title("Operational cost [NOK/s]")
        ax = plt.gca()
        ax.set_ylabel("NOK/s")
        ax.set_xlabel("Timestep")
        sim_result.co2_rate.plot()
        if filename is not None:
            plt.savefig(filename, bbox_inches="tight")


def plot_op_cost_per_device(
    sim_result,
    optimisation_model,
    filename=None,
    reverse_legend=False,
    devs_shareload=None,
    devs_combine=None,
    devs_exclude=None,
    dev_color_dict=None,
    names=None,
):

    dfopcost = sim_result.op_cost_per_dev.unstack("device")
    all_devices = optimisation_model.all_devices
    dfplot = dfopcost.loc[:, ~(dfopcost == 0).all()].copy()

    if devs_shareload is None:
        # gas turbines:
        devs_shareload = [d for d, d_obj in all_devices.items() if d_obj.dev_data.model == "gasturbine"]

    if devs_shareload:  # list is non-empty
        devs_shareload = [d for d in devs_shareload if d in dfplot]
        devs_online = (dfplot[devs_shareload] > 0).sum(axis=1)
        devs_sum = dfplot[devs_shareload].sum(axis=1)
        devs_mean = devs_sum / devs_online
        for c in devs_shareload:
            mask = dfplot[c] > 0
            dfplot.loc[mask, c] = devs_mean[mask]

    if devs_combine:
        for i in range(len(devs_combine[0])):
            devs_sum = dfplot[[x for x in devs_combine[1][i] if x in dfplot.columns]].sum(axis=1)
            dfplot.insert(loc=i, column=devs_combine[0][i], value=devs_sum)
            if dev_color_dict is not None:
                dev_color_dict[devs_combine[0][i]] = dev_color_dict[devs_combine[1][i][0]]
            dfplot.drop(dfplot.filter(devs_combine[1][i]), axis=1, inplace=True)
    if devs_exclude is None:
        devs_exclude = []
    if names is not None:
        for y in dfplot:
            if y in names.keys():
                dfplot.rename(columns={y:names[y]}, inplace=True)
        for d in devs_exclude:
            if d in names.keys():
                devs_exclude.append(names[d])

    if plotter == "plotly":
        fig = plotly.subplots.make_subplots(rows=1, cols=1)
        for col in dfplot:
            if col not in devs_exclude:
                if dev_color_dict is not None:
                    fig.add_scatter(
                        x=dfplot.index,
                        y=dfplot[col],
                        line_shape="hv",
                        line_width=1,
                        line_color=dev_color_dict[col],
                        name=col,
                        stackgroup="one",
                        row=1,
                        col=1,
                    )
                else:
                    fig.add_scatter(
                        x=dfplot.index,
                        y=dfplot[col],
                        line_shape="hv",
                        line_width=1,
                        name=col,
                        stackgroup="one",
                        row=1,
                        col=1,
                    )
        fig.update_xaxes(title_text="Timestep")
        fig.update_yaxes(title_text="Operational cost [NOK/s]")
        if reverse_legend:
            fig.update_layout(legend_traceorder="reversed")
        fig.update_layout(height=600)
    else:
        fig = plt.figure(figsize=(12, 4))
        ax = plt.gca()
        ax.set_ylabel("Operational cost [NOK/s]")
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


def plot_all_costs(
    sim_result: oogeso.dto.SimulationResult,
    optimization_model: oogeso.OptimisationModel,
    filename: str = None,
    fixed: bool = True,
    investment: bool = True,
    fuel: bool = True,
    co2: bool = True,
    year_avg: bool = False,
    reverse_legend: bool = True,
    divide_investment_by_lifetime: bool = False,
    annuity_interest_rate: float = None,
    non_inv_mult_factor: float = 1,
    ):
    data = [["All costs"]]
    cols = ["device"]
    color_dict = dict()
    # Timestep length = minutes per timestep * seconds per minute
    # Simulation time = number of timesteps timestep length
    timestep_length = optimization_model.paramTimestepDeltaMinutes.value * 60
    simulation_time = sim_result.op_cost.shape[0] * timestep_length
    inv_sum = 0
    fixed_op_cost_sum = 0
    op_cost_sum = 0
    fuel_cost_sum = 0
    co2_cost_sum = 0
    inv_lifetime = 1
    hours_per_year = 8760
    for dev, dev_obj in optimization_model.all_devices.items():
        if divide_investment_by_lifetime or annuity_interest_rate is not None:
            if dev_obj.dev_data.lifetime is not None:
                inv_lifetime = dev_obj.dev_data.lifetime
        if dev_obj.dev_data.model in ["storageel", "storagehydrogen", "storagehydrogencompressor"]:
            installed_capacity = dev_obj.dev_data.E_max
        elif dev_obj.dev_data.model == "heatpump":
            installed_capacity = dev_obj.dev_data.flow_max*dev_obj.dev_data.eta
        else:
            installed_capacity = dev_obj.dev_data.flow_max
        if dev_obj.dev_data.model == "sourcediesel":
            if fuel and (dev_obj.dev_data.op_cost is not None or dev_obj.dev_data.op_cost_in is not None or dev_obj.dev_data.op_cost_out is not None) and sim_result.op_cost_per_dev is not None:
                if year_avg:
                    fuel_cost_sum += sim_result.op_cost_per_dev[dev].mean() * hours_per_year * timestep_length * non_inv_mult_factor
                else:
                    fuel_cost_sum += sim_result.op_cost_per_dev[dev].sum() * timestep_length * non_inv_mult_factor
        else:
            if dev_obj.dev_data.investment_cost is not None and installed_capacity is not None and annuity_interest_rate is not None:
                inv_sum += dev_obj.dev_data.investment_cost*installed_capacity / ((1 - 1 / ((1 + annuity_interest_rate)**inv_lifetime)) / annuity_interest_rate)
            elif dev_obj.dev_data.investment_cost is not None and installed_capacity is not None:
                inv_sum += dev_obj.dev_data.investment_cost*installed_capacity / inv_lifetime
            if dev_obj.dev_data.fixed_op_cost is not None:
                if year_avg:
                    fixed_op_cost_sum += dev_obj.dev_data.fixed_op_cost * installed_capacity * hours_per_year * timestep_length * non_inv_mult_factor    
                else:
                    fixed_op_cost_sum += dev_obj.dev_data.fixed_op_cost * installed_capacity * simulation_time * non_inv_mult_factor
            if (dev_obj.dev_data.op_cost is not None or dev_obj.dev_data.op_cost_in is not None or dev_obj.dev_data.op_cost_out is not None) and sim_result.op_cost_per_dev is not None:
                if year_avg:
                    op_cost_sum += sim_result.op_cost_per_dev[dev].mean() * hours_per_year * timestep_length * non_inv_mult_factor
                else:
                    op_cost_sum += sim_result.op_cost_per_dev[dev].sum() * timestep_length * non_inv_mult_factor
            if sim_result.co2_rate_per_dev[dev] is not None:
                if year_avg:
                    co2_cost_sum += sim_result.co2_rate_per_dev[dev].mean() * hours_per_year * timestep_length * non_inv_mult_factor
                else:
                    co2_cost_sum += sim_result.co2_rate_per_dev[dev].sum() * timestep_length * non_inv_mult_factor
    if investment:
        if inv_sum > 0:
            data[0].append(inv_sum)
            if divide_investment_by_lifetime:
                cols.append("Investment costs per device lifetime")
                color_dict["Investment costs per device lifetime"] = "#9467bd" # Purple
            elif annuity_interest_rate is not None:
                cols.append("Annualized investment costs")
                color_dict["Annualized investment costs"] = "#9467bd" # Purple
            else:
                cols.append("Investment costs")
                color_dict["Investment costs"] = "#9467bd" # Purple
    if fixed and fixed_op_cost_sum > 0:
        data[0].append(fixed_op_cost_sum)
        cols.append("Fixed O&M costs") #operational and maintenance costs")
        color_dict["Fixed operational and maintenance costs"] = "#8c564b" # Brown
        color_dict["Fixed O&M costs"] = "#8c564b" # Brown
    if op_cost_sum > 0:
        data[0].append(op_cost_sum)
        cols.append("Variable O&M costs") #operational and maintenance costs")
        color_dict["Variable operational and maintenance costs"] = "#636EFA" # Blue
        color_dict["Variable O&M costs"] = "#636EFA" # Blue
    if fuel_cost_sum > 0:
        data[0].append(fuel_cost_sum)
        cols.append("Fuel costs (diesel)")
        color_dict["Fuel costs (diesel)"] = "#808080" # Gray
    if co2 and co2_cost_sum > 0:
        data[0].append(co2_cost_sum)
        cols.append("CO<sub>2</sub> costs")
        color_dict["CO<sub>2</sub> costs"] = "#000000" # Black
    costs = pd.DataFrame(data, columns=cols)
    if plotter == "plotly":
        fig = px.bar(costs, y = cols, x='device', color_discrete_map = color_dict)#, orientation = 'h')
        fig.update_xaxes(title_text="Device")
        fig.update_yaxes(title_text="Total costs [NOK]")
        fig.update_layout(title="Total costs over the simulation", height=600)
        if reverse_legend:
            fig.update_layout(legend_traceorder="reversed")
        return fig
    else:
        raise ValueError(f"Plotter: {plotter} has not been implemented for plot all costs.")


def plot_all_losses(
    sim_result: oogeso.dto.SimulationResult,
    simulator: oogeso.Simulator,
    pv_devs = None,
    wind_devs = None,
    battery_devs = None,
    eboiler_devs = None,
    h2_devs = None,
    heat_dump_devs = None,
    reverse_legend: bool = True,
    ):
    data = [["All losses"]]
    cols = ["device"]
    color_dict = dict()

    hours_per_year = 8760
    curtailed_pv = 0
    curtailed_wind = 0
    battery_loss = 0
    eboiler_loss = 0
    hydrogen_loss = 0
    curtailed_heat = 0

    incl_pv = False
    incl_wind = False
    incl_battery = False
    incl_eboiler = False
    incl_hydrogen = False
    incl_heat_dump = False

    if pv_devs is not None and pv_devs[0] in simulator.optimiser.setDevice:
        incl_pv = True
    if wind_devs is not None and wind_devs[0] in simulator.optimiser.setDevice:
        incl_wind = True
    if battery_devs is not None and battery_devs[0] in simulator.optimiser.setDevice:
        incl_battery = True
    if eboiler_devs is not None and eboiler_devs[0] in simulator.optimiser.setDevice:
        incl_eboiler = True
    if h2_devs is not None and h2_devs[0] in simulator.optimiser.setDevice:
        incl_hydrogen = True
    if heat_dump_devs is not None and heat_dump_devs[0] in simulator.optimiser.setDevice:
        incl_heat_dump = True

    if incl_pv or incl_wind:
        profiles = simulator.profiles["nowcast"]
        unused_power = []
        labels = []
        for d, d_obj in simulator.optimiser.all_devices.items():
            if d_obj.dev_data.model == "sourceel" and d_obj.dev_data.profile is not None:
                id = d_obj.dev_data.id
                labels.append(id)
                unused_power.append(profiles[d_obj.dev_data.profile][:len(simulator.result_object.device_flow[id]["el"]["out"])] * d_obj.dev_data.flow_max - simulator.result_object.device_flow[id]["el"]["out"])
        curtailed_data = pd.DataFrame(np.array(unused_power).T, columns=labels)

    for dev, dev_obj in simulator.optimiser.all_devices.items():
        if incl_pv:
            if dev_obj.dev_data.id in pv_devs:
                curtailed_pv += curtailed_data[dev_obj.dev_data.id].mean()*hours_per_year
        if incl_wind:
            if dev_obj.dev_data.id in wind_devs:
                curtailed_wind += curtailed_data[dev_obj.dev_data.id].mean()*hours_per_year
        if incl_battery:
            if dev_obj.dev_data.id in battery_devs:
                battery_el_in = sim_result.device_flow[dev, "el", "in"].mean()*hours_per_year
                battery_el_out = sim_result.device_flow[dev, "el", "out"].mean()*hours_per_year
                battery_energy_loss = battery_el_in - battery_el_out
                battery_loss += battery_energy_loss
        if incl_eboiler:
            if dev_obj.dev_data.id in eboiler_devs:
                eboiler_el_in = sim_result.device_flow[dev, "el", "in"].mean()*hours_per_year
                eboiler_heat_out = sim_result.device_flow[dev, "heat", "out"].mean()*hours_per_year
                eboiler_energy_loss = eboiler_el_in - eboiler_heat_out
                eboiler_loss += eboiler_energy_loss
        if incl_hydrogen:
            if dev_obj.dev_data.id in h2_devs:
                h2_el_in = sim_result.device_flow[dev, "el", "in"].mean()*hours_per_year
                h2_el_out = sim_result.device_flow[dev, "el", "out"].mean()*hours_per_year
                h2_heat_out = sim_result.device_flow[dev, "heat", "out"].mean()*hours_per_year
                h2_energy_loss = h2_el_in - h2_el_out - h2_heat_out
                hydrogen_loss += h2_energy_loss
        if incl_heat_dump:
            if dev_obj.dev_data.id in heat_dump_devs:
                curtailed_heat += sim_result.device_flow[dev, "heat", "in"].mean()*hours_per_year

    if incl_heat_dump:
        data[0].append(curtailed_heat)
        cols.append("Curtailed heat")
        color_dict["Curtailed heat"] = "#1a601a" # Dark green
    if incl_battery:
        data[0].append(battery_loss)
        cols.append("Round-trip battery loss") #operational and maintenance costs")
        color_dict["Round-trip battery loss"] = "#7B0000" # Dark red
    if incl_pv:
        data[0].append(curtailed_pv)
        cols.append("Curtailed PV")
        color_dict["Curtailed PV"] = "#ff7f0e" # Orange
    if incl_wind:
        data[0].append(curtailed_wind)
        cols.append("Curtailed wind")
        color_dict["Curtailed wind"] = "#2ca02c" # Green
    if incl_eboiler:
        data[0].append(eboiler_loss)
        cols.append("E-boiler loss")
        color_dict["E-boiler loss"] = "#e377c2" # Pink
    if incl_hydrogen:
        data[0].append(hydrogen_loss)
        cols.append("Round-trip H<sub>2</sub> loss")
        color_dict["Round-trip H<sub>2</sub> loss"] = "#9467bd" # Purple

    losses = pd.DataFrame(data, columns=cols)
    if plotter == "plotly":
        fig = px.bar(losses, y = cols, x='device', color_discrete_map = color_dict)#, orientation = 'h')
        fig.update_xaxes(title_text="Device system")
        fig.update_yaxes(title_text="Total energy losses per year [MWH]")
        fig.update_layout(title="Losses over the simulation", height=600)
        if reverse_legend:
            fig.update_layout(legend_traceorder="reversed")
        return fig
    else:
        raise ValueError(f"Plotter: {plotter} has not been implemented for plot all costs.")


def plot_all_costs_per_device(
    sim_result: oogeso.dto.SimulationResult,
    optimization_model: oogeso.OptimisationModel,
    filename: str = None,
    fixed: bool = True,
    investment: bool = True,
    fuel: bool = True,
    include_sum: bool = False,
    year_avg: bool = False,
    reverse_legend: bool = True,
    devs_combine: Optional[list] = None,
    devs_exclude = None,
    names = None,
    divide_investment_by_lifetime: bool = False,
    annuity_interest_rate: float = None,
    non_inv_mult_factor: float = 1,
    ):
    data = []
    cols = ["device"]
    color_dict = dict()
    # Timestep length = minutes per timestep * seconds per minute
    # Simulation time = number of timesteps * timestep length
    timestep_length = optimization_model.paramTimestepDeltaMinutes.value * 60
    simulation_time = sim_result.op_cost.shape[0] * timestep_length
    for dev, dev_obj in optimization_model.all_devices.items():
        inv_sum = 0
        fixed_op_cost_sum = 0
        op_cost_sum = 0
        fuel_cost_sum = 0
        co2_cost_sum = 0
        inv_lifetime = 1
        hours_per_year = 8760
        if investment and (divide_investment_by_lifetime or annuity_interest_rate is not None):
            if dev_obj.dev_data.lifetime is not None:
                inv_lifetime = dev_obj.dev_data.lifetime
        if dev_obj.dev_data.model in ["storageel", "storagehydrogen", "storagehydrogencompressor"]:
            installed_capacity = dev_obj.dev_data.E_max
        elif dev_obj.dev_data.model == "heatpump":
            installed_capacity = dev_obj.dev_data.flow_max*dev_obj.dev_data.eta
        else:
            installed_capacity = dev_obj.dev_data.flow_max
        if dev_obj.dev_data.model == "sourcediesel" and fuel:
            if (dev_obj.dev_data.op_cost is not None or dev_obj.dev_data.op_cost_in is not None or dev_obj.dev_data.op_cost_out is not None) and sim_result.op_cost_per_dev is not None:
                if year_avg:
                    fuel_cost_sum += sim_result.op_cost_per_dev[dev].mean() * hours_per_year * timestep_length * non_inv_mult_factor
                else:
                    fuel_cost_sum += sim_result.op_cost_per_dev[dev].sum() * timestep_length * non_inv_mult_factor
        else:
            if investment:
                if dev_obj.dev_data.investment_cost is not None and installed_capacity is not None and annuity_interest_rate is not None:
                    inv_sum = dev_obj.dev_data.investment_cost*installed_capacity / ((1 - 1 / ((1 + annuity_interest_rate)**inv_lifetime)) / annuity_interest_rate)
                elif dev_obj.dev_data.investment_cost is not None and installed_capacity is not None:
                    inv_sum = dev_obj.dev_data.investment_cost*installed_capacity / inv_lifetime
            if dev_obj.dev_data.fixed_op_cost is not None:
                if year_avg:
                    fixed_op_cost_sum = dev_obj.dev_data.fixed_op_cost * installed_capacity * hours_per_year * timestep_length * non_inv_mult_factor
                else:
                    fixed_op_cost_sum = dev_obj.dev_data.fixed_op_cost * installed_capacity * simulation_time * non_inv_mult_factor
            if (dev_obj.dev_data.op_cost is not None or dev_obj.dev_data.op_cost_in is not None or dev_obj.dev_data.op_cost_out is not None) and sim_result.op_cost_per_dev is not None:
                if year_avg:
                    op_cost_sum = sim_result.op_cost_per_dev[dev].mean() *hours_per_year * timestep_length * non_inv_mult_factor
                else:
                    op_cost_sum = sim_result.op_cost_per_dev[dev].sum() * timestep_length * non_inv_mult_factor
            if sim_result.co2_rate_per_dev[dev] is not None:
                if year_avg:
                    co2_cost_sum += sim_result.co2_rate_per_dev[dev].mean() * hours_per_year * timestep_length * non_inv_mult_factor
                else:
                    co2_cost_sum += sim_result.co2_rate_per_dev[dev].sum() * timestep_length * non_inv_mult_factor
        if investment and fuel and (inv_sum > 0 or fixed_op_cost_sum > 0 or op_cost_sum > 0 or fuel_cost_sum > 0 or co2_cost_sum > 0):
            data.append([dev, inv_sum, fixed_op_cost_sum, op_cost_sum, fuel_cost_sum, co2_cost_sum])
        elif investment and not fuel and (inv_sum > 0 or fixed_op_cost_sum > 0 or op_cost_sum > 0):
            data.append([dev, inv_sum, fixed_op_cost_sum, op_cost_sum])
        elif not investment and fuel and (fixed_op_cost_sum > 0 or op_cost_sum > 0 or fuel_cost_sum > 0 or co2_cost_sum > 0):
            data.append([dev, fixed_op_cost_sum, op_cost_sum, fuel_cost_sum, co2_cost_sum])
        elif not investment and not fuel and (fixed_op_cost_sum > 0 or op_cost_sum > 0):
            data.append([dev, fixed_op_cost_sum, op_cost_sum])
    if investment:
        if divide_investment_by_lifetime:
            cols.append("Investment costs per device lifetime")
            color_dict["Investment costs per device lifetime"] = "#9467bd" # Purple
        elif annuity_interest_rate is not None:
            cols.append("Annualized investment costs")
            color_dict["Annualized investment costs"] = "#9467bd" # Purple
        else:
            cols.append("Investment costs")
            color_dict["Investment costs"] = "#9467bd" # Purple
    if fixed:
        cols.append("Fixed O&M costs") #operational and maintenance costs")
        color_dict["Fixed operational and maintenance costs"] = "#8c564b" # Brown
        color_dict["Fixed O&M costs"] = "#8c564b" # Brown
    cols.append("Variable O&M costs") #operational and maintenance costs")
    color_dict["Variable operational and maintenance costs"] = "#636EFA" # Blue
    color_dict["Variable O&M costs"] = "#636EFA" # Blue
    if fuel:
        cols.append("Fuel costs (diesel)")
        color_dict["Fuel costs (diesel)"] = "#808080" # Gray
        cols.append("CO<sub>2</sub> costs")
        color_dict["CO<sub>2</sub> costs"] = "#000000" # Black
    if devs_combine:
        for dev in devs_combine[0]:
            combine = [dev]
            for i in range(len(data[0])-1):
                combine.append(0)
            data.append(combine)
        for dev in data:
            for i in range(len(devs_combine[1])):
                if dev[0] in devs_combine[1][i]:
                    for j in range(len(data)):
                        if data[j][0] == devs_combine[0][i]:
                            new_data = []
                            for k in range(len(data[0])-1):
                                new_data.append(data[j][k+1] + dev[k+1])
                            data[j][1:] = new_data
    if include_sum:
        totals = ["All devices"]
        for i in range(len(data[0])-1):
            totals.append(0)
        for device in data:
            for cost_id in range(1,len(data[0])):
                totals[cost_id] += device[cost_id]
        data.append(totals)

    costs = pd.DataFrame(data, columns=cols)
    if devs_combine:
        for i in range(len(devs_combine[0])):
            for dev in devs_combine[1][i]:
                costs.drop(costs[costs.device == dev].index, inplace=True)
    if devs_exclude:
        for dev in devs_exclude:
            costs.drop(costs[costs.device == dev].index, inplace=True)
    if names is not None:
        for y in costs['device']:
            if y in names.keys():
                costs['device'].replace({y:names[y]}, inplace=True)

    if plotter == "plotly":
        fig = px.bar(costs, y = cols, x='device', color_discrete_map=color_dict)#, orientation = 'h')
        fig.update_xaxes(title_text="Device")
        fig.update_yaxes(title_text="Total costs [NOK]")
        fig.update_layout(title="Total costs over the simulation", height=600)
        if reverse_legend:
            fig.update_layout(legend_traceorder="reversed")
        return fig
    else:
        raise ValueError(f"Plotter: {plotter} has not been implemented for plot all costs per device.")


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


def plot_unused_profile_power(
    simulator: oogeso.Simulator,
    reverse_legend: bool = True,
    names=None,
    ):
    fig = None
    profiles = simulator.profiles["nowcast"]
    unused_power = []
    labels = []
    for d, d_obj in simulator.optimiser.all_devices.items():
        if d_obj.dev_data.model == "sourceel" and d_obj.dev_data.profile is not None:
            id = d_obj.dev_data.id
            labels.append(id)
            unused_power.append(profiles[d_obj.dev_data.profile][:len(simulator.result_object.device_flow[id]["el"]["out"])] * d_obj.dev_data.flow_max - simulator.result_object.device_flow[id]["el"]["out"])
    data = pd.DataFrame(np.array(unused_power).T, columns=labels)
    if names is not None:
        for y in data:
            if y in names.keys():
                data.rename(columns={y:names[y]}, inplace=True)
    if plotter == "plotly":
        fig = plotly.subplots.make_subplots(rows=1, cols=1)
        for col in data:
            fig.add_scatter(
                x=data.index,
                y=data[col],
                line_shape="hv",
                line_width=1,
                name=col,
                stackgroup="one",
                row=1,
                col=1,
            )
        fig.update_xaxes(title_text="Timestep")
        fig.update_yaxes(title_text="Unused power [MW]")
        if reverse_legend:
            fig.update_layout(legend_traceorder="reversed")
        fig.update_layout(height=600)
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
    carrier="el",
    include_margin=True,
    dynamic_margin=True,
    use_forecast=False,
    include_sum=True,
    devs_combine=None,
    devs_exclude=None,
    dev_color_dict=None,
    names = None,
):
    """Plot unused online capacity by all {carrier} devices"""
    df_devs = pd.DataFrame()
    res = sim_result
    optimiser = optimisation_model
    if carrier == "el":
        timerange = list(res.el_reserve.index)
    elif carrier == "heat":
        timerange = list(res.heat_reserve.index)
    else:
        raise ValueError(f"Carrier: {carrier} has not been implemented for plot reserve.")
    margin_incr = pd.DataFrame(0, index=timerange, columns=["margin"])
    for d, dev_obj in optimiser.all_devices.items():
        dev_data = dev_obj.dev_data
        device_model = dev_data.model
        rf = 1
        if carrier in dev_obj.carrier_out:
            # Generators and storage
            max_value = dev_data.flow_max
            if dev_data.profile is not None:
                ext_profile = dev_data.profile
                if use_forecast or (ext_profile not in res.profiles_nowcast):
                    max_value = max_value * res.profiles_forecast.loc[timerange, ext_profile]
                else:
                    max_value = max_value * res.profiles_nowcast.loc[timerange, ext_profile]
            if dev_data.start_stop is not None and dev_data.start_stop.delay_start_minutes>0:
                is_on = res.device_is_on[d]
                max_value = is_on * max_value
            elif device_model in ["storage_"+carrier]:
                max_value = res.dfDeviceStoragePmax[d] + res.device_flow[d, carrier, "in"]
            if carrier == "el":
                if dev_data.reserve_factor is not None:
                    reserve_factor = dev_data.reserve_factor
                    max_value = max_value * reserve_factor
            elif carrier == "heat":
                if dev_data.reserve_heat_factor is not None:
                    reserve_factor = dev_data.reserve_heat_factor
                if "el" in dev_obj.carrier_out:
                    max_value = max_value * (1 - dev_data.eta) / dev_data.eta * dev_data.eta_heat
                elif "el" in dev_obj.carrier_in:
                    max_value = max_value * dev_data.eta
                else:
                    max_value = max_value * reserve_factor
            if reserve_factor == 0:
                # device does not count towards reserve
                rf = 0
            if dynamic_margin:
                # instead of reducing reserve, increase the margin instead
                # R*0.8-M = R - (M+0.2R) - this shows better in the plot what
                margin_incr["margin"] += rf * max_value - rf * reserve_factor * res.device_flow[d, carrier, "out"]
            cap_avail = rf * max_value
            p_generating = rf * reserve_factor * res.device_flow[d, carrier, "out"]
            reserv = cap_avail - p_generating
            df_devs[d] = reserv
    df_devs.columns.name = "device"
    if devs_combine:
        for i in range(len(devs_combine[0])):
            devs_sum = df_devs[[x for x in devs_combine[1][i] if x in df_devs.columns]].sum(axis=1)
            df_devs.insert(loc=i, column=devs_combine[0][i], value=devs_sum)
            if dev_color_dict is not None:
                dev_color_dict[devs_combine[0][i]] = dev_color_dict[devs_combine[1][i][0]]
            df_devs.drop(df_devs.filter(devs_combine[1][i]), axis=1, inplace=True)
    if devs_exclude:
        for dev in devs_exclude:
            df_devs.drop(df_devs[df_devs.device == dev].index, inplace=True)
    if names is not None:
        for y in df_devs:
            if y in names.keys():
                df_devs.rename(columns={y:names[y]}, inplace=True)
    if plotter == "plotly":
        if dev_color_dict is not None:
            fig = px.area(df_devs, line_shape="hv", color="device", color_discrete_map=dev_color_dict)
        else:
            fig = px.area(df_devs, line_shape="hv")
        if include_sum:
            fig.add_scatter(
                x=df_devs.index,
                y=df_devs.sum(axis=1),
                name="SUM",
                line=dict(dash="dot", color="black", width=1),
                line_shape="hv",
            )
        if include_margin:
            if carrier == "el":
                margin = optimiser.all_networks[carrier].carrier_data.el_reserve_margin
            elif carrier == "heat":
                margin = optimiser.all_networks[carrier].carrier_data.heat_reserve_margin
            if dynamic_margin:
                margin_incr["margin"] = margin_incr["margin"] - margin
            else:
                margin_incr["margin"] = margin
            fig.add_scatter(
                x=margin_incr.index,
                y=margin_incr["margin"],
                name="Margin",
                line=dict(dash="dot", color="red"),
                mode="lines",
            )
        fig.update_xaxes(title_text="Timestep")
        fig.update_yaxes(title_text="Reserve [MW]")
        #    return df_devs,margin_incr
    elif plotter == "matplotlib":
        ax = df_devs.plot.area()
        if include_sum:
            df_devs.sum(axis=1).plot(style=":", color="black", drawstyle="steps-post")
        if include_margin:
            if carrier == "el":
                margin = optimiser.all_networks[carrier].carrier_data.el_reserve_margin
            elif carrier == "heat":
                margin = optimiser.all_networks[carrier].carrier_data.heat_reserve_margin
            # wind contribution (cf compute reserve)
            margin_incr["margin"] = margin_incr["margin"] + margin
            margin_incr[["margin"]].plot(style=":", color="red", ax=ax)

        plt.xlabel("Timestep")
        plt.ylabel("Reserve [MW]")
        fig = plt.gcf()
    else:
        raise ValueError(f"Plotter: {plotter} has not been implemented for plot reserve.")
    return fig


def plot_el_backup(sim_result, filename=None, show_margin=False, return_margin=False, names=None):
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
    if names is not None:
        for y in res_dev:
            if y in names.keys():
                res_dev.rename(columns={y:names[y]}, inplace=True)
        for y in df_device_flow:
            if y in names.keys():
                df_device_flow.rename(columns={y:names[y]}, inplace=True)
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
        fig.update_yaxes(title_text="Power [MW]")
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
