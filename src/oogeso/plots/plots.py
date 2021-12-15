import logging

import numpy as np
import pandas as pd
import pydot
import pyomo.environ as pyo

try:
    import matplotlib.pyplot as plt
    import plotly
    import plotly.express as px
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
    sim_results : SimulationResults object
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

    """
    res = sim_result
    optimiser = optimisation_model
    if type(devs) is not list:
        devs = [devs]
    if include_forecasts & (len(devs) > 1):
        print("Can only plot one device when showing forecasts")
        return
    df = res.device_flow.unstack(["carrier", "terminal"])[("el", "out")].unstack("device")
    if devs_shareload is None:
        # gas turbines:
        devs_shareload = [d for d, d_obj in optimiser.all_devices.items() if d_obj.dev_data.model == "gasturbine"]
        # devs_shareload = [d for d in mc.instance.setDevice
        #    if mc.instance.paramDevice[d]['model']=='gasturbine']
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
    dfPrep = res.device_is_prep.unstack("device")[devs]
    timerange = list(res.device_is_on.index.get_level_values("time"))
    if plotter == "plotly":
        fig = plotly.subplots.make_subplots(rows=nrows, cols=1, shared_xaxes=True)
        colour = plotly.colors.DEFAULT_PLOTLY_COLORS
        k = 0
        rowOnOff = 2
        rowPrep = 2
        if include_on_off:
            rowPrep = 3
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
                    row=rowOnOff,
                    col=1,
                    showlegend=False,
                )
            if include_prep & (dev_data.start_stop is not None):
                fig.add_scatter(
                    x=dfPrep.index,
                    y=dfPrep[col],
                    line_shape="hv",
                    name=col,
                    line=dict(color=colour[k], dash="dash"),
                    stackgroup="ison",
                    legendgroup=col,
                    row=rowPrep,
                    col=1,
                    showlegend=False,
                )
            if include_forecasts & (dev_data.profile is not None):
                curve = dev_data.profile
                devPmax = dev_data.flow_max
                if curve in res.profiles_nowcast:
                    fig.add_scatter(
                        x=timerange,
                        y=res.profiles_nowcast.loc[timerange, curve] * devPmax,
                        line_shape="hv",
                        line=dict(color=colour[k + 1]),
                        name="--nowcast",
                        legendgroup=col,
                        row=1,
                        col=1,
                    )
                fig.add_scatter(
                    x=timerange,
                    y=res.profiles_forecast.loc[timerange, curve] * devPmax,
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
            fig.update_yaxes(row=rowOnOff, col=1, title_text="On/off status")
        if include_prep:
            fig.update_yaxes(row=rowPrep, col=1, title_text="Startup", nticks=2)
        if reverse_legend:
            fig.update_layout(legend_traceorder="reversed")
        fig.update_layout(height=600)
        # fig.show()
    elif plotter == "matplotlib":
        fig, axs = plt.subplots(nrows=nrows, ncols=1, shared_xaxes=True, figsize=(12, 1 + 3 * nrows))
        #
        #        fig = plt.figure()
        ax = axs[0]
        labels = []
        offset_online = 0
        # df.plot(ax=ax)
        for dev in devs:
            # dev_param = mc.instance.paramDevice[dev]
            dev_data = optimiser.all_devices[dev].dev_data
            devname = "{}:{}".format(dev, dev_data.name)
            devPmax = dev_data.flow_max
            df[dev].plot(ax=ax)
            # get the color of the last plotted line (the one just plotted)
            col = ax.get_lines()[-1].get_color()
            labels = labels + [devname]
            if include_forecasts & (dev_data.profile is not None):
                curve = dev_data.profile
                (res.profiles_nowcast.loc[timerange, curve] * devPmax).plot(ax=ax, linestyle="--")
                # ax.set_prop_cycle(None)
                (res.profiles_forecast.loc[timerange, curve] * devPmax).plot(ax=ax, linestyle=":")
                labels = labels + ["--nowcast", "--forecast"]
            if include_on_off & (dev_data.start_stop is not None):
                # df2=res.dfDeviceIsOn.unstack(0)[dev]+offset_online
                offset_online += 0.1
                df2[dev].plot(ax=ax, linestyle="--", color=col)
                labels = labels + ["--online"]
        plt.xlim(df.index.min(), df.index.max())
        ax.legend(labels, loc="lower left", bbox_to_anchor=(1.01, 0), frameon=False)
        if filename is not None:
            plt.savefig(filename, bbox_inches="tight")
    return fig


def plot_device_power_energy(sim_result, optimisation_model: pyo.Model, dev, filename=None, energy_fill_opacity=None):
    """Plot power in/out of device and storage level (if any)"""
    res = sim_result
    optimiser = optimisation_model
    dev_data = optimiser.all_devices[dev].dev_data
    devname = "{}:{}".format(dev, dev_data.name)

    if dev_data.model == "storage_hydrogen":
        carrier = "hydrogen"
        Ptitle = "Flow (Sm3/s)"
        Etitle = "Energy storage( Sm3)"
    else:
        carrier = "el"
        Ptitle = "Power (MW)"
        Etitle = "Energy storage (MWh)"
    # Power flow in/out
    dfF = res.device_flow[dev, carrier].unstack("terminal")
    dfE = res.device_storage_energy.unstack("device")
    # dfE = pd.DataFrame()
    if dev in dfE:
        dfE = dfE[dev]
        # Shift time by one, since the dfDeviceEnergy[t] is the energy _after_
        # timestep t:
        #        dfE = dfE.unstack().T
        dfE.index = dfE.index + 1
        # dfE = dfE.rename(columns={0: "storage"})
        # dfE.loc[dfE.index.min()-1,dev] = mc.instance.param

    if plotter == "plotly":
        fig = plotly.subplots.make_subplots(specs=[[{"secondary_y": True}]])
        for col in dfF.columns:
            fig.add_scatter(
                x=dfF.index,
                y=dfF[col],
                line_shape="hv",
                name=col,
                secondary_y=True,
                fill="tozeroy",
            )
        if not dfE.empty:
            fig.add_scatter(
                x=dfE.index,
                y=dfE,
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
            fig.update_yaxes(title_text=Etitle, secondary_y=False, side="right")
        fig.update_xaxes(title_text="Timestep")
        fig.update_yaxes(title_text=Ptitle, secondary_y=True, side="left")

    elif plotter == "matplotlib":
        fig = plt.figure(figsize=(12, 4))
        plt.title(devname)
        ax = plt.gca()
        dfF.plot(ax=ax, drawstyle="steps-post", marker=".")
        ax.set_xlabel("Timestep")
        ax.set_ylabel(Ptitle)
        tmin = dfF.index.get_level_values("time").min()
        tmax = dfF.index.get_level_values("time").max() + 1
        ax.set_ylim(0, dev_data.flow_max)
        ax.legend(loc="upper left")  # , bbox_to_anchor =(1.01,0),frameon=False)

        if not dfE.empty:
            ax2 = ax.twinx()
            ax2.grid(None)
            dfE.plot(ax=ax2, linestyle=":", color="black")
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
    """
    # optimiser = simulator.optimiser
    res = sim_result
    optimiser = optimisation_model
    # Power flow in/out
    dfF = res.device_flow
    tmin = dfF.index.get_level_values("time").min()
    tmax = dfF.index.get_level_values("time").max() + 1
    mask_carrier = dfF.index.get_level_values("carrier") == carrier
    mask_in = dfF.index.get_level_values("terminal") == "in"
    mask_out = dfF.index.get_level_values("terminal") == "out"
    dfF_out = dfF[mask_carrier & mask_out]
    dfF_out.index = dfF_out.index.droplevel(level=("carrier", "terminal"))
    dfF_out = dfF_out.unstack(0)
    keepcols = optimiser.getDevicesInout(carrier_out=carrier)
    # logger.info("in: {}".format(keepcols))
    dfF_out = dfF_out[keepcols]
    dfF_in = dfF[mask_carrier & mask_in]
    dfF_in.index = dfF_in.index.droplevel(level=("carrier", "terminal"))
    dfF_in = dfF_in.unstack(0)
    keepcols = optimiser.getDevicesInout(carrier_in=carrier)
    # logger.info("out: {}".format(keepcols))
    dfF_in = dfF_in[keepcols]

    if (devs_shareload is None) and (carrier in ["el", "heat"]):
        # gas turbines:
        devs_shareload = [d for d, d_obj in optimiser.all_devices.items() if d_obj.dev_data.model == "gasturbine"]
        logger.debug("Shared load=%s", devs_shareload)
    if devs_shareload:  # list is non-empty
        devs_online = (dfF_out[devs_shareload] > 0).sum(axis=1)
        devs_sum = dfF_out[devs_shareload].sum(axis=1)
        devs_mean = devs_sum / devs_online
        for c in devs_shareload:
            mask = dfF_out[c] > 0
            dfF_out.loc[mask, c] = devs_mean[mask]

    if exclude_zero:
        dfF_in = dfF_in.loc[:, dfF_in.sum() != 0]
        dfF_out = dfF_out.loc[:, dfF_out.sum() != 0]

    if plotter == "plotly":
        fig = plotly.subplots.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
        for col in dfF_in:
            fig.add_scatter(
                x=dfF_in.index,
                y=dfF_in[col],
                line_shape="hv",
                name="in:" + col,
                stackgroup="in",
                legendgroup=col,
                row=2,
                col=1,
            )
        for col in dfF_out:
            fig.add_scatter(
                x=dfF_out.index,
                y=dfF_out[col],
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
        # plt.figure(figsize=(12,4))
        plt.suptitle("Sum power ({})".format(carrier))
        dfF_out.plot.area(ax=axes[0], linewidth=0)
        dfF_in.plot.area(ax=axes[1], linewidth=0)
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
            ax.set_xlim(tmin, tmax)

        if filename is not None:
            plt.savefig(filename, bbox_inches="tight")
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


def plot_CO2_rate_per_dev(
    sim_result,
    optimisation_model,
    filename=None,
    reverseLegend=False,
    devs_shareload=[],
):

    # df_info = pd.DataFrame.from_dict(dict(mc.instance.paramDevice.items())).T
    # labels = (df_info.index.astype(str))# +'_'+df_info['name'])
    dfco2rate = sim_result.co2_rate_per_dev.unstack("device")
    all_devices = optimisation_model.all_devices
    dfplot = dfco2rate.loc[:, ~(dfco2rate == 0).all()].copy()

    if devs_shareload is None:
        # gas turbines:
        devs_shareload = [d for d, d_obj in all_devices.items() if d_obj.dev_data.model == "gasturbine"]
    #        devs_shareload = [d for d in mc.instance.setDevice
    #            if ((mc.instance.paramDevice[d]['model']=='gasturbine') and
    #                (d in dfplot))]
    if devs_shareload:  # list is non-empty
        devs_shareload = [d for d in devs_shareload if d in dfplot]
        devs_online = (dfplot[devs_shareload] > 0).sum(axis=1)
        devs_sum = dfplot[devs_shareload].sum(axis=1)
        devs_mean = devs_sum / devs_online
        for c in devs_shareload:
            mask = dfplot[c] > 0
            dfplot.loc[mask, c] = devs_mean[mask]

    # dfplot.columns=labels
    if plotter == "plotly":
        fig = plotly.subplots.make_subplots(rows=1, cols=1)
        # ,shared_xaxes=True,vertical_spacing=0.05)
        # fig = px.line(dfplot,x="time",y=ylabel,color=id_var,title=title)
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
        if reverseLegend:
            fig.update_layout(legend_traceorder="reversed")
        fig.update_layout(height=600)
    else:
        fig = plt.figure(figsize=(12, 4))
        ax = plt.gca()
        ax.set_ylabel("Emission rate (kgCO2/s)")
        ax.set_xlabel("Timestep")
        dfplot.plot.area(ax=ax, linewidth=0)
        if reverseLegend:
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
    xlabel = "Timestep"
    ylabel = "CO2 intensity (kgCO2/Sm3oe)"
    dfplot = sim_result.co2_intensity
    if plotter == "plotly":
        fig = plotly.subplots.make_subplots(rows=1, cols=1)
        # ,shared_xaxes=True,vertical_spacing=0.05)
        fig = px.line(dfplot, x=dfplot.index, y=dfplot.values)  # ,title=title)
        fig.update_xaxes(title_text=xlabel)
        fig.update_yaxes(title_text=ylabel)
    else:
        fig = plt.figure(figsize=(12, 4))
        plt.title(title)
        ax = plt.gca()
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        dfplot.plot()
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
        # df = df.stack("type")
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
    # linecycler = itertools.cycle(['-','--',':','-.']*10)
    if carriers_inout is None:
        carriers_inout = {"in": dev_obj.carrier_in, "out": dev_obj.carrier_out}
    if "serial" in carriers_inout:
        del carriers_inout["serial"]

    plt.figure(figsize=(12, 4))
    ax = plt.gca()
    # ax.plot(res.dfDevicePower.unstack(0)[dev],'-.',label="DevicePower")
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


def plot_network(
    simulator,
    timestep=None,
    filename=None,
    prog="dot",
    only_carrier=None,
    rankdir="LR",
    plotDevName=False,
    numberformat="{:.2g}",
    hide_losses=False,
    hide_edgelabel=False,
    **kwargs
):
    """Plot energy network

    simulator : Simulation object
    timestep : int
        which timestep to show values for
    filename : string
        Name of file
    only_carrier : str or list
        Restrict energy carriers to these types (None=plot all)
    rankdir : str
        Plotting direction TB=top to bottom, LR=left to right
    numberformat : str
        specify how numbers should be represented in plot
    hide_losses : bool
        Don't show losses on edges (if any)
    hide_edgelabel : bool
        Don't show any labels on edges
    **kwargs :
        Additional arguments passed on to pydot.Dot(...)
    """

    # Idea: in general, there are "in" and "out" terminals. If there are
    # no serial devices, then these are merged into a single terminal
    # (prettier plot"). Whether the single terminal is shown as an in or out
    # terminal (left or irght), depends on whether it is an input or output
    # of a majority of the connected devices.

    optimiser = simulator.optimiser
    model = optimiser
    res = simulator.result_object

    cluster = {}
    col = {
        "t": {
            "el": "red",
            "gas": "orange",
            "heat": "darkgreen",
            "wellstream": "brown",
            "oil": "black",
            "water": "blue4",
            "hydrogen": "deepskyblue2",
        },
        "e": {
            "el": "red",
            "gas": "orange",
            "heat": "darkgreen",
            "wellstream": "brown",
            "oil": "black",
            "water": "blue4",
            "hydrogen": "deepskyblue2",
        },
        "d": "white",
        "cluster": "lightgray",
    }
    # dotG = pydot.Dot(graph_type='digraph') #rankdir='LR',newrank='false')
    dotG = pydot.Dot(graph_type="digraph", rankdir=rankdir, **kwargs)
    if only_carrier is None:
        carriers = model.setCarrier
    elif type(only_carrier) is str:
        carriers = [only_carrier]
    else:
        carriers = only_carrier

    # devicemodels = milp_compute.devicemodel_inout()

    # plot all node and terminals:
    for n_id, node_obj in optimiser.all_nodes.items():
        cluster = pydot.Cluster(graph_name=n_id, label=n_id, style="filled", color=col["cluster"])
        terms_in = pydot.Subgraph(rank="min")
        gr_devices = pydot.Subgraph(rank="same")
        terms_out = pydot.Subgraph(rank="max")
        for carrier in carriers:
            # add only terminals that are connected to something (device or edge)
            if node_obj.is_non_trivial(carrier):
                devs = node_obj.devices
                num_in = 0
                num_out = 0
                for d, dev_obj in devs.items():
                    dev_model = dev_obj.dev_data.model
                    devlabel = d  # use index as label
                    devlabel = "{}\n{}".format(d, dev_model)
                    if plotDevName:
                        dev_name = dev_obj.dev_data.name
                        devlabel = "{} {}".format(devlabel, dev_name)
                    carriers_in = dev_obj.carrier_in
                    carriers_out = dev_obj.carrier_out
                    carriers_in_lim = list(set(carriers_in) & set(carriers))
                    carriers_out_lim = list(set(carriers_out) & set(carriers))
                    if (carriers_in_lim != []) or (carriers_out_lim != []):
                        gr_devices.add_node(pydot.Node(d, color=col["d"], style="filled", label=devlabel))
                    if carrier in carriers_in_lim:
                        num_in += 1
                        if timestep is None:
                            devedgelabel = ""
                        else:
                            f_in = res.device_flow[(d, carrier, "in", timestep)]
                            devedgelabel = numberformat.format(f_in)
                        if carrier in node_obj.devices_serial:
                            n_in = n_id + "_" + carrier + "_in"
                        else:
                            n_in = n_id + "_" + carrier
                        dotG.add_edge(
                            pydot.Edge(
                                dst=d,
                                src=n_in,
                                color=col["e"][carrier],
                                fontcolor=col["e"][carrier],
                                label=devedgelabel,
                            )
                        )
                    if carrier in carriers_out_lim:
                        num_out += 1
                        if timestep is None:
                            devedgelabel = ""
                        else:
                            f_out = res.device_flow[(d, carrier, "out", timestep)]
                            devedgelabel = numberformat.format(f_out)
                        if carrier in node_obj.devices_serial:
                            n_out = n_id + "_" + carrier + "_out"
                        else:
                            n_out = n_id + "_" + carrier
                        dotG.add_edge(
                            pydot.Edge(
                                dst=n_out,
                                src=d,
                                color=col["e"][carrier],
                                fontcolor=col["e"][carrier],
                                label=devedgelabel,
                            )
                        )

                # add in/out terminals
                supp = ""
                if carrier in node_obj.devices_serial:
                    supp = "_out"  # noqa
                label_in = carrier  # + "_in "
                label_out = carrier  # + supp + " "
                if timestep is None:
                    pass
                elif carrier in ["gas", "wellstream", "oil", "water"]:
                    label_in += numberformat.format(res.terminal_pressure[(n_id, carrier, "in", timestep)])
                    label_out += numberformat.format(res.terminal_pressure[(n_id, carrier, "out", timestep)])
                elif carrier == "el":
                    if optimiser.all_networks["el"].carrier_data.powerflow_method == "dc-pf":
                        label_in += numberformat.format(res.el_voltage_angle[(n_id, timestep)])
                        label_out += numberformat.format(res.el_voltage_angle[(n_id, timestep)])
                # Add two terminals if there are serial devices, otherwise one:
                if carrier in node_obj.devices_serial:
                    terms_in.add_node(
                        pydot.Node(
                            name=n_id + "_" + carrier + "_in",
                            color=col["t"][carrier],
                            label=label_in,
                            shape="box",
                        )
                    )
                    terms_out.add_node(
                        pydot.Node(
                            name=n_id + "_" + carrier + "_out",
                            color=col["t"][carrier],
                            label=label_out,
                            shape="box",
                        )
                    )
                else:
                    # TODO: make this in or out depending on connected devices
                    if num_out > num_in:
                        terms_out.add_node(
                            pydot.Node(
                                name=n_id + "_" + carrier,
                                color=col["t"][carrier],
                                label=label_out,
                                shape="box",
                            )
                        )
                    else:
                        terms_in.add_node(
                            pydot.Node(
                                name=n_id + "_" + carrier,
                                color=col["t"][carrier],
                                label=label_out,
                                shape="box",
                            )
                        )

        cluster.add_subgraph(terms_in)
        cluster.add_subgraph(gr_devices)
        cluster.add_subgraph(terms_out)
        dotG.add_subgraph(cluster)

    # plot all edges (per carrier):
    for carrier in carriers:
        for i, edge_obj in optimiser.all_edges.items():
            edge_data = edge_obj.edge_data
            if edge_data.carrier == carrier:
                headlabel = ""
                taillabel = ""
                if hide_edgelabel:
                    edgelabel = ""
                elif timestep is None:
                    edgelabel = ""
                    if (not hide_edgelabel) and hasattr(edge_data, "pressure_from"):
                        edgelabel = "{} {}-{}".format(
                            edgelabel,
                            edge_data.pressure_from,
                            edge_data.pressure_to,
                        )
                        # edgelabel = "{}-{}".format(edgelabel, edge_data.pressure["to"])
                else:
                    edgelabel = numberformat.format(res.dfEdgeFlow[(i, timestep)])
                    # Add loss
                    if (not hide_losses) and (res.edge_loss is not None) and ((i, timestep) in res.edge_loss):
                        # taillabel = " " + edgelabel
                        losslabel = numberformat.format(res.edge_loss[(i, timestep)])
                        edgelabel = "{} [{}]".format(edgelabel, losslabel)
                n_from = edge_data.node_from
                n_to = edge_data.node_to
                n_from_obj = optimiser.all_nodes[n_from]
                n_to_obj = optimiser.all_nodes[n_to]
                # name of terminal depends on whether it serial or single
                if carrier in n_from_obj.devices_serial:
                    t_out = n_from + "_" + carrier + "_out"
                else:
                    t_out = n_from + "_" + carrier
                if carrier in n_to_obj.devices_serial:
                    t_in = n_to + "_" + carrier + "_in"
                else:
                    t_in = n_to + "_" + carrier

                dotG.add_edge(
                    pydot.Edge(
                        src=t_out,
                        dst=t_in,
                        color='"{0}:invis:{0}"'.format(col["e"][carrier]),
                        fontcolor=col["e"][carrier],
                        label=edgelabel,
                        taillabel=taillabel,
                        headlabel=headlabel,
                    )
                )

    if filename is not None:
        # prog='dot' gives the best layout.
        dotG.write_png(filename, prog=prog)
    return dotG


def plot_gas_turbine_efficiency(fuelA=2.35, fuelB=0.53, energycontent=40, co2content=2.34, filename=None, Pmax=None):
    """
    co2content : CO2 content, kgCO2/Sm3gas
    energycontent: energy content, MJ/Sm3gas
    A,B : linear parameters
    """

    x_pow = np.linspace(0, 1, 50)
    y_fuel = fuelB + fuelA * x_pow
    # Pgas = Qgas*energycontent: Qgas=Pgas/Pmax * Pmax/energycontent

    nplots = 3 if Pmax is None else 4
    # if Pmax is not None
    plt.figure(figsize=(4 * nplots, 4))
    # plt.suptitle("Gas turbine fuel characteristics")

    if Pmax is not None:
        y_fuel_sm3 = y_fuel * Pmax / energycontent  # Sm3/s
        plt.subplot(1, nplots, 1)
        plt.title("Fuel usage (Sm3/h)")
        plt.xlabel("Electric power output (MW)")
        plt.plot(x_pow * Pmax, y_fuel_sm3 * 3600)  # per hour
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
        emissions = 3600 * co2content / energycontent * y_fuel / x_pow
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
    includeMargin=True,
    dynamicMargin=True,
    useForecast=False,
    includeSum=True,
):
    """Plot unused online capacity by all el devices"""
    df_devs = pd.DataFrame()
    res = sim_result
    optimiser = optimisation_model
    timerange = list(res.el_reserve.index)
    marginIncr = pd.DataFrame(0, index=timerange, columns=["margin"])
    for d, dev_obj in optimiser.all_devices.items():
        dev_data = dev_obj.dev_data
        devmodel = dev_data.model
        rf = 1
        if "el" in dev_obj.carrier_out:
            # Generators and storage
            maxValue = dev_data.flow_max
            if dev_data.profile is not None:
                extprofile = dev_data.profile
                if useForecast or (extprofile not in res.profiles_nowcast):
                    maxValue = maxValue * res.profiles_forecast.loc[timerange, extprofile]
                else:
                    maxValue = maxValue * res.profiles_nowcast.loc[timerange, extprofile]
            if dev_data.start_stop is not None:  # devmodel in ["gasturbine"]:
                ison = res.device_is_on[d]
                maxValue = ison * maxValue
            elif devmodel in ["storage_el"]:
                maxValue = res.dfDeviceStoragePmax[d] + res.device_flow[d, "el", "in"]
            if dev_data.reserve_factor is not None:
                reserve_factor = dev_data.reserve_factor
                if reserve_factor == 0:
                    # device does not count towards reserve
                    rf = 0
                if dynamicMargin:
                    # instead of reducing reserve, increase the margin instead
                    # R*0.8-M = R - (M+0.2R) - this shows better in the plot what
                    marginIncr["margin"] += rf * maxValue * (1 - reserve_factor)
                else:
                    maxValue = maxValue * reserve_factor
            cap_avail = rf * maxValue
            p_generating = rf * res.device_flow[d, "el", "out"]
            reserv = cap_avail - p_generating
            df_devs[d] = reserv
    df_devs.columns.name = "device"
    if plotter == "plotly":
        fig = px.area(df_devs, line_shape="hv")
        if includeSum:
            fig.add_scatter(
                x=df_devs.index,
                y=df_devs.sum(axis=1),
                name="SUM",
                line=dict(dash="dot", color="black"),
                line_shape="hv",
            )
        if includeMargin:
            margin = optimiser.all_networks["el"].carrier_data.el_reserve_margin
            # wind contribution (cf compute reserve)
            marginIncr["margin"] = marginIncr["margin"] + margin
            fig.add_scatter(
                x=marginIncr.index,
                y=marginIncr["margin"],
                name="Margin",
                line=dict(dash="dot", color="red"),
                mode="lines",
            )
        fig.update_xaxes(title_text="Timestep")
        fig.update_yaxes(title_text="Reserve (MW)")
        #    return df_devs,marginIncr
    elif plotter == "matplotlib":
        ax = df_devs.plot.area()
        if includeSum:
            df_devs.sum(axis=1).plot(style=":", color="black", drawstyle="steps-post")
        if includeMargin:
            margin = optimiser.all_networks["el"].carrier_data.el_reserve_margin
            # wind contribution (cf compute reserve)
            marginIncr["margin"] = marginIncr["margin"] + margin
            marginIncr[["margin"]].plot(style=":", color="red", ax=ax)

        plt.xlabel("Timestep")
        plt.ylabel("Reserve (MW)")
        fig = plt.gcf()
    return fig


def plot_el_backup(sim_result, filename=None, showMargin=False, returnMargin=False):
    """plot reserve capacity vs device power output"""
    res = sim_result
    res_dev = res.el_backup
    dfP = res.device_flow.copy()
    carrier = "el"
    mask_carrier = dfP.index.get_level_values("carrier") == carrier
    mask_out = dfP.index.get_level_values("terminal") == "out"
    dfP.index = dfP.index.droplevel(level=("carrier", "terminal"))
    dfP = dfP[mask_carrier & mask_out].unstack(0)
    dfP = dfP[res_dev.columns]
    dfMargin = (res_dev - dfP).min(axis=1)
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
                x=dfP.index,
                y=dfP[col],
                legendgroup=col,
                name="{} P_out".format(col),
                line_shape="hv",
                line=dict(color=colour[k], dash="dot"),
            )
            k = k + 1
        if showMargin:
            fig.add_scatter(
                x=dfMargin.index,
                y=dfMargin,
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
        if showMargin:
            dfMargin.plot(ax=ax, linestyle="-", linewidth=3, color="black", label="MARGIN")
            labels = labels + ["MARGIN"]
        plt.gca().set_prop_cycle(None)
        dfP.plot(ax=ax, linestyle=":", legend=False, alpha=1)
        plt.title("Online backup capacity (solid lines) vs device output (dotted lines)")
        ax.legend(labels, loc="lower left", bbox_to_anchor=(1.01, 0), frameon=False)
        if filename is not None:
            plt.savefig(filename, bbox_inches="tight")
    if returnMargin:
        return dfMargin
    return fig


def recompute_elBackup(res, optimisation_model: pyo.Model):
    """Compute reserve
    should give the same as mc.compute_el_reserve"""
    optimiser = optimisation_model
    model = optimiser  # noqa
    all_devices = optimiser.all_devices

    # used capacity for all (relevant) devices:
    carrier = "el"
    devices_elin = optimiser.getDevicesInout(carrier_in=carrier)
    devices_elout = optimiser.getDevicesInout(carrier_out=carrier)
    dfP = res.device_flow
    mask_carrier = dfP.index.get_level_values("carrier") == carrier
    mask_out = dfP.index.get_level_values("terminal") == "out"
    mask_in = dfP.index.get_level_values("terminal") == "out"
    dfPin = dfP[mask_carrier & mask_in]
    dfPin.index = dfPin.index.droplevel(level=("carrier", "terminal"))
    dfPin = dfPin.unstack(0)
    dfPin = dfPin[devices_elin]
    dfPout = dfP[mask_carrier & mask_out]
    dfPout.index = dfPout.index.droplevel(level=("carrier", "terminal"))
    dfPout = dfPout.unstack(0)
    dfPout = dfPout[devices_elout]

    # reserve (unused) capacity by other devices:
    res_dev = pd.DataFrame(columns=dfPout.columns, index=dfPout.index)
    df_ison = res.device_is_on.unstack(0)
    for dev in devices_elout:
        otherdevs = [d for d in devices_elout if d != dev]
        cap_avail = 0
        for d in otherdevs:
            dev2 = all_devices[d].dev_data
            devmodel = dev2.model  # noqa
            maxValue = dev2.flow_max
            if dev2.profile is not None:
                extprofile = dev2.profile
                maxValue = maxValue * res.profiles_nowcast[extprofile]
            ison = 1
            if dev2.start_stop is not None:
                ison = df_ison[d]
            cap_avail += ison * maxValue
        otherdevs_in = [d for d in devices_elin if d != dev]  # noqa
        # TODO: include sheddable load
        res_dev[dev] = cap_avail - dfPout[otherdevs].sum(axis=1)
    return res_dev, dfPout


def plot_el_backup2(mc, filename=None):
    """plot reserve capacity vs device power output"""
    res_dev, dfP = recompute_elBackup(mc)
    plt.figure(figsize=(12, 4))
    ax = plt.gca()
    # default line zorder=2
    # res_dev.plot(ax=ax, linestyle="-", legend=True, alpha=0.5)
    res_dev.plot(ax=ax, drawstyle="steps-post", linestyle="-", legend=True, alpha=0.5)
    #    # critical is the largest load:
    #    c_critical = dfP.idxmax(axis=1)
    # critical is the smallest reserve margin:
    c_critical = (res_dev - dfP).idxmin(axis=1)  # noqa
    #    pd.Series(res_dev.lookup(res_dev.index,c_critical)).plot(ax=ax,
    #             linestyle='-',linewidth=2,label="CRITICAL",
    #             zorder=2.1,legend=True,color="black")
    (res_dev - dfP).min(axis=1).plot(ax=ax, linestyle="--", color="black", linewidth=2)

    # use the same colors
    plt.gca().set_prop_cycle(None)
    # dfP.plot(ax=ax, linestyle=":", legend=False, alpha=0.5)
    dfP.plot(ax=ax, drawstyle="steps-post", linestyle=":", legend=False, alpha=0.5)

    #    pd.Series(dfP.lookup(dfP.index,c_critical)).plot(ax=ax,
    #             linestyle=':',legend=False,linewidth=2,
    #             zorder=2.1,color="black")
    plt.title("Reserve capacity (solid line) vs device output (dotted line)")
    #    leg = ax.get_legend()
    #    leg.set_bbox_to_anchor((1.01,0))
    #    leg.set_frame_on(False)
    labels = list(dfP.columns) + ["MARGIN"]
    ax.legend(labels, loc="lower left", bbox_to_anchor=(1.01, 0), frameon=False)
    #    plt.legend(dfP.columns,loc='lower left', bbox_to_anchor =(1.01,0),
    #               frameon=False)
    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")
    return ax
