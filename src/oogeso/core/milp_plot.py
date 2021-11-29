"""Module for plotting optimisation problem"""

import matplotlib.pyplot as plt
import pandas as pd


def plotDevicePowerLastOptimisation1(mc, device, filename=None):
    model = mc.instance
    devname = model.paramDevice[device]["name"]
    maxP = model.paramDevice[device]["Pmax"]
    plt.figure(figsize=(12, 4))
    ax = plt.gca()
    # plt.title("Results from last optimisation")
    plt.xlabel("Timestep in planning horizon")
    plt.ylabel("Device power (MW)")
    label_profile = ""
    if "profile" in model.paramDevice[device]:
        profile = model.paramDevice[device]["profile"]
        dfa = mc.getProfiles(profile)
        dfa = dfa * maxP
        dfa[profile].plot(ax=ax, label="available")
        label_profile = "({})".format(profile)

    dfIn = pd.DataFrame.from_dict(model.varDeviceFlow.get_values(), orient="index")
    dfIn.index = pd.MultiIndex.from_tuples(dfIn.index, names=("device", "carrier", "terminal", "time"))
    dfIn = dfIn[0].dropna()
    for carr in dfIn.index.levels[1]:
        for term in dfIn.index.levels[2]:
            mask = (dfIn.index.get_level_values(1) == carr) & (dfIn.index.get_level_values(2) == term)
            df_this = dfIn[mask].unstack(0).reset_index()
            if device in df_this:
                df_this[device].plot(
                    ax=ax,
                    linestyle="-",
                    drawstyle="steps-post",
                    marker=".",
                    label="actual ({} {})".format(carr, term),
                )
    ax.legend(loc="upper left")  # , bbox_to_anchor =(1.01,0),frameon=False)

    dfE = pd.DataFrame.from_dict(model.varDeviceStorageEnergy.get_values(), orient="index")
    dfE.index = pd.MultiIndex.from_tuples(dfE.index, names=("device", "time"))
    dfE = dfE[0].dropna()
    df_this = dfE.unstack(0)
    # shift by one because storage at t is storage value _after_ t
    # (just before t+1)
    df_this.index = df_this.index + 1
    if device in df_this:
        ax2 = ax.twinx()
        ax2.set_ylabel("Energy (MWh)", color="red")
        ax2.tick_params(axis="y", labelcolor="red")
        df_this[device].plot(ax=ax2, linestyle="--", color="red", label="storage".format(carr, term))  # noqa
        ax2.legend(
            loc="upper right",
        )
    ax.set_xlim(left=0)

    plt.title("{}:{} {}".format(device, devname, label_profile))
    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")


# def plotDevicePowerLastOptimisation(model,devices='all',filename=None):
#     """Plot power schedule over planning horizon (last optimisation)"""
#     if devices=='all':
#         devices = list(model.setDevice)
#     varPower = model.getDevicePower()
#     df = pd.DataFrame.from_dict(model.varDevicePower.get_values(),
#                                 orient="index")
#     df.index = pd.MultiIndex.from_tuples(df.index,names=('device','time'))
#     df = df[0].unstack(level=0)
#     df_info = pd.DataFrame.from_dict(dict(model.paramDevice.items())).T
#
#     plt.figure(figsize=(12,4))
#     ax=plt.gca()
#     df[devices].plot(ax=ax)
#     labels = (df_info.loc[devices].index.astype(str)
#               +'_'+df_info.loc[devices,'name'])
#     plt.legend(labels,loc='lower left', bbox_to_anchor =(1.01,0),
#                frameon=False)
#     plt.xlabel("Timestep")
#     plt.ylabel("Device power (MW)")
#     if filename is not None:
#         plt.savefig(filename,bbox_inches = 'tight')
#


def plotDeviceSumPowerLastOptimisation(model, carrier="el", filename=None):
    """Plot power schedule over planning horizon (last optimisation)"""

    df = pd.DataFrame.from_dict(model.varDeviceFlow.get_values(), orient="index")
    df.index = pd.MultiIndex.from_tuples(df.index, names=("device", "carrier", "inout", "time"))

    # separate out in out
    df = df[0].unstack(level=2)
    df = df.fillna(0)
    df = df["out"] - df["in"]
    df = df.unstack(level=1)
    dfprod = df[df > 0][carrier].dropna()
    dfcons = df[df < 0][carrier].dropna()
    #        dfprod = df[df>=0].unstack(level=1)[carrier]
    #        dfprod = dfprod[dfprod>0]
    #        dfcons = df[df<0].unstack(level=1)[carrier]
    #        dfcons = dfcons[dfcons<0]

    df_info = pd.DataFrame.from_dict(dict(model.paramDevice.items())).T
    labels = df_info.index.astype(str) + "_" + df_info["name"]

    # plt.figure(figsize=(12,4))
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
    dfprod.unstack(level=0).rename(columns=labels).plot.area(ax=axes[0], linewidth=0)
    # reverse axes to match stack order
    handles, lgnds = axes[0].get_legend_handles_labels()
    axes[0].legend(
        handles[::-1],
        lgnds[::-1],
        loc="lower left",
        bbox_to_anchor=(1.01, 0),
        frameon=False,
    )
    axes[0].set_ylabel("Produced power (MW)")
    axes[0].set_xlabel("")

    dfcons.unstack(level=0).rename(columns=labels).plot.area(ax=axes[1])
    axes[1].legend(loc="lower left", bbox_to_anchor=(1.01, 0), frameon=False)
    axes[1].set_ylabel("Consumed power (MW)")

    axes[1].set_xlabel("Timestep in planning horizon")
    plt.suptitle("Result from last optimisation")
    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")


def plotEmissionRateLastOptimisation(model, filename=None):
    devices = model.setDevice
    timesteps = model.setHorizon
    df_info = pd.DataFrame.from_dict(dict(model.paramDevice.items())).T
    labels = df_info.index.astype(str) + "_" + df_info["name"]

    df = pd.DataFrame(index=timesteps, columns=devices)
    for d in devices:
        for t in timesteps:
            co2 = Multicarrier.compute_CO2(model, devices=[d], timesteps=[t])  # noqa: Fixme: This is broken.
            df.loc[t, d] = pyo.value(co2)  # noqa: Fixme: This is broken.
    plt.figure(figsize=(12, 4))
    ax = plt.gca()
    df.loc[:, ~(df == 0).all()].rename(columns=labels).plot.area(ax=ax, linewidth=0)
    plt.xlabel("Timestep")
    plt.ylabel("Emission rate (kgCO2/s)")
    ax.legend(loc="lower left", bbox_to_anchor=(1.01, 0), frameon=False)
    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")
