#import plotly.plotly as py
import plotly
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")
sns.set_palette("dark")

def plot_df(df,id_var,filename=None,title=None,ylabel="value"):
    '''Plot dataframe using plotly (saved to file)'''
    
    df_tidy = df.reset_index()
    df_tidy.rename(columns={0:ylabel},inplace=True)
    fig = px.line(df_tidy,x="time",y=ylabel,color=id_var,title=title)
    
    #fig = px.line()
    #for cols in df:
    #    fig.add_scatter(df[cols])
    #fig = px.line(df,x=df.index,y=df.values,title=title)
    #py.iplot('data':[{'x':df.index,'y':df.values,'type':'line'}],
    #        'layout': {'title':'title'},
    #        filename=filename)
    if filename is not None:
        plotly.offline.plot(fig,filename=filename)
    
    return fig

def plot_deviceprofile(mc,dev,profiles=None):
    '''plot forecast and actual profile (available power), and device output'''
    dev_param = mc.instance.paramDevice[dev]
    curve = dev_param['external']
    devname = "{}:{}".format(dev,dev_param["name"])
    devPmax = dev_param['Pmax']
    plt.figure(figsize=(12,4))
    ax=plt.gca()
    df= mc._dfDevicePower.unstack(0)[dev]
    df.name = devname
    #plt.plot(df,label=devname,ax=ax)
    df.plot(ax=ax)
    labels=[devname]
    if ((not profiles is None) and (not pd.isna(curve))):
        (profiles['actual'][curve]*devPmax).plot(ax=ax,linestyle='--')
        ax.set_prop_cycle(None)
        (profiles['forecast'][curve]*devPmax).plot(ax=ax,linestyle=":")
        labels = labels+['actual','forecast']
    plt.xlim(df.index.min(),df.index.max())
    ax.legend(labels,loc='lower left', bbox_to_anchor =(1.01,0),
              frameon=False)

def plot_devicePowerEnergy(mc,dev,filename=None):
    model = mc.instance
    dev_param = model.paramDevice[dev]
    devname = "{}:{}".format(dev,dev_param["name"])

    plt.figure(figsize=(12,4))
    plt.title(devname)
    ax=plt.gca()
    
    # Power flow in/out
    dfF = mc._dfDeviceFlow[
            mc._dfDeviceFlow.index.get_level_values('device')==dev
            ]
    #dfF = dfF.reset_index("time")   
    #dfF["power flow"]=dfF.index
    #sns.lineplot(x="time",y=0,data=dfF,ax=ax,hue="power flow",legend="full")
    #ax.set_xlim(dfF['time'].min(),dfF['time'].max())
    dfF.unstack().T.plot(ax=ax,drawstyle="steps-post",marker=".")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Power (MW)")
    tmin = dfF.index.get_level_values('time').min()
    tmax = dfF.index.get_level_values('time').max()+1
    #ax.set_ylim(-dev_param['Pmax'],dev_param['Pmax'])
    ax.legend(loc='upper left')#, bbox_to_anchor =(1.01,0),frameon=False)

    # Energy stored:
    dfE = mc._dfDeviceEnergy[
            mc._dfDeviceEnergy.index.get_level_values('device')==dev
            ]
    if not dfE.empty:
        ax2=ax.twinx()
        ax2.grid(None)
        #dfE = dfE.reset_index("time")
        #dfE["storage"] = [("h",i) for i in dfE.index]
        #sns.lineplot(data=dfE,ax=ax2,x="time",y=0,color="black",label="storage",
        #             legend="full")
        # Shift time by one, since the dfDeviceEnergy[t] is the energy _after_
        # timestep t:
        dfE = dfE.unstack().T
        dfE.index = dfE.index+1
        #dfE.loc[dfE.index.min()-1,dev] = mc.instance.param
        dfE.rename(columns={0:"storage"}).plot(ax=ax2,
                  linestyle=":",color="black")
        ax2.set_ylabel("Energy (MWh)")#,color="red")
        ax2.set_ylim(0,dev_param['Emax'])
        #ax2.tick_params(axis='y', labelcolor="red")
        ax2.legend(loc='upper right')
    ax.set_xlim(tmin,tmax)

    

    if filename is not None:
        plt.savefig(filename,bbox_inches = 'tight')
