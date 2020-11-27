import plotly
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import pydot
import logging


sns.set_style("whitegrid")
#sns.set_palette("dark")

plotter="matplotlib"

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

def plot_deviceprofile(mc,devs,filename=None,reverseLegend=True,
        includeForecasts=False,includeOnOff=False,includePrep=False):
    '''plot forecast and actual profile (available power), and device output'''
    if type(devs) is not list:
        devs = [devs]
    if includeForecasts & (len(devs)>1):
        print("Can only plot one device when showing forecasts")
        return
    df = mc._dfDeviceFlow.unstack(['carrier','terminal'])[
            ('el','out')].unstack('device')[devs]
    df.columns.name="devices"
    nrows=1
    if includeOnOff:
        nrows=nrows+1
    if includePrep:
        nrows=nrows+1
    df2 = mc._dfDeviceIsOn.unstack('device')[devs]
    dfPrep = mc._dfDeviceIsPrep.unstack('device')[devs]
    timerange=list(mc._dfExportRevenue.index)
    if plotter=="plotly":
        fig = plotly.subplots.make_subplots(rows=nrows, cols=1,shared_xaxes=True)
        colour = plotly.colors.DEFAULT_PLOTLY_COLORS
        k=0
        rowOnOff=2
        rowPrep=2
        if includeOnOff:
            rowPrep=3
        for col in df:
            dev=col
            dev_param = mc.instance.paramDevice[dev]
            k=k+1
            fig.add_scatter(x=df.index,y=df[col],line_shape='hv',name=col,
                line=dict(color=colour[k]),
                stackgroup="P",legendgroup=col,row=1,col=1)
            if includeOnOff & (dev_param['model']=='gasturbine'):
                fig.add_scatter(x=df2.index,y=df2[col],line_shape='hv',name=col,
                    line=dict(color=colour[k],dash='dash'),
                    stackgroup="ison",legendgroup=col,row=rowOnOff,col=1,
                    showlegend=False)
            if includePrep & (dev_param['model']=='gasturbine'):
                fig.add_scatter(x=dfPrep.index,y=dfPrep[col],line_shape='hv',name=col,
                    line=dict(color=colour[k],dash='dash'),
                    stackgroup="ison",legendgroup=col,row=rowPrep,col=1,
                    showlegend=False)
            if includeForecasts & ('profile' in dev_param):
                curve = dev_param['profile']
                devPmax=dev_param['Pmax']
                fig.add_scatter(x=timerange,
                    y=mc._df_profiles_actual.loc[timerange,curve]*devPmax,
                    line_shape='hv',line=dict(color=colour[k+1]),
                    name='--nowcast',legendgroup=col,row=1,col=1)
                fig.add_scatter(x=timerange,
                    y=mc._df_profiles_forecast.loc[timerange,curve]*devPmax,
                    line_shape='hv',line=dict(color=colour[k+2]),
                    name='--forecast',legendgroup=col,row=1,col=1)
        fig.update_xaxes(row=1,col=1,title_text="")
        fig.update_xaxes(row=nrows,col=1,title_text="Timestep")
        fig.update_yaxes(row=1,col=1,title_text="Power supply (MW)")
        if includeOnOff:
            fig.update_yaxes(row=rowOnOff,col=1,title_text="On/off status")
        if includePrep:
            fig.update_yaxes(row=rowPrep,col=1,title_text="Startup",nticks=2)
        if reverseLegend:
            fig.update_layout(legend_traceorder="reversed")
        fig.update_layout(height=600)
        #fig.show()
    elif plotter=="matplotlib":

        fig=plt.figure(figsize=(12,4))
        ax=plt.gca()
        labels=[]
        offset_online=0
        #df.plot(ax=ax)
        for dev in devs:
            dev_param = mc.instance.paramDevice[dev]
            devname = "{}:{}".format(dev,dev_param["name"])
            devPmax = dev_param['Pmax']
            # el power out:
            #df= mc._dfDeviceFlow.unstack([1,2])[('el','out')].unstack(0)[dev]
            #df.name = devname
            df[dev].plot(ax=ax)
            #get the color of the last plotted line (the one just plotted)
            col = ax.get_lines()[-1].get_color()
            labels=labels+[devname]
            if 'profile' in dev_param:
                curve = dev_param['profile']
                (mc._df_profiles_actual.loc[timerange,curve]*devPmax).plot(
                    ax=ax,linestyle='--')
                #ax.set_prop_cycle(None)
                (mc._df_profiles_forecast.loc[timerange,curve]*devPmax).plot(
                    ax=ax,linestyle=":")
                labels = labels+['--nowcast','--forecast']
            if dev_param['model']=='gasturbine':
                #df2=mc._dfDeviceIsOn.unstack(0)[dev]+offset_online
                offset_online +=0.1
                df2[dev].plot(ax=ax,linestyle='--',color=col)
                labels = labels+['--online']
        plt.xlim(df.index.min(),df.index.max())
        ax.legend(labels,loc='lower left', bbox_to_anchor =(1.01,0),
                  frameon=False)
        if filename is not None:
            plt.savefig(filename,bbox_inches = 'tight')
    return fig

def plot_devicePowerEnergy(mc,dev,filename=None,energy_fill_opacity=None):
    '''Plot power in/out of device and storage level (if any)'''
    model = mc.instance
    dev_param = model.paramDevice[dev]
    devname = "{}:{}".format(dev,dev_param["name"])

    carrier = 'el'
    # Power flow in/out
    dfF = mc._dfDeviceFlow[dev,carrier].unstack('terminal')
#    dfF = mc._dfDeviceFlow[
#            mc._dfDeviceFlow.index.get_level_values('device')==dev,carrier
#            ]
#    dfF =dfF.unstack('terminal') # now index is time, columns are in/out
    # Energy stored:
#    dfE = mc._dfDeviceStorageEnergy[
#            mc._dfDeviceStorageEnergy.index.get_level_values('device')==dev
#            ]
    dfE=pd.DataFrame()
    if dev in mc._dfDeviceStorageEnergy:
        dfE =  pd.DataFrame(mc._dfDeviceStorageEnergy[dev])
        # Shift time by one, since the dfDeviceEnergy[t] is the energy _after_
        # timestep t:
#        dfE = dfE.unstack().T
        dfE.index = dfE.index+1
        dfE = dfE.rename(columns={0:"storage"})
        #dfE.loc[dfE.index.min()-1,dev] = mc.instance.param

    if plotter=="plotly":
        fig = plotly.subplots.make_subplots(specs=[[{"secondary_y": True}]])
        for col in dfF.columns:
            fig.add_scatter(x=dfF.index,y=dfF[col],line_shape='hv',name=col,
                secondary_y=True,fill='tozeroy')
        if not dfE.empty:
            fig.add_scatter(x=dfE.index,y=dfE['storage'],name='storage',
                secondary_y=False,fill='tozeroy')#,line=dict(dash='dot'))
            if energy_fill_opacity is not None:
                k=len(fig['data'])-1
                linecol=plotly.colors.DEFAULT_PLOTLY_COLORS[k]
                opacity=energy_fill_opacity
                fillcol='rgba({}, {})'.format(linecol[4:][:-1],opacity)
                fig['data'][k]['fillcolor']=fillcol
                fig['data'][k]['fill']='tozeroy'
            fig.update_yaxes(title_text="Energy storage (MWh)",
            secondary_y=False,side="right")
        fig.update_xaxes(title_text="Timestep")
        fig.update_yaxes(title_text="Power (MW)", secondary_y=True,side="left")

    elif plotter=="matplotlib":
        fig=plt.figure(figsize=(12,4))
        plt.title(devname)
        ax=plt.gca()
        dfF.plot(ax=ax,drawstyle="steps-post",marker=".")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Power (MW)")
        tmin = dfF.index.get_level_values('time').min()
        tmax = dfF.index.get_level_values('time').max()+1
        ax.set_ylim(0,dev_param['Pmax'])
        ax.legend(loc='upper left')#, bbox_to_anchor =(1.01,0),frameon=False)

        if not dfE.empty:
            ax2=ax.twinx()
            ax2.grid(None)
            dfE.plot(ax=ax2,
                      linestyle=":",color="black")
            ax2.set_ylabel("Energy (MWh)")#,color="red")
            if dev_param['model'] in ['storage_el']:
                ax2.set_ylim(0,dev_param['Emax'])
            elif dev_param['model'] in ['well_injection']:
                ax2.set_ylim(-dev_param['Emax']/2,dev_param['Emax']/2)
            #ax2.tick_params(axis='y', labelcolor="red")
            ax2.legend(loc='upper right')
        ax.set_xlim(tmin,tmax)
        if filename is not None:
            plt.savefig(filename,bbox_inches = 'tight')
    return fig

def plot_SumPowerMix(mc,carrier,filename=None,reverseLegend=True):

    # Power flow in/out
    dfF = mc._dfDeviceFlow
    tmin = dfF.index.get_level_values('time').min()
    tmax = dfF.index.get_level_values('time').max()+1
    mask_carrier = (dfF.index.get_level_values('carrier')==carrier)
    mask_in = (dfF.index.get_level_values('terminal')=='in')
    mask_out = (dfF.index.get_level_values('terminal')=='out')
    dfF_out = dfF[mask_carrier&mask_out]
    dfF_out.index = dfF_out.index.droplevel(level=("carrier","terminal"))
    dfF_out = dfF_out.unstack(0)
    keepcols = mc.getDevicesInout(carrier_out=carrier)
    dfF_out = dfF_out[keepcols]
    dfF_in = dfF[mask_carrier&mask_in]
    dfF_in.index = dfF_in.index.droplevel(level=("carrier","terminal"))
    dfF_in = dfF_in.unstack(0)
    keepcols = mc.getDevicesInout(carrier_in=carrier)
    dfF_in = dfF_in[keepcols]
#    dfF_in.rename(columns={d:"{}:{}".format(d,mc.instance.paramDevice[d]['name'])
#        for d in dfF_in.columns},inplace=True)
#    dfF_out.rename(columns={d:"{}:{}".format(d,mc.instance.paramDevice[d]['name'])
#        for d in dfF_out.columns},inplace=True)


    if plotter=="plotly":
        fig = plotly.subplots.make_subplots(rows=2, cols=1,shared_xaxes=True,
            vertical_spacing=0.05)
        for col in dfF_in:
            fig.add_scatter(x=dfF_in.index,y=dfF_in[col],
                line_shape='hv',name="in:"+col,stackgroup="in",
                legendgroup=col,row=2,col=1)
        for col in dfF_out:
            fig.add_scatter(x=dfF_out.index,y=dfF_out[col],
                line_shape='hv',name="out:"+col,stackgroup="out",
                legendgroup=col,row=1,col=1)
        fig.update_xaxes(row=1,col=1,title_text="")
        fig.update_xaxes(row=2,col=1,title_text="Timestep")
        fig.update_yaxes(row=1,col=1,title_text="Power supply (MW)")
        fig.update_yaxes(row=2,col=1,title_text="Power consumption (MW)")
        if reverseLegend:
            fig.update_layout(legend_traceorder="reversed")
        fig.update_layout(height=600)
        #fig.show()
    elif plotter=="matplotlib":
        fig,axes = plt.subplots(nrows=2,ncols=1,figsize=(12,8))
        #plt.figure(figsize=(12,4))
        plt.suptitle("Sum power ({})".format(carrier))
        dfF_out.plot.area(ax=axes[0],linewidth=0)
        dfF_in.plot.area(ax=axes[1],linewidth=0)
        axes[0].set_ylabel("Power supply (MW)")
        axes[1].set_ylabel("Power consumption (MW)")
        axes[0].set_xlabel("")
        axes[1].set_xlabel("Timestep")
        for ax in axes:
            if reverseLegend:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles[::-1], labels[::-1],
                          loc='lower left', bbox_to_anchor =(1.01,0),frameon=False)
            else:
                ax.legend(loc='lower left', bbox_to_anchor =(1.01,0),frameon=False)
            ax.set_xlim(tmin,tmax)

        if filename is not None:
            plt.savefig(filename,bbox_inches = 'tight')
    return fig

def plot_ExportRevenue(mc,filename=None,currency="$"):
    if plotter=="plotly":
        dfplot = mc._dfExportRevenue.loc[:,mc._dfExportRevenue.sum()>0]
        fig = px.area(dfplot)
        fig.update_xaxes(title_text="Timestep")
        fig.update_yaxes(title_text="Revenue ({}/s)".format(currency))
    elif plotter=="matplotlib":
        fig=plt.figure(figsize=(12,4))
        plt.title("Export revenue ($/s)")
        ax=plt.gca()
        ax.set_ylabel("{}/s".format(currency))
        ax.set_xlabel("Timestep")
        (mc._dfExportRevenue.loc[:,mc._dfExportRevenue.sum()>0]).plot.area(
                ax=ax,linewidth=0)
        if filename is not None:
            plt.savefig(filename,bbox_inches = 'tight')
    return fig

def plot_CO2rate(mc,filename=None):
    plt.figure(figsize=(12,4))
    plt.title("CO2 emission rate (kgCO2/s)")
    ax=plt.gca()
    ax.set_ylabel("kgCO2/s")
    ax.set_xlabel("Timestep")
    mc._dfCO2rate.plot()
    if filename is not None:
        plt.savefig(filename,bbox_inches = 'tight')

def plot_CO2rate_per_dev(mc,filename=None,reverseLegend=False):
    df_info = pd.DataFrame.from_dict(dict(mc.instance.paramDevice.items())).T
    #labels = (df_info.index.astype(str))# +'_'+df_info['name'])
    dfplot = mc._dfCO2rate_per_dev.loc[:,~(mc._dfCO2rate_per_dev==0).all()
                    ]
    #dfplot.columns=labels
    if plotter=="plotly":
        fig = plotly.subplots.make_subplots(rows=1, cols=1)
        #,shared_xaxes=True,vertical_spacing=0.05)
        #fig = px.line(dfplot,x="time",y=ylabel,color=id_var,title=title)
        for col in dfplot:
            fig.add_scatter(x=dfplot.index,y=dfplot[col],
                line_shape='hv',name=col,
                stackgroup="one",row=1,col=1)
        fig.update_xaxes(title_text="Timestep")
        fig.update_yaxes(title_text="Emission rate (kgCO2/s)")
        if reverseLegend:
            fig.update_layout(legend_traceorder="reversed")
        fig.update_layout(height=600)
    else:
        fig=plt.figure(figsize=(12,4))
        ax=plt.gca()
        ax.set_ylabel("Emission rate (kgCO2/s)")
        ax.set_xlabel("Timestep")
        dfplot.plot.area(ax=ax,linewidth=0)
        if reverseLegend:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[::-1], labels[::-1],
                      loc='lower left', bbox_to_anchor =(1.01,0),frameon=False)
        else:
            ax.legend(loc='lower left', bbox_to_anchor =(1.01,0),frameon=False)

        if filename is not None:
            plt.savefig(filename,bbox_inches = 'tight')
    return fig

def plot_CO2_intensity(mc,filename=None):
    title="CO2 intensity (kgCO2/Sm3oe)"
    xlabel="Timestep"
    ylabel="CO2 intensity (kgCO2/Sm3oe)"
    dfplot=mc._dfCO2intensity
    if plotter=="plotly":
        fig = plotly.subplots.make_subplots(rows=1, cols=1)
        #,shared_xaxes=True,vertical_spacing=0.05)
        fig = px.line(dfplot,x=dfplot.index,y=dfplot.values)#,title=title)
        fig.update_xaxes(title_text=xlabel)
        fig.update_yaxes(title_text=ylabel)
    else:
        fig=plt.figure(figsize=(12,4))
        plt.title(title)
        ax=plt.gca()
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        dfplot.plot()
        if filename is not None:
            plt.savefig(filename,bbox_inches = 'tight')
    return fig


def plotProfiles(profiles,curves=None,filename=None):
    '''Plot profiles (forecast and actual)'''

    if curves is None:
        curves = profiles['actual'].columns
    if plotter=="plotly":
        pd.concat(profiles).unstack(0).melt()
        df = pd.concat({'actual':profiles['actual'][curves],
                        'forecast':profiles['forecast'][curves]})
        df =df.reset_index().rename(columns={'level_0':'type'}).melt(
            id_vars=['type','timestep'])
        fig = px.line(df,x='timestep',y='value',
            line_group='type',color='variable',line_dash='type')
        fig.show()
    elif plotter=="matplotlib":
        plt.figure(figsize=(12,4))
        ax=plt.gca()
        profiles['actual'][curves].plot(ax=ax)
        #reset color cycle (so using the same as for the actual plot):
        ax.set_prop_cycle(None)
        profiles['forecast'][curves].plot(ax=ax,linestyle=":")
        labels = curves
        ax.legend(labels=labels,loc='lower left', bbox_to_anchor =(1.01,0),
                  frameon=False)
        plt.xlabel("Timestep")
        plt.ylabel("Relative value")
        plt.title("Actual vs forecast profile (actual=sold line)")
        if filename is not None:
            plt.savefig(filename,bbox_inches = 'tight')

def plotDevicePowerFlowPressure(mc,dev,carriers_inout=None,filename=None):
    model = mc.instance
    dev_param = model.paramDevice[dev]
    model = dev_param["model"]
    node = dev_param["node"]
    devname = "{}:{}".format(dev,dev_param["name"])
    #linecycler = itertools.cycle(['-','--',':','-.']*10)
    if carriers_inout is None:
        carriers_inout = mc.devicemodel_inout()[model]
    if 'serial' in carriers_inout:
        del carriers_inout['serial']

    plt.figure(figsize=(12,4))
    ax=plt.gca()
    #ax.plot(mc._dfDevicePower.unstack(0)[dev],'-.',label="DevicePower")
    for inout,carriers in carriers_inout.items():
        if inout=='in':
            ls='--'
        else:
            ls=':'
        for carr in carriers:
            ax.plot(mc._dfDeviceFlow.unstack([0,1,2])[(dev,carr,inout)],
                     ls,label="DeviceFlow ({},{})".format(carr,inout))
    #Pressure
    for inout,carriers in carriers_inout.items():
        for carr in carriers:
            if carr!='el':
                ax.plot(mc._dfTerminalPressure.unstack(0)[node].unstack([0,1])
                    [(carr,inout)],label="TerminalPressure ({},{})"
                    .format(carr,inout))
    plt.title(devname)
    plt.xlabel("Timestep")
    plt.ylabel("Value")
    plt.legend(loc='lower left', bbox_to_anchor =(1.01,0),frameon=False)
    if filename is not None:
        plt.savefig(filename,bbox_inches = 'tight')


def plotNetwork(mc,timestep=0,filename=None,prog='dot',
    only_carrier=None,rankdir='LR',plotDevName=False, numberformat="{:.2g}",
     **kwargs):
    """Plot energy network

    mc : object
        Multicarrier object
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
    """

    # Idea: in general, there are "in" and "out" terminals. If there are
    # no serial devices, then these are merged into a single terminal
    # (prettier plot"). Whether the single terminal is shown as an in or out
    # terminal (left or irght), depends on whether it is an input or output
    # of a majority of the connected devices.

    cluster = {}
    col = {'t': {'el':'red','gas':'orange','heat':'darkgreen',
                 'wellstream':'brown','oil':'black','water':'blue'},
           'e': {'el':'red','gas':'orange','heat':'darkgreen',
                 'wellstream':'brown','oil':'black','water':'blue'},
           'd': 'white',
           'cluster':'lightgray'
           }
    #dotG = pydot.Dot(graph_type='digraph') #rankdir='LR',newrank='false')
    dotG = pydot.Dot(graph_type='digraph',rankdir=rankdir,**kwargs)
    model = mc.instance
    if only_carrier is None:
        carriers = model.setCarrier
    elif type(only_carrier) is str:
        carriers = [only_carrier]
    else:
        carriers = only_carrier

    devicemodels = mc.devicemodel_inout()

    # plot all node and terminals:
    for n_id in model.setNode:
        cluster = pydot.Cluster(graph_name=n_id,label=n_id,
               style='filled',color=col['cluster'])
        terms_in=pydot.Subgraph(rank='min')
        gr_devices = pydot.Subgraph(rank='same')
        terms_out=pydot.Subgraph(rank='max')
        for carrier in carriers:
            #add only terminals that are connected to something (device or edge)
            if mc.nodeIsNonTrivial(n_id,carrier):

                # add devices at this node
                if n_id in model.paramNodeDevices:
                    devs = model.paramNodeDevices[n_id]
                else:
                    devs=[]
                num_in=0
                num_out=0
                for d in devs:
                    dev_model = model.paramDevice[d]['model']
                    devlabel = d # use index as label
                    devlabel = "{}\n{}".format(d,dev_model)
                    if plotDevName:
                        devlabel = "{} {}".format(devlabel,model.paramDevice[d]['name'])
                    #if timestep is not None:
                    #    p_dev = mc._dfDevicePower[(d,timestep)]
                    #    devlabel = "{}\n{:.2f}".format(devlabel,p_dev)
                    carriers_in = devicemodels[dev_model]['in']
                    carriers_out = devicemodels[dev_model]['out']
                    carriers_in_lim = list(set(carriers_in)&set(carriers))
                    carriers_out_lim = list(set(carriers_out)&set(carriers))
                    if (carriers_in_lim!=[]) or (carriers_out_lim!=[]):
                        gr_devices.add_node(pydot.Node(d,color=col['d'],
                               style='filled',label=devlabel))
                    if carrier in carriers_in_lim:
                        num_in += 1
                        if timestep is None:
                            devedgelabel=''
                        else:
                            f_in = mc._dfDeviceFlow[(d,carrier,'in',timestep)]
                            #logging.info("{},{},{},{}".format(d,carrier,timestep,f_in))
                            devedgelabel = numberformat.format(f_in)
                        if model.paramNodeCarrierHasSerialDevice[n_id][carrier]:
                            n_in = n_id+'_'+carrier+'_in'
                        else:
                            n_in = n_id+'_'+carrier
                        dotG.add_edge(pydot.Edge(dst=d,src=n_in,
                             color=col['e'][carrier],
                             fontcolor=col['e'][carrier],
                             label=devedgelabel))
                    if carrier in carriers_out_lim:
                        num_out += 1
                        if timestep is None:
                            devedgelabel = ''
                        else:
                            f_out = mc._dfDeviceFlow[(d,carrier,'out',timestep)]
                            devedgelabel = numberformat.format(f_out)
                        if model.paramNodeCarrierHasSerialDevice[n_id][carrier]:
                            n_out = n_id+'_'+carrier+'_out'
                        else:
                            n_out = n_id+'_'+carrier
                        dotG.add_edge(pydot.Edge(dst=n_out,src=d,
                             color=col['e'][carrier],
                             fontcolor=col['e'][carrier],
                             label=devedgelabel))

                # add in/out terminals
                supp=""
                if model.paramNodeCarrierHasSerialDevice[n_id][carrier]:
                    supp = '_out'
                label_in = carrier+'_in '
                label_out= carrier+supp+' '
                if timestep is None:
                    pass
                elif carrier in ['gas','wellstream','oil','water']:
                    label_in += numberformat.format(mc._dfTerminalPressure[(n_id,carrier,'in',timestep)])
                    label_out += numberformat.format(mc._dfTerminalPressure[(n_id,carrier,'out',timestep)])
                elif carrier=='el':
                    label_in += numberformat.format(mc._dfElVoltageAngle[(n_id,timestep)])
                    label_out += numberformat.format(mc._dfElVoltageAngle[(n_id,timestep)])
                # Add two terminals if there are serial devices, otherwise one:
                if model.paramNodeCarrierHasSerialDevice[n_id][carrier]:
                    terms_in.add_node(pydot.Node(name=n_id+'_'+carrier+'_in',
                           color=col['t'][carrier],label=label_in,shape='box'))
                    terms_out.add_node(pydot.Node(name=n_id+'_'+carrier+'_out',
                           color=col['t'][carrier],label=label_out,shape='box'))
                else:
                    #TODO: make this in or out depending on connected devices
                    if num_out>num_in:
                        terms_out.add_node(pydot.Node(name=n_id+'_'+carrier,
                           color=col['t'][carrier],label=label_out,shape='box'))
                    else:
                        terms_in.add_node(pydot.Node(name=n_id+'_'+carrier,
                           color=col['t'][carrier],label=label_out,shape='box'))



        cluster.add_subgraph(terms_in)
        cluster.add_subgraph(gr_devices)
        cluster.add_subgraph(terms_out)
        dotG.add_subgraph(cluster)

    # plot all edges (per carrier):
    for carrier in carriers:
        for i,e in model.paramEdge.items():
            if e['type']==carrier:
                if timestep is None:
                    edgelabel=''
                    if 'pressure.from' in e:
                        edgelabel = '{} {}-'.format(edgelabel,e['pressure.from'])
                    if 'pressure.to' in e:
                        edgelabel = '{}-{}'.format(edgelabel,e['pressure.to'])
                else:
                    edgelabel = numberformat.format(mc._dfEdgeFlow[(i,timestep)])
                n_from = e['nodeFrom']
                n_to = e['nodeTo']
                # name of terminal depends on whether it serial or single
                if model.paramNodeCarrierHasSerialDevice[n_from][carrier]:
                    t_out = n_from+'_'+carrier+'_out'
                else:
                    t_out = n_from+'_'+carrier
                if model.paramNodeCarrierHasSerialDevice[n_to][carrier]:
                    t_in = n_to+'_'+carrier+'_in'
                else:
                    t_in = n_to+'_'+carrier

                dotG.add_edge(pydot.Edge(src=t_out,dst=t_in,
                                         color='"{0}:invis:{0}"'.format(col['e'][carrier]),
                                         fontcolor=col['e'][carrier],
                                         label=edgelabel))

#    # plot devices and device connections:
#    devicemodels = mc.devicemodel_inout()
#    for n,devs in model.paramNodeDevices.items():
#        for d in devs:
#            dev_model = model.paramDevice[d]['model']
#            nodelabel = "{} {}".format(d,model.paramDevice[d]['name'])
#            if timestep is not None:
#                #p_dev = pyo.value(model.varDevicePower[d,timestep])
#                p_dev = mc._dfDevicePower[(d,timestep)]
#                nodelabel = "{}\n{:.2f}".format(nodelabel,p_dev)
#            carriers_in = devicemodels[dev_model]['in']
#            carriers_out = devicemodels[dev_model]['out']
#            carriers_in_lim = list(set(carriers_in)&set(carriers))
#            carriers_out_lim = list(set(carriers_out)&set(carriers))
#            if (carriers_in_lim!=[]) or (carriers_out_lim!=[]):
#                cluster[n].add_node(pydot.Node(d,color=col['d'],style='filled',
#                   label=nodelabel))
#            #print("carriers in/out:",d,carriers_in,carriers_out)
#            for carrier in carriers_in_lim:
#                if timestep is None:
#                    devlabel=''
#                else:
#                    #f_in = pyo.value(model.varDeviceFlow[d,carrier,'in',timestep])
#                    f_in = mc._dfDeviceFlow[(d,carrier,'in',timestep)]
#                    devlabel = "{:.2f}".format(f_in)
#                if model.paramNodeCarrierHasSerialDevice[n][carrier]:
#                    n_in = n+'_'+carrier+'_in'
#                else:
#                    n_in = n+'_'+carrier
#                dotG.add_edge(pydot.Edge(dst=d,src=n_in,
#                     color=col['e'][carrier],
#                     fontcolor=col['e'][carrier],
#                     label=devlabel))
#            for carrier in carriers_out_lim:
#                if timestep is None:
#                    devlabel = ''
#                else:
#                    #f_out = pyo.value(model.varDeviceFlow[d,carrier,'out',timestep])
#                    f_out = mc._dfDeviceFlow[(d,carrier,'out',timestep)]
#                    devlabel = "{:.2f}".format(f_out)
#                if model.paramNodeCarrierHasSerialDevice[n][carrier]:
#                    n_out = n+'_'+carrier+'_out'
#                else:
#                    n_out = n+'_'+carrier
#                dotG.add_edge(pydot.Edge(dst=n_out,src=d,
#                     color=col['e'][carrier],
#                     fontcolor=col['e'][carrier],
#                     label=devlabel))
    if filename is not None:
        #prog='dot' gives the best layout.
        dotG.write_png(filename,prog=prog)
    return dotG

def plotGasTurbineEfficiency(fuelA=2.35,fuelB=0.53,energycontent=40,
        co2content=2.34,filename=None,Pmax=None):
    '''
    co2content : CO2 content, kgCO2/Sm3gas
    energycontent: energy content, MJ/Sm3gas
    A,B : linear parameters
    '''

    x_pow = np.linspace(0,1,50)
    y_fuel = fuelB + fuelA*x_pow
    #Pgas = Qgas*energycontent: Qgas=Pgas/Pmax * Pmax/energycontent

    nplots=3 if Pmax is None else 4
    #if Pmax is not None
    plt.figure(figsize=(4*nplots,4))
    #plt.suptitle("Gas turbine fuel characteristics")

    if Pmax is not None:
        y_fuel_sm3 = y_fuel*Pmax/energycontent #Sm3/s
        plt.subplot(1,nplots,1)
        plt.title("Fuel usage (Sm3/h)")
        plt.xlabel("Electric power output (MW)")
        plt.plot(x_pow*Pmax,y_fuel_sm3*3600) #per hour
        plt.ylim(bottom=0)

    plt.subplot(1,nplots,nplots-2)
    plt.title("Fuel usage ($P_{gas}/P_{el}^{max}$)")
    plt.xlabel("Electric power output ($P_{el}/P_{el}^{max}$)")
    #plt.ylabel("Gas power input ($P_{gas}/P_{el}^{max}$)")
    plt.plot(x_pow,y_fuel)
    plt.ylim(bottom=0)

    plt.subplot(1,nplots,nplots-1)
    plt.title("Efficiency ($P_{el}/P_{gas}$)")
    plt.xlabel("Electric power output ($P_{el}/P_{el}^{max}$)")
    plt.plot(x_pow,x_pow/y_fuel)

#    plt.subplot(1,3,3)
#    plt.title("Specific fuel usage ($P_{gas}/P_{el}$)")
#    plt.xlabel("Electric power output ($P_{el}/P_{el}^{max}$)")
#    plt.plot(x_pow,y_fuel/x_pow)
#    plt.ylim(top=30)

    # 1 MJ = 3600 MWh
    with np.errstate(divide='ignore', invalid='ignore'):
        emissions = 3600*co2content/energycontent * y_fuel/x_pow
    plt.subplot(1,nplots,nplots)
    plt.title("Emission intensity (kgCO2/MWh)")
    plt.xlabel("Electric power output ($P_{el}/P_{el}^{max}$)")
    plt.plot(x_pow,emissions)
    plt.ylim(top=2000)

    if filename is not None:
        plt.savefig(filename,bbox_inches = 'tight')


def plotReserve(mc,includeMargin=True,dynamicMargin=True,useForecast=False):
    '''Plot unused online capacity by all el devices'''
    df_devs=pd.DataFrame()
    model=mc.instance
    inout=mc.devicemodel_inout()
    timerange=list(mc._dfExportRevenue.index)
    marginIncr = pd.DataFrame(0,index=timerange,columns=['margin'])
    for d in mc.instance.setDevice:
        devmodel = model.paramDevice[d]['model']
        rf=1
        if 'el' in inout[devmodel]['out']:
            # Generators and storage
            maxValue = model.paramDevice[d]['Pmax']
            if 'profile' in model.paramDevice[d]:
                extprofile = model.paramDevice[d]['profile']
                if useForecast:
                    maxValue = maxValue*mc._df_profiles_forecast.loc[
                        timerange,extprofile]
                else:
                    maxValue = maxValue*mc._df_profiles_actual.loc[
                        timerange,extprofile]
            if devmodel in ['gasturbine']:
                ison = mc._dfDeviceIsOn[d]
                maxValue = ison*maxValue
            elif devmodel in ['storage_el']:
                maxValue = (mc._dfDeviceStoragePmax[d]
                            +mc._dfDeviceFlow[d,'el','in'])
            if ('reserve_factor' in model.paramDevice[d]):
                reserve_factor=model.paramDevice[d]['reserve_factor']
                if reserve_factor==0:
                    # device does not count towards reserve
                    rf=0
                if dynamicMargin:
                    # instead of reducing reserve, increase the margin instead
                    # R*0.8-M = R - (M+0.2R) - this shows better in the plot what
                    marginIncr['margin'] += rf*maxValue*(1-reserve_factor)
                else:
                    maxValue = maxValue*reserve_factor
            cap_avail = rf*maxValue
            p_generating = rf*mc._dfDeviceFlow[d,'el','out']
            reserv = cap_avail-p_generating
            df_devs[d] = reserv
    df_devs.columns.name="device"
    fig=px.area(df_devs,line_shape='hv')
    fig.add_scatter(x=df_devs.index,y=df_devs.sum(axis=1),name='SUM',
        line=dict(dash='dot',color='black'),line_shape='hv')
    if includeMargin:
        margin=mc.instance.paramParameters['elReserveMargin']
        # wind contribution (cf compute reserve)
        marginIncr['margin'] = marginIncr['margin'] + margin
        fig.add_scatter(x=marginIncr.index, y=marginIncr['margin'],
            name='Margin',line=dict(dash='dot',color='red'),mode="lines")
    fig.update_xaxes(title_text="Timestep")
    fig.update_yaxes(title_text="Reserve (MW)")
#    return df_devs,marginIncr
    return fig


def plotElBackup(mc,filename=None,showMargin=False,returnMargin=False):
    '''plot reserve capacity vs device power output'''
    res_dev = mc._dfElBackup
    dfP = mc._dfDeviceFlow.copy()
    carrier='el'
    mask_carrier = (dfP.index.get_level_values('carrier')==carrier)
    mask_out = (dfP.index.get_level_values('terminal')=='out')
    dfP.index = dfP.index.droplevel(level=("carrier","terminal"))
    dfP = dfP[mask_carrier&mask_out].unstack(0)
    dfP = dfP[res_dev.columns]
    dfMargin = (res_dev-dfP).min(axis=1)
    if plotter=="plotly":
        fig=px.line()#title="Online backup capacity (solid lines) vs device output (dotted lines)")
        colour = plotly.colors.DEFAULT_PLOTLY_COLORS
        k=0
        for col in res_dev:
            fig.add_scatter(x=res_dev.index,y=res_dev[col],mode="lines",
                legendgroup=col,name="{} R_other".format(col),
                line_shape='hv',line=dict(color=colour[k]))
            fig.add_scatter(x=dfP.index,y=dfP[col],
                legendgroup=col,name="{} P_out".format(col),
                line_shape='hv',line=dict(color=colour[k],dash="dot"))
            k=k+1
        if showMargin:
            fig.add_scatter(x=dfMargin.index,y=dfMargin,name="MARGIN",
                line=dict(color="black"),line_shape='hv')
        fig.update_xaxes(title_text="Timestep")
        fig.update_yaxes(title_text="Power (MW)")
    elif plotter=="matplotlib":
        fig=plt.figure(figsize=(12,4))
        ax = plt.gca()
        res_dev.plot(ax=ax,legend=True,alpha=1,linestyle="-")
        labels=list(res_dev.columns)
        if showMargin:
            dfMargin.plot(ax=ax,linestyle="-",linewidth=3,
                                       color="black",label="MARGIN")
            labels = labels + ['MARGIN']
        plt.gca().set_prop_cycle(None)
        dfP.plot(ax=ax,linestyle=':',legend=False,alpha=1)
        plt.title("Online backup capacity (solid lines) vs device output (dotted lines)")
        ax.legend(labels,loc='lower left', bbox_to_anchor =(1.01,0),
                   frameon=False)
        if filename is not None:
            plt.savefig(filename,bbox_inches = 'tight')
    if returnMargin:
        return dfMargin
    return fig

def recompute_elBackup(mc):
    '''Compute reserve
    should give the same as mc.compute_elReserve'''
    model = mc.instance

    # used capacity for all (relevant) devices:
    carrier = 'el'
    devices_elin = mc.getDevicesInout(carrier_in=carrier)
    devices_elout = mc.getDevicesInout(carrier_out=carrier)
    dfP = mc._dfDeviceFlow
    mask_carrier = (dfP.index.get_level_values('carrier')==carrier)
    mask_out = (dfP.index.get_level_values('terminal')=='out')
    mask_in = (dfP.index.get_level_values('terminal')=='out')
    dfPin = dfP[mask_carrier&mask_in]
    dfPin.index = dfPin.index.droplevel(level=("carrier","terminal"))
    dfPin = dfPin.unstack(0)
    dfPin = dfPin[devices_elin]
    dfPout = dfP[mask_carrier&mask_out]
    dfPout.index = dfPout.index.droplevel(level=("carrier","terminal"))
    dfPout = dfPout.unstack(0)
    dfPout = dfPout[devices_elout]

    # reserve (unused) capacity by other devices:
    res_dev = pd.DataFrame(columns=dfPout.columns,index=dfPout.index)
    df_ison = mc._dfDeviceIsOn.unstack(0)
    for dev in devices_elout:
        otherdevs = [d for d in devices_elout if d!=dev]
        cap_avail = 0
        for d in otherdevs:
            devmodel = model.paramDevice[d]['model']
            maxValue = model.paramDevice[d]['Pmax']
            if 'profile' in model.paramDevice[d]:
                extprofile = model.paramDevice[d]['profile']
                maxValue = maxValue*mc._df_profiles_actual[extprofile]
            ison = 1
            if devmodel in ['gasturbine']:
                ison = df_ison[d]
            cap_avail += ison*maxValue
        otherdevs_in = [d for d in devices_elin if d!=dev]
        #TODO: include sheddable load
        res_dev[dev] = cap_avail-dfPout[otherdevs].sum(axis=1)
    return res_dev,dfPout

def plotElBackup2(mc,filename=None):
    '''plot reserve capacity vs device power output'''
    res_dev,dfP = recompute_elBackup(mc)
    plt.figure(figsize=(12,4))
    ax = plt.gca()
    # default line zorder=2
    res_dev.plot(ax=ax,linestyle='-',legend=True,alpha=0.5)
#    # critical is the largest load:
#    c_critical = dfP.idxmax(axis=1)
    # critical is the smallest reserve margin:
    c_critical = (res_dev-dfP).idxmin(axis=1)
#    pd.Series(res_dev.lookup(res_dev.index,c_critical)).plot(ax=ax,
#             linestyle='-',linewidth=2,label="CRITICAL",
#             zorder=2.1,legend=True,color="black")
    (res_dev-dfP).min(axis=1).plot(ax=ax,linestyle="--",
                                   color="black",linewidth=2)

    #use the same colors
    plt.gca().set_prop_cycle(None)
    dfP.plot(ax=ax,linestyle=':',legend=False,alpha=0.5)

#    pd.Series(dfP.lookup(dfP.index,c_critical)).plot(ax=ax,
#             linestyle=':',legend=False,linewidth=2,
#             zorder=2.1,color="black")
    plt.title("Reserve capacity (solid line) vs device output (dotted line)")
#    leg = ax.get_legend()
#    leg.set_bbox_to_anchor((1.01,0))
#    leg.set_frame_on(False)
    labels=list(dfP.columns) + ['MARGIN']
    ax.legend(labels,loc='lower left', bbox_to_anchor =(1.01,0),
               frameon=False)
#    plt.legend(dfP.columns,loc='lower left', bbox_to_anchor =(1.01,0),
#               frameon=False)
    if filename is not None:
        plt.savefig(filename,bbox_inches = 'tight')
