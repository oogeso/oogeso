#import plotly.plotly as py
import plotly
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pydot

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

def plot_deviceprofile(mc,devs,profiles=None,filename=None):
    '''plot forecast and actual profile (available power), and device output'''
    if type(devs) is not list:
        devs = [devs]
    plt.figure(figsize=(12,4))
    ax=plt.gca()
    labels=[]
    for dev in devs:
        dev_param = mc.instance.paramDevice[dev]
        devname = "{}:{}".format(dev,dev_param["name"])
        devPmax = dev_param['Pmax']
        df= mc._dfDevicePower.unstack(0)[dev]
        df.name = devname
        df.plot(ax=ax)
        labels=labels+[devname]
        if 'profile' in dev_param:
            curve = dev_param['profile']
#        if ((not profiles is None) and (not pd.isna(curve))):
            (profiles['actual'][curve]*devPmax).plot(ax=ax,linestyle='--')
            ax.set_prop_cycle(None)
            (profiles['forecast'][curve]*devPmax).plot(ax=ax,linestyle=":")
            labels = labels+['--actual','--forecast']
        if dev_param['model']=='gasturbine':
            df2=mc._dfDeviceIsOn.unstack(0)[dev]
            df2.plot(ax=ax,linestyle='-.')
            labels = labels+['--online']
    plt.xlim(df.index.min(),df.index.max())
    ax.legend(labels,loc='lower left', bbox_to_anchor =(1.01,0),
              frameon=False)
    if filename is not None:
        plt.savefig(filename,bbox_inches = 'tight')
    return ax

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
    ax.set_ylim(0,dev_param['Pmax'])
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
        if dev_param['model'] in ['storage_el']:
            ax2.set_ylim(0,dev_param['Emax'])
        elif dev_param['model'] in ['well_injection']:
            ax2.set_ylim(-dev_param['Emax']/2,dev_param['Emax']/2)
        #ax2.tick_params(axis='y', labelcolor="red")
        ax2.legend(loc='upper right')
    ax.set_xlim(tmin,tmax)

    if filename is not None:
        plt.savefig(filename,bbox_inches = 'tight')

def plot_SumPowerMix(mc,carrier,filename=None,reverseLegend=True):
    fig,axes = plt.subplots(nrows=2,ncols=1,figsize=(12,8))
    #plt.figure(figsize=(12,4))
    plt.suptitle("Sum power ({})".format(carrier))
    
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
    dfF_in = dfF[mask_carrier&mask_in]
    dfF_in.index = dfF_in.index.droplevel(level=("carrier","terminal"))
    dfF_in = dfF_in.unstack(0)
    dfF_in.rename(columns={d:"{}:{}".format(d,mc.instance.paramDevice[d]['name']) 
        for d in dfF_in.columns},inplace=True)
    dfF_out.rename(columns={d:"{}:{}".format(d,mc.instance.paramDevice[d]['name']) 
        for d in dfF_out.columns},inplace=True)
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

def plot_ExportRevenue(mc,filename=None):
    plt.figure(figsize=(12,4))
    plt.title("Export price ($/hour)")
    ax=plt.gca()
    ax.set_ylabel("$/hour")
    ax.set_xlabel("Timestep")
    (mc._dfExportRevenue.loc[:,mc._dfExportRevenue.sum()>0]).plot.area(
            ax=ax,linewidth=0)
    if filename is not None:
        plt.savefig(filename,bbox_inches = 'tight')


def plot_CO2_rate(mc,filename=None):
    plt.figure(figsize=(12,4))
    plt.title("CO2 emission rate (kgCO2/hour)")
    ax=plt.gca()
    ax.set_ylabel("kgCO2/hour")
    ax.set_xlabel("Timestep")
    mc._dfCO2rate.plot()
    if filename is not None:
        plt.savefig(filename,bbox_inches = 'tight')

def plot_CO2rate_per_dev(mc,filename=None,reverseLegend=False):
    df_info = pd.DataFrame.from_dict(dict(mc.instance.paramDevice.items())).T
    labels = (df_info.index.astype(str) +'_'+df_info['name'])

    plt.figure(figsize=(12,4))
    ax=plt.gca()
    ax.set_ylabel("Emission rate (kgCO2/hour)")
    ax.set_xlabel("Timestep")
    mc._dfCO2rate_per_dev.loc[:,~(mc._dfCO2rate_per_dev==0).all()
                    ].rename(columns=labels
                    ).plot.area(ax=ax,linewidth=0)
    
    if reverseLegend:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1],
                  loc='lower left', bbox_to_anchor =(1.01,0),frameon=False)
    else:
        ax.legend(loc='lower left', bbox_to_anchor =(1.01,0),frameon=False)
        
    if filename is not None:
        plt.savefig(filename,bbox_inches = 'tight')

def plot_CO2_intensity(mc,filename=None):
    plt.figure(figsize=(12,4))
    plt.title("CO2 intensity (kgCO2/Sm3oe)")
    ax=plt.gca()
    ax.set_ylabel("kgCO2/Sm3oe")
    ax.set_xlabel("Timestep")
    mc._dfCO2intensity.plot()
    if filename is not None:
        plt.savefig(filename,bbox_inches = 'tight')
   

def plotProfiles(profiles,curves=None,filename=None):
    '''Plot profiles (forecast and actual)'''
    
    plt.figure(figsize=(12,4))
    ax=plt.gca()
    if curves is None:
        curves = profiles['actual'].columns
    profiles['actual'][curves].plot(ax=ax)
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
    ax.plot(mc._dfDevicePower.unstack(0)[dev],'-.',label="DevicePower")
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
    

def plotNetwork(mc,timestep=0,filename=None,
                        only_carrier=None,rankdir='LR'):
    """Plot energy network
    
    mc : object
        Multicarrier object
    timestep : int
        which timestep to show values for
    filename : string
        Name of file
    only_carrier : list
        Restrict energy carriers to these types (None=plot all)
    rankdir : str
        Plotting direction TB=top to bottom, LR=left to right
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
    dotG = pydot.Dot(graph_type='digraph',rankdir=rankdir,newrank='false')
    model = mc.instance
    if only_carrier is None:
        carriers = model.setCarrier
    else:
        carriers = [only_carrier]
    
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
                    devlabel = "{} {}".format(d,model.paramDevice[d]['name'])
                    if timestep is not None:
                        p_dev = mc._dfDevicePower[(d,timestep)]
                        devlabel = "{}\n{:.2f}".format(devlabel,p_dev)
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
                            devedgelabel = "{:.2f}".format(f_in)
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
                            devedgelabel = "{:.2f}".format(f_out)
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
                label_in = carrier+'_in'
                label_out= carrier+supp
                if timestep is None:
                    pass
                elif carrier in ['gas','wellstream','oil','water']:
                    label_in +=':{:3.2f}'.format(mc._dfTerminalPressure[(n_id,carrier,'in',timestep)])
                    label_out +=':{:3.2f}'.format(mc._dfTerminalPressure[(n_id,carrier,'out',timestep)])
                elif carrier=='el':
                    label_in +=':{:3.2g}'.format(mc._dfElVoltageAngle[(n_id,timestep)])
                    label_out +=':{:3.2g}'.format(mc._dfElVoltageAngle[(n_id,timestep)])
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
                else:
                    edgelabel = '{:.2f}'.format(mc._dfEdgeFlow[(i,timestep)])
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
        dotG.write_png(filename,prog='dot')    
 
def plotGasTurbineEfficiency(fuelA=2.35,fuelB=0.53, filename=None):
    #fuelA = model.paramDevice[dev]['fuelA']
    #fuelB = model.paramDevice[dev]['fuelB']
    x_pow = pd.np.linspace(0,1,50)
    y_fuel = fuelB + fuelA*x_pow
    plt.figure(figsize=(12,4))
    #plt.suptitle("Gas turbine fuel characteristics")
    
    plt.subplot(1,3,1)
    plt.title("Fuel usage ($P_{gas}/P_{el}^{max}$)")
    plt.xlabel("Electric power output ($P_{el}/P_{el}^{max}$)")
    #plt.ylabel("Gas power input ($P_{gas}/P_{el}^{max}$)")
    plt.plot(x_pow,y_fuel)
    plt.ylim(bottom=0)
    
    plt.subplot(1,3,2)
    plt.title("Specific fuel usage ($P_{gas}/P_{el}$)")
    plt.xlabel("Electric power output ($P_{el}/P_{el}^{max}$)")
    plt.plot(x_pow,y_fuel/x_pow)
    plt.ylim(top=30)
    
    plt.subplot(1,3,3)
    plt.title("Efficiency ($P_{el}/P_{gas}$)")
    plt.xlabel("Electric power output ($P_{el}/P_{el}^{max}$)")
    plt.plot(x_pow,x_pow/y_fuel)
    if filename is not None:
        plt.savefig(filename,bbox_inches = 'tight')        

       
