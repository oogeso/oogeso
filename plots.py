#import plotly.plotly as py
import plotly
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd

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
    if ((not profiles is None) and (not pd.isna(curve))):
        (profiles['actual'][curve]*devPmax).plot(ax=ax,linestyle='--')
        ax.set_prop_cycle(None)
        (profiles['forecast'][curve]*devPmax).plot(ax=ax,linestyle=":")
    plt.xlim(df.index.min(),df.index.max())
    ax.legend(loc='lower left', bbox_to_anchor =(1.01,0),
              frameon=False)
   