"""This module contains methods to read and save to file"""

import yaml
import pandas as pd
import numpy as np
import logging
from core import electricalsystem
from core import milp_compute


def _convert_xls_input(df,columns,index_col='id'):
    '''Convert from XLSX format input to flat DataFrame'''
    df[columns] = df[columns].fillna(method='ffill')
    if index_col is None:
        df.reset_index(drop=True)

    df2=df[[index_col,'param_id','param_value']].set_index([index_col,'param_id'])
    df2=df2.squeeze().unstack()
    df2=df2.dropna(axis=1,how='all')
    df4=df[columns].set_index(index_col)
    # drop duplicate (indices):
    df4=df4.loc[~df4.index.duplicated(keep='first')]
    df5 = df4.join(df2)
    return df5



def read_data_from_yaml(filename):
    """Read input data from yaml file"""
    data_dict = None
    with open(filename,"r") as text_file:
        data_dict = yaml.safe_load(text_file)
    #data = create_initdata(data_dict=data_dict)
    return data_dict

def read_profiles_from_hd5(filename,key_actual="actual",key_forecast="forecast"):
    """Read input data profiles from HDF5 file"""
    profiles = {}
    profiles['actual'] = pd.read_hdf(filename, key=key_actual)
    profiles['forecast'] = pd.read_hdf(filename, key=key_forecast)
    return profiles

def save_profiles_to_hd5(filename,profiles):
    """Save profiles to HDF5"""
    for k in profiles:
        profiles[k].to_hdf(filename,key=k,mode="a")
    return

def read_profiles_from_xlsx(filename,
        sheet_forecast="profiles",sheet_nowcast="profiles_forecast",
        exclude_cols=[]):
    """Read input data profiles from XLSX to a dicitonary of pandas dataframes"""
    df_profiles = pd.read_excel(filename,sheet_name=sheet_forecast,
        index_col="timestep",
        usecols=lambda col: col not in exclude_cols)
    df_profiles_forecast = pd.read_excel(filename,
        sheet_name=sheet_nowcast,index_col="timestep",
        usecols=lambda col: col not in exclude_cols)
    profiles = {'actual':df_profiles,'forecast':df_profiles_forecast}
    return profiles

def read_data_from_xlsx(filename,includeforecast=True):
    """Read input data from spreadsheet.

    Parameters
    ----------
    filename : str
        name of file
    includeForecast : boolean
        wheter to read profiles from the same file
    """
    dfs = {}
    dfs['node'] = _convert_xls_input(pd.read_excel(filename,sheet_name="node"),
        columns=['id','name'],index_col='id')
    dfs['edge'] = _convert_xls_input(pd.read_excel(filename,sheet_name="edge"),
        columns=['id','include','nodeFrom','nodeTo','type',
        'length_km'],index_col='id')
    dfs['device'] = _convert_xls_input(pd.read_excel(filename,sheet_name="device"),
        columns=['id','include','node','model','name'],index_col='id')
    dfs['parameters'] = pd.read_excel(filename,sheet_name="parameters",index_col=0)
    dfs['parameters'].rename(
        columns={'param_id':'id','param_value':'value'},inplace=True)
    dfs['carriers'] = _convert_xls_input(pd.read_excel(
        filename,sheet_name="carriers"),
        columns=['id'],index_col='id')

    data = create_initdata(dfs=dfs)
    if includeforecast:
        profiles = read_profiles_from_xlsx(filename)
        return data, profiles
    return data


def _nodeCarrierHasSerialDevice(df_node,df_device):
    #devmodel_inout = device_models #Multicarrier.devicemodel_inout()
    devmodel_inout = milp_compute.devicemodel_inout()
    node_devices = df_device.groupby('node').groups

    # extract carriers (from defined device models)
    sublist = [v['in']+v['out'] for k,v in devmodel_inout.items()]
    flatlist = [item for sublist2 in sublist for item in sublist2]
    allCarriers = set(flatlist)
    node_carrier_has_serialdevice = {}
    for n in df_node.index:
        devs_at_node=[]
        if n in node_devices:
            devs_at_node = node_devices[n]
        node_carrier_has_serialdevice[n] = {}
        for carrier in allCarriers:
            # default to false:
            node_carrier_has_serialdevice[n][carrier] = False
            for dev_mod in df_device.loc[devs_at_node,'model']:
                #has_series = ((carrier in devmodel_inout[dev_mod]['in'])
                #                and (carrier in devmodel_inout[dev_mod]['out']))
                if (('serial' in devmodel_inout[dev_mod]) and
                            (carrier in devmodel_inout[dev_mod]['serial'])):
                    node_carrier_has_serialdevice[n][carrier] = True
                    break
    return node_carrier_has_serialdevice

def _oogeso_dict_to_df(data_dict):
    '''convert dict to dataframes for further processing in create_initdata'''
    def d2d(key):
        '''convert from dict to dataframe, makins sure to not miss indices
        without parameters'''
        return pd.DataFrame.from_dict(data_dict[key],orient="index"
                    ).reindex(data_dict[key].keys())

    dfs = {}
    dfs['parameters'] = pd.DataFrame.from_dict(
        data_dict['paramParameters'],orient='index',columns=['value'])
    dfs['carriers'] = d2d('paramCarriers')
    dfs['node'] = d2d('paramNode')
    dfs['edge'] = d2d('paramEdge')
    dfs['device'] = d2d('paramDevice')
    return dfs


def to_dict_dropna(df):
  return {k:r.dropna().to_dict() for k,r in df.iterrows()}


def create_initdata(data_dict=None,dfs=None):
    """Convert input data to data structure required by pyomo model

    Parameters
    ----------
    data_dict : dict
        Input data in python dict format
    dfs : dict of Pandas dataframe
        Input data in pandas format

    Either the dict or the dataframe input argument should be provided

    Returns
    -------
    data : dict
        Model data in a dictionary mirroring the Pyomo model, and used used
        when creating a model instance
    """

    if data_dict is not None:
        if dfs is not None:
            raise Exception("Provide only one of data_dict and dfs parameters")
        dfs = _oogeso_dict_to_df(data_dict)

    # Use device id as name if name is missing
    dfs['device']['name'].fillna(pd.Series(
        data=dfs['device'].index, index=dfs['device'].index),inplace=True)

    df_node = dfs['node']
    df_edge= dfs['edge']
    df_device= dfs['device']
    df_carriers= dfs['carriers']
    df_parameters= dfs['parameters']

    # default values if missing from input:
    if 'height_m'  in df_edge:
        df_edge['height_m'] = df_edge['height_m'].fillna(0)
    else:
        df_edge['height_m']=0

    # discard edges and devices not to be included:
    df_edge = df_edge[df_edge['include']==1]
    df_device = df_device[df_device['include']==1]


    if not 'profile' in df_device:
        df_device['profile'] = np.nan

    # Set node terminal nominal pressure based on edge from/to pressure values
    for i,edg in df_edge.iterrows():
        if 'pressure.from' not in edg:
            continue
        if np.isnan(edg['pressure.from']):
            continue
        n_from=edg['nodeFrom']
        n_to =edg['nodeTo']
        typ=edg['type']
        p_from=edg['pressure.from']
        #m_from=(df_node.index==n_from)
        c_out='pressure.{}.out'.format(typ)
        p_to=edg['pressure.to']
        #m_to=(df_node.index==n_to)
        c_in='pressure.{}.in'.format(typ)
        # Check that pressure values are consistent:
        is_consistent = True
        existing_p_from=None
        existing_p_to=None
        try:
            existing_p_from = df_node.loc[n_from,c_out]
            logging.debug("{} p_from = {} (existing) / {} (new)"
                         .format(i,existing_p_from,p_from))
            if ((not np.isnan(existing_p_from)) and (existing_p_from!=p_from)):
                msg =("Input data edge pressure from values are"
                      " inconsistent (edge={}, {}!={})"
                      ).format(i,existing_p_from,p_from)
                is_consistent=False
        except:
            pass
        try:
            existing_p_to = df_node.loc[n_to,c_in]
            logging.debug("{} p_to = {} (existing) / {} (new)"
                         .format(i,existing_p_to,p_to))
            if ((not np.isnan(existing_p_to)) and (existing_p_to!=p_to)):
                msg =("Input data edge pressure to values are"
                      " inconsistent (edge={})").format(i)
                is_consistent=False
        except:
            pass
        if not is_consistent:
            print(df_node)
            raise Exception(msg)

        df_node.loc[n_from,c_out] = p_from
        df_node.loc[n_to,c_in] = p_to

    carrier_properties = to_dict_dropna(df_carriers)
    allCarriers = list(carrier_properties.keys())

    node_carrier_has_serialdevice = _nodeCarrierHasSerialDevice(
        df_node,df_device)

    # # gas pipeline parameters - derive k and exp(s) parameters:
    # ga=carrier_properties['gas']
    # temp = df_edge['temperature_K']
    # height_difference = df_edge['height_m']
    # s = 0.0684 * (ga['G_gravity']*height_difference
    #                 /(temp*ga['Z_compressibility']))
    # sfactor= (np.exp(s)-1)/s
    # sfactor.loc[s==0] = 1
    # length = df_edge['length_km']*sfactor
    # diameter = df_edge['diameter_mm']
    #
    # gas_edge_k = (4.3328e-8*ga['Tb_basetemp_K']/ga['Pb_basepressure_MPa']
    #     *(ga['G_gravity']*temp*length*ga['Z_compressibility'])**(-1/2)
    #     *diameter**(8/3))
    # df_edge['gasflow_k'] = gas_edge_k
    # df_edge['exp_s'] = np.exp(s)

    coeffB,coeffDA = electricalsystem.computePowerFlowMatrices(
        df_node,df_edge,baseZ=1)
    planning_horizon = df_parameters.loc['planning_horizon','value']
    data = {}
    data['setCarrier'] = {None:allCarriers}
    data['setNode'] = {None:df_node.index.tolist()}
    data['setEdge'] = {None:df_edge.index.tolist()}
    data['setEdgeWithFlowComponents'] = {
        None:df_edge[df_edge['type']=='wellstream'].index.tolist()}
    data['setDevice'] = {None:df_device.index.tolist()}
    #data['setDevicemodel'] = {None:Multicarrier.devicemodel_inout().keys()}
    data['setHorizon'] = {None:range(planning_horizon)}
    data['setParameters'] = {None:df_parameters.index.tolist()}
    data['setProfile'] = {None:df_device['profile'].dropna().unique().tolist()}
    data['paramNode'] = to_dict_dropna(df_node)
    data['paramNodeCarrierHasSerialDevice'] = node_carrier_has_serialdevice
    data['paramNodeDevices'] = df_device.groupby('node').groups
#    data['paramDevice'] = df_device.to_dict(orient='index')
    data['paramDevice'] = to_dict_dropna(df_device)
    data['paramDeviceIsOnInitially'] = {k:v['isOn_init']
        for k,v in data['paramDevice'].items() if 'isOn_init' in v }
    data['paramDevicePowerInitially'] = {k:v['P_init']
        for k,v in data['paramDevice'].items() if 'P_init' in v }
    data['paramDeviceEnergyInitially'] = {k:v['E_init']
        for k,v in data['paramDevice'].items() if 'E_init' in v }
    data['paramEdge'] = to_dict_dropna(df_edge)
    data['paramNodeEdgesFrom'] = df_edge.groupby(['type','nodeFrom']).groups
    data['paramNodeEdgesTo'] = df_edge.groupby(['type','nodeTo']).groups
    #data['paramDevicemodel'] = devmodel_inout
    data['paramParameters'] = df_parameters['value'].to_dict()#orient='index')
    #unordered set error - but is this needed - better use dataframe diretly instead?
#    data['paramProfiles'] = df_profiles_forecast.loc[
#            range(planning_horizon),data['setProfile'][None]
#            ].T.stack().to_dict()
    data['paramCarriers'] = carrier_properties
    data['paramCoeffB'] = coeffB
    data['paramCoeffDA'] = coeffDA
    return data
