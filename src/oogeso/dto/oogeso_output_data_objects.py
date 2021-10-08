import pandas as pd
from dataclasses import dataclass, fields


@dataclass
class SimulationResult:
    """Results from oogeso simulation

    The results are stored in a set of multi-index Dataframes, with
    index names indicating what they are:

    device - device identifier
    node - node identifier
    edge - edge identifier
    carrier - network type ("el", "gas", "oil", "water", "hydrogen", "heat")
    terminal - input/output ("in" or "out"),
    time (integer timestep)
    """

    # Input/output flow per device and network type:
    dfDeviceFlow: pd.DataFrame = None
    # Device startup preparation status (boolean):
    dfDeviceIsPrep: pd.DataFrame = None
    # Device on/off status (boolean):
    dfDeviceIsOn: pd.DataFrame = None
    # Device starting status (boolean):
    dfDeviceStarting: pd.DataFrame = None
    # Device stopping status (boolean):
    dfDeviceStopping: pd.DataFrame = None
    # Energy storage filling level (Sm3 or MJ)
    dfDeviceStorageEnergy: pd.DataFrame = None
    # Max available "flow" (power/fluid) from storage (Sm3/s or MW):
    dfDeviceStoragePmax: pd.DataFrame = None
    # Device assosiated penalty rate (some unit per sec):
    dfPenalty: pd.DataFrame = None
    # Flow rate (Sm3/s or MW):
    dfEdgeFlow: pd.DataFrame = None
    # Loss rate (MW) - only relevant for energy flow (el and heat):
    dfEdgeLoss: pd.DataFrame = None
    # Voltage angle at node - only relevant for electricity floc computed via dc-pf:
    dfElVoltageAngle: pd.DataFrame = None
    # Pressure at node (MPa):
    dfTerminalPressure: pd.DataFrame = None
    # Direct flow between in and out terminal of node - relevant if there is no device inbetween:
    dfTerminalFlow: pd.DataFrame = None
    # Emission rate (sum of all devices) (kgCO2/s):
    dfCO2rate: pd.DataFrame = None
    # Emission rate per device (kgCO2/s):
    dfCO2rate_per_dev: pd.DataFrame = None
    # Revenue rate for exported oil/gas (CURRENCY/s):
    dfExportRevenue: pd.DataFrame = None
    # CO2 intensity of exported oil/gas (kgCO2/Sm3oe):
    dfCO2intensity: pd.DataFrame = None
    # Available online electrical reserve capacity (MW):
    dfElReserve: pd.DataFrame = None
    # Available online electrical backup margin (MW):
    dfElBackup: pd.DataFrame = None
    # Value of duals (associated with constraints)
    dfDuals: pd.DataFrame = None
    # Time-series profiles used in simulation (copied from the input)
    df_profiles_forecast: pd.DataFrame = None
    df_profiles_nowcast: pd.DataFrame = None

    def append_results(self, sim_res):
        exclude_list = ["optimiser", "df_profiles_forecast", "df_profiles_forecast"]
        for my_field in fields(self):
            field_name = my_field.name
            if field_name not in exclude_list:
                my_df = getattr(self, field_name)
                other_df = getattr(sim_res, field_name)
                if other_df is not None:
                    setattr(self, field_name, pd.concat([my_df, other_df]).sort_index())
