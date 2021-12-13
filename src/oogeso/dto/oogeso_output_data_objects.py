from dataclasses import dataclass, fields
from typing import Optional

import pandas as pd


@dataclass
class SimulationResult:
    """Results from oogeso simulation

    The results are stored in a set of multi-index Series, with
    index names indicating what they are:

    device - device identifier
    node - node identifier
    edge - edge identifier
    carrier - network type ("el", "gas", "oil", "water", "hydrogen", "heat")
    terminal - input/output ("in" or "out"),
    time (integer timestep)
    """

    # Input/output flow per device and network type:
    device_flow: Optional[pd.Series] = None
    # Device startup preparation status (boolean):
    device_is_prep: Optional[pd.Series] = None
    # Device on/off status (boolean):
    device_is_on: Optional[pd.Series] = None
    # Device starting status (boolean):
    device_starting: Optional[pd.Series] = None
    # Device stopping status (boolean):
    device_stopping: Optional[pd.Series] = None
    # Energy storage filling level (Sm3 or MJ)
    device_storage_energy: Optional[pd.Series] = None
    # Max available "flow" (power/fluid) from storage (Sm3/s or MW):
    device_storage_pmax: Optional[pd.Series] = None
    # Device assosiated penalty rate (PENALTY_UNIT/s):
    penalty: Optional[pd.Series] = None
    # Flow rate (Sm3/s or MW):
    edge_flow: Optional[pd.Series] = None
    # Loss rate (MW) - only relevant for energy flow (el and heat):
    edge_loss: Optional[pd.Series] = None
    # Voltage angle at node - only relevant for electricity floc computed via dc-pf:
    el_voltage_angle: Optional[pd.Series] = None
    # Pressure at node (MPa):
    terminal_pressure: Optional[pd.Series] = None
    # Direct flow between in and out terminal of node - relevant if there is no device inbetween:
    terminal_flow: Optional[pd.Series] = None
    # Emission rate (sum of all devices) (kgCO2/s):
    co2_rate: Optional[pd.Series] = None
    # Emission rate per device (kgCO2/s):
    co2_rate_per_dev: Optional[pd.Series] = None
    # Revenue rate for exported oil/gas (CURRENCY/s):
    export_revenue: Optional[pd.Series] = None
    # CO2 intensity of exported oil/gas (kgCO2/Sm3oe):
    co2_intensity: Optional[pd.Series] = None
    # Available online electrical reserve capacity (MW):
    el_reserve: Optional[pd.Series] = None
    # Available online electrical backup per device (MW):
    el_backup: Optional[pd.Series] = None
    # Value of duals (associated with constraints)
    duals: Optional[pd.Series] = None
    # Time-series profiles used in simulation (copied from the input)
    profiles_forecast: Optional[pd.DataFrame] = None
    profiles_nowcast: Optional[pd.DataFrame] = None

    def append_results(self, sim_res):
        exclude_list = ["df_profiles_forecast", "df_profiles_forecast"]
        for my_field in fields(self):
            field_name = my_field.name
            if field_name not in exclude_list:
                my_df = getattr(self, field_name)
                other_df = getattr(sim_res, field_name)
                if other_df is not None:
                    setattr(self, field_name, pd.concat([my_df, other_df]).sort_index())
