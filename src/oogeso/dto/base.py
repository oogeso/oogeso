from typing import List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, Field, Extra

from oogeso.dto.types import CarrierType, ModelType


class StartStopData(BaseModel, extra=Extra.forbid):
    is_on_init: bool = False  # Initial on/off status
    penalty_start: float = 0  # Startup "cost"
    penalty_stop: float = 0  # Shutdown "cost"
    delay_start_minutes: int = 0  # Delay in minutes from activation to online
    minimum_time_on_minutes: float = 0  # Minimum on-time in minutes once started
    minimum_time_off_minutes: float = 0  # Minimum off-time in minutes once stopped


class TimeSeriesData(BaseModel, extra=Extra.forbid):
    id: str
    data: List[float]
    data_nowcast: Optional[List[float]] = None


class OptimisationParametersData(BaseModel, extra=Extra.forbid):
    # name of objective function to use:
    objective: str
    # minutes per timestep:
    time_delta_minutes: int
    # timesteps in each rolling optimisation:
    planning_horizon: int
    # timesteps between each optimisation:
    optimisation_timesteps: int
    # timesteps beyond which forecast (instead of nowcast) profile is used:
    forecast_timesteps: int
    # costs for co2 emissions (currency/kgCO2)
    co2_tax: Optional[float] = None
    # limit on allowable emission intensity (kgCO2/Sm3oe), -1=no limit
    emission_intensity_max: Optional[float] = -1
    # limit on allowable emission intensity (kgCO2/hour), -1= no limit
    emission_rate_max: Optional[float] = -1
    # how to represent piecewise linear constraints:
    piecewise_repn: str = "SOS2"
    # specify which data to return from simulation as a list. None=return all data
    optimisation_return_data: Optional[List[str]] = None


class CarrierData(BaseModel, extra=Extra.forbid):
    id: str


class DeviceData(BaseModel, extra=Extra.forbid):  # Parent class - use subclasses instead
    id: str  # unique identifier
    node_id: str  # reference to node identifier
    name: str = ""
    include: Optional[bool] = True
    profile: Optional[str] = None  # reference to time-series
    flow_min: Optional[float] = None  # Energy or fluid flow limit
    flow_max: Optional[float] = None
    # Ramp rates are given as change relative to capacity per minute, 1=100%/min:
    max_ramp_down: Optional[float] = None
    max_ramp_up: Optional[float] = None
    start_stop: Optional[StartStopData] = None
    reserve_factor: float = 0  # contribution to electrical spinning reserve
    op_cost: Optional[float] = None
    # Penalty may be fuel, emissions, cost and combinations of these
    penalty_function: Optional[Tuple[List[float], List[float]]] = None
    model: ModelType


class EdgeData(BaseModel, extra=Extra.forbid):  # Base model, use implementations below
    id: str
    node_from: str
    node_to: str
    length_km: Optional[float] = None
    flow_max: float = None  # Maximum flow (MW or Sm3/s)
    bidirectional: Optional[bool] = True
    include: bool = True  # whether to include object in problem formulation
    carrier: CarrierType


class NodeData(BaseModel, extra=Extra.forbid):
    # unique identifier:
    id: str


class SimulationResult(BaseModel, extra=Extra.forbid, arbitrary_types_allowed=True):
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
    device_flow: Optional[pd.Series] = Field(default_factory=pd.Series)
    # Device startup preparation status (boolean):
    device_is_prep: Optional[pd.Series] = Field(default_factory=pd.Series)
    # Device on/off status (boolean):
    device_is_on: Optional[pd.Series] = Field(default_factory=pd.Series)
    # Device starting status (boolean):
    device_starting: Optional[pd.Series] = Field(default_factory=pd.Series)
    # Device stopping status (boolean):
    device_stopping: Optional[pd.Series] = Field(default_factory=pd.Series)
    # Energy storage filling level (Sm3 or MJ)
    device_storage_energy: Optional[pd.Series] = Field(default_factory=pd.Series)
    # Max available "flow" (power/fluid) from storage (Sm3/s or MW):
    device_storage_pmax: Optional[pd.Series] = Field(default_factory=pd.Series)
    # Device assosiated penalty rate (PENALTY_UNIT/s):
    penalty: Optional[pd.Series] = Field(default_factory=pd.Series)
    # Flow rate (Sm3/s or MW):
    edge_flow: Optional[pd.Series] = Field(default_factory=pd.Series)
    # Loss rate (MW) - only relevant for energy flow (el and heat):
    edge_loss: Optional[pd.Series] = Field(default_factory=pd.Series)
    # Voltage angle at node - only relevant for electricity floc computed via dc-pf:
    el_voltage_angle: Optional[pd.Series] = Field(default_factory=pd.Series)
    # Pressure at node (MPa):
    terminal_pressure: Optional[pd.Series] = Field(default_factory=pd.Series)
    # Direct flow between in and out terminal of node - relevant if there is no device inbetween:
    terminal_flow: Optional[pd.Series] = Field(default_factory=pd.Series)
    # Emission rate (sum of all devices) (kgCO2/s):
    co2_rate: Optional[pd.Series] = Field(default_factory=pd.Series)
    # Emission rate per device (kgCO2/s):
    co2_rate_per_dev: Optional[pd.Series] = Field(default_factory=pd.Series)
    # Revenue rate for exported oil/gas (CURRENCY/s):
    export_revenue: Optional[pd.Series] = Field(default_factory=pd.Series)
    # CO2 intensity of exported oil/gas (kgCO2/Sm3oe):
    co2_intensity: Optional[pd.Series] = Field(default_factory=pd.Series)
    # Available online electrical reserve capacity (MW):
    el_reserve: Optional[pd.Series] = Field(default_factory=pd.Series)
    # Available online electrical backup per device (MW):
    el_backup: Optional[pd.Series] = Field(default_factory=pd.Series)
    # Value of duals (associated with constraints)
    duals: Optional[pd.Series] = Field(default_factory=pd.Series)
    # Time-series profiles used in simulation (copied from the input)
    profiles_forecast: Optional[pd.DataFrame] = Field(default_factory=pd.DataFrame)
    profiles_nowcast: Optional[pd.DataFrame] = Field(default_factory=pd.DataFrame)

    def append_results(self, sim_res):
        exclude_list = ["df_profiles_forecast", "df_profiles_forecast"]
        for field_name in self.__fields__:
            if field_name not in exclude_list:
                my_df = getattr(self, field_name)
                other_df = getattr(sim_res, field_name)
                if other_df is not None:
                    new_df = pd.concat([my_df, other_df]).sort_index()
                    if isinstance(new_df.index[0], tuple):  # Fix for multi-index DataFrame and Series
                        new_df.index = pd.MultiIndex.from_tuples(new_df.index)
                    new_df.index.names = other_df.index.names
                    setattr(self, field_name, new_df)


class EnergySystemData(BaseModel, extra=Extra.forbid):
    parameters: OptimisationParametersData
    carriers: List[CarrierData]
    nodes: List[NodeData]
    edges: List[EdgeData]
    devices: List[DeviceData]
    profiles: List[TimeSeriesData]
