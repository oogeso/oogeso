from enum import Enum


class CarrierType(str, Enum):
    # Fixe: Ensure that this is independent of core!
    EL = "el"
    OIL = "oil"
    WELLSTREAM = "wellstream"
    WATER = "water"
    HEAT = "heat"
    HYDROGEN = "hydrogen"
    FLUID = "fluid"
    GAS = "gas"


class ModelType(str, Enum):
    POWER_SOURCE = "powersource"
    POWER_SINK = "powersink"
    STORAGE_EL = "storageel"
    SOURCE_EL = "sourceel"
    SOURCE_GAS = "sourcegas"
    SOURCE_OIL = "sourceoil"
    SOURCE_WATER = "sourcewater"
    SINK_EL = "sinkel"
    SINK_HEAT = "sinkheat"
    SINK_GAS = "sinkgas"
    SINK_OIL = "sinkoil"
    SINK_WATER = "sinkwater"
    COMPRESSOR_EL = "compressorel"
    COMPRESSOR_GAS = "compressorgas"
    ELECTROLYSER = "electrolyser"
    FUEL_CELL = "fuelcell"
    GAS_HEATER = "gasheater"
    GAS_TURBINE = "gasturbine"
    PUMP = "pump"
    HEAT_PUMP = "heatpump"
    PUMP_OIL = "pumpoil"
    PUMP_WATER = "pumpwater"
    SEPARATOR = "separator"
    SEPARATOR2 = "separator2"
    STORAGE_HYDROGEN = "storagehydrogen"
    WELL_PRODUCTION = "wellproduction"
    WELL_GAS_LIFT = "wellgaslift"


class PressureMethodType(str, Enum):
    WEYMOUTH = "weymouth"
    DARCY_WEISSBACH = "darcy-weissbach"
