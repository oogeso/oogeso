from typing import Callable

from oogeso import dto
from oogeso.core import devices, networks


def get_network_from_carrier_name(carrier_name: str) -> Callable:
    map_carrier_to_model = {
        "el": networks.ElNetwork,
        "gas": networks.GasNetwork,
        "oil": networks.OilNetwork,
        "water": networks.WaterNetwork,
        "wellstream": networks.WellStreamNetwork,
        "heat": networks.HeatNetwork,
        "hydrogen": networks.HydrogenNetwork,
    }
    if not carrier_name.lower() in map_carrier_to_model.keys():
        raise NotImplementedError(f"Carrier: {carrier_name} has not implemented a corresponding Network mapping")
    return map_carrier_to_model[carrier_name.lower()]


def get_device_from_model_name(model_name: str) -> Callable:
    map_device_name_to_class = {
        "powersource": devices.PowerSource,
        "powersink": devices.PowerSink,
        "storageel": devices.StorageEl,
        "compressorel": devices.CompressorEl,
        "compressorgas": devices.CompressorGas,
        "electrolyser": devices.Electrolyser,
        "fuelcell": devices.FuelCell,
        "gasheater": devices.GasHeater,
        "gasturbine": devices.GasTurbine,
        "heatpump": devices.HeatPump,
        "pumpoil": devices.PumpOil,
        "pumpwater": devices.PumpWater,
        "separator": devices.Separator,
        "separator2": devices.Separator2,
        "sinkel": devices.SinkEl,
        "sinkheat": devices.SinkHeat,
        "sinkgas": devices.SinkGas,
        "sinkoil": devices.SinkOil,
        "sinkwater": devices.SinkWater,
        "sourceel": devices.SourceEl,
        "sourcegas": devices.SourceGas,
        "sourceoil": devices.SourceOil,
        "sourcewater": devices.SourceWater,
        "storagehydrogen": devices.StorageHydrogen,
        "wellgaslift": devices.WellGasLift,
        "wellproduction": devices.WellProduction,
    }
    if model_name in map_device_name_to_class:
        return map_device_name_to_class[model_name]
    else:
        raise NotImplementedError(f"Device {model_name} has not been implemented.")


def get_device_data_class_from_str(model_name: str) -> Callable:
    model_name = model_name.replace("_", "")
    map_device_name_to_class = {
        "powersource": dto.DevicePowerSourceData,
        "powersink": dto.DevicePowerSinkData,
        "storageel": dto.DeviceStorageElData,
        "compressorel": dto.DeviceCompressorElData,
        "compressorgas": dto.DeviceCompressorGasData,
        "electrolyser": dto.DeviceElectrolyserData,
        "fuelcell": dto.DeviceFuelCellData,
        "gasheater": dto.DeviceGasHeaterData,
        "gasturbine": dto.DeviceGasTurbineData,
        "heatpump": dto.DeviceHeatPumpData,
        "pumpoil": dto.DevicePumpOilData,
        "pumpwater": dto.DevicePumpWaterData,
        "separator": dto.DeviceSeparatorData,
        "separator2": dto.DeviceSeparator2Data,
        "sinkel": dto.DeviceSinkElData,
        "sinkheat": dto.DeviceSinkHeatData,
        "sinkgas": dto.DeviceSinkGasData,
        "sinkoil": dto.DeviceSinkOilData,
        "sinkwater": dto.DeviceSinkWaterData,
        "sourceel": dto.DeviceSourceElData,
        "sourcegas": dto.DeviceSourceGasData,
        "sourceoil": dto.DeviceSourceOilData,
        "sourcewater": dto.DeviceSourceWaterData,
        "storagehydrogen": dto.DeviceStorageHydrogenData,
        "wellgaslift": dto.DeviceWellGasLiftData,
        "wellproduction": dto.DeviceWellProductionData,
    }
    if model_name in map_device_name_to_class:
        return map_device_name_to_class[model_name]
    else:
        raise NotImplementedError(f"Device data class for {model_name} has not been implemented.")


def get_carrier_data_class_from_str(model_name: str) -> Callable:
    map_carrier_name_to_class = {
        "el": dto.CarrierElData,
        "gas": dto.CarrierGasData,
        "oil": dto.CarrierOilData,
        "water": dto.CarrierWaterData,
        "hydrogen": dto.CarrierHydrogenData,
        "heat": dto.CarrierHeatData,
        "wellstream": dto.CarrierWellStreamData,
    }
    if model_name in map_carrier_name_to_class:
        return map_carrier_name_to_class[model_name]
    else:
        raise NotImplementedError(f"Carrier data class for {model_name} has not been implemented.")


def get_edge_data_class_from_str(carrier_name: str) -> Callable:
    map_edge_name_to_class = {
        "el": dto.EdgeElData,
        "gas": dto.EdgeGasData,
        "oil": dto.EdgeOilData,
        "water": dto.EdgeWaterData,
        "hydrogen": dto.EdgeHydrogenData,
        "heat": dto.EdgeHeatData,
        "wellstream": dto.EdgeWellstreamData,
    }
    if carrier_name in map_edge_name_to_class:
        return map_edge_name_to_class[carrier_name]
    else:
        raise NotImplementedError(f"Edge data class for {carrier_name} has not been implemented.")
