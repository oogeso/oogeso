from typing import Dict, Union

import pyomo.environ as pyo

from oogeso import dto
from oogeso.core.devices.base import Device


class PowerSink(Device):
    """
    Generic electricity consumption
    """

    carrier_in = ["el"]
    carrier_out = []
    serial = []

    def __init__(
        self,
        dev_data: Union[dto.DevicePowerSinkData, dto.DeviceSinkElData],
        carrier_data_dict: Dict[str, dto.CarrierElData],
    ):
        super().__init__(dev_data=dev_data, carrier_data_dict=carrier_data_dict)
        self.dev_data = dev_data
        self.id = dev_data.id
        self.carrier_data = carrier_data_dict

    def get_flow_var(self, pyomo_model: pyo.Model, t: int):
        return pyomo_model.varDeviceFlow[self.id, "el", "in", t]


# Just another name for PowerSink
class SinkEl(PowerSink):
    pass


class SinkHeat(Device):
    """
    Generic heat consumption
    """

    carrier_in = ["heat"]
    carrier_out = []
    serial = []

    def __init__(
        self,
        dev_data: dto.DeviceSinkHeatData,
        carrier_data_dict: Dict[str, dto.CarrierHeatData],
    ):
        super().__init__(dev_data=dev_data, carrier_data_dict=carrier_data_dict)
        self.dev_data = dev_data
        self.id = dev_data.id
        self.carrier_data = carrier_data_dict

    def get_flow_var(self, pyomo_model: pyo.Model, t: int):
        return pyomo_model.varDeviceFlow[self.id, "heat", "in", t]


class SinkGas(Device):
    """
    Generic electricity consumption
    """

    carrier_in = ["gas"]
    carrier_out = []
    serial = []

    def __init__(
        self,
        dev_data: dto.DeviceSinkGasData,
        carrier_data_dict: Dict[str, dto.CarrierGasData],
    ):
        super().__init__(dev_data=dev_data, carrier_data_dict=carrier_data_dict)
        self.dev_data = dev_data
        self.id = dev_data.id
        self.carrier_data = carrier_data_dict

    def get_flow_var(self, pyomo_model: pyo.Model, t: int):
        return pyomo_model.varDeviceFlow[self.id, "gas", "in", t]


class SinkOil(Device):
    """
    Generic oil consumption
    """

    carrier_in = ["oil"]
    carrier_out = []
    serial = []

    def __init__(
        self,
        dev_data: dto.DeviceSinkOilData,
        carrier_data_dict: Dict[str, dto.CarrierOilData],
    ):
        super().__init__(dev_data=dev_data, carrier_data_dict=carrier_data_dict)
        self.dev_data = dev_data
        self.id = dev_data.id
        self.carrier_data = carrier_data_dict

    def get_flow_var(self, pyomo_model: pyo.Model, t: int):
        return pyomo_model.varDeviceFlow[self.id, "oil", "in", t]


class SinkWater(Device):
    """
    Generic water consumption
    """

    carrier_in = ["water"]
    carrier_out = []
    serial = []

    def __init__(
        self,
        dev_data: dto.DeviceSinkWaterData,
        carrier_data_dict: Dict[str, dto.CarrierWaterData],
    ):
        super().__init__(dev_data=dev_data, carrier_data_dict=carrier_data_dict)
        self.dev_data = dev_data
        self.id = dev_data.id
        self.carrier_data = carrier_data_dict

    def get_flow_var(self, pyomo_model: pyo.Model, t: int):
        return pyomo_model.varDeviceFlow[self.id, "water", "in", t]


class SinkCarbon(Device):
    """
    CO2 sink (emitted or stored)
    """

    carrier_in = ["carbon"]
    carrier_out = []
    serial = []

    def __init__(
        self,
        dev_data: dto.DeviceSinkCarbonData,
        carrier_data_dict: Dict[str, dto.CarrierCarbonData],
    ):
        super().__init__(dev_data=dev_data, carrier_data_dict=carrier_data_dict)
        self.dev_data = dev_data
        self.id = dev_data.id
        self.carrier_data = carrier_data_dict

    def get_flow_var(self, pyomo_model: pyo.Model, t: int):
        return pyomo_model.varDeviceFlow[self.id, "carbon", "in", t]
