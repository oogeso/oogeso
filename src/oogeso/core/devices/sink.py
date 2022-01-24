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

    def rule_devmodel_sink_water(self, model: pyo.Model, t: int, i: int):
        dev = self.id
        dev_data = self.dev_data
        time_delta_minutes = model.paramTimestepDeltaMinutes

        if dev_data.flow_avg is None:
            return pyo.Constraint.Skip
        if dev_data.max_accumulated_deviation is None:
            return pyo.Constraint.Skip
        if dev_data.max_accumulated_deviation == 0:
            return pyo.Constraint.Skip
        if i == 1:
            # FLEXIBILITY
            # (water_in-water_avg)*dt = delta buffer
            delta_t = time_delta_minutes / 60  # hours
            lhs = (model.varDeviceFlow[dev, "water", "in", t] - dev_data.flow_avg) * delta_t
            if t > 0:
                E_prev = model.varDeviceStorageEnergy[dev, t - 1]
            else:
                E_prev = model.paramDeviceEnergyInitially[dev]
            rhs = model.varDeviceStorageEnergy[dev, t] - E_prev
            return lhs == rhs
        elif i == 2:
            # energy buffer limit
            E_max = dev_data.max_accumulated_deviation
            return pyo.inequality(-E_max / 2, model.varDeviceStorageEnergy[dev, t], E_max / 2)
        else:
            raise Exception("impossible")

    def define_constraints(self, pyomo_model: pyo.Model):
        """Specifies the list of constraints for the device"""
        list_to_reconstruct = super().define_constraints(pyomo_model)

        constr = pyo.Constraint(
            pyomo_model.setHorizon,
            pyo.RangeSet(1, 2),
            rule=self.rule_devmodel_sink_water,
        )
        # add constraints to model:
        setattr(pyomo_model, "constr_{}_{}".format(self.id, "flex"), constr)
        return list_to_reconstruct

    def get_flow_var(self, pyomo_model: pyo.Model, t: int):
        return pyomo_model.varDeviceFlow[self.id, "water", "in", t]
