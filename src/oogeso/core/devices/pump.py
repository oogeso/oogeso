import logging
from typing import Dict, Union

import pyomo.environ as pyo

from oogeso import dto
from oogeso.core.devices.base import Device
from oogeso.core.networks.network_node import NetworkNode

logger = logging.getLogger(__name__)


class PumpDevice(Device):
    """Parent class for pumps. Don't use this class directly."""

    dev_data: dto.DevicePumpData

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_pump_demand(
        self,
        pyomo_model: pyo.Model,
        carrier: str,
        linear: bool = False,
        Q: float = None,
        p1: float = None,
        p2: float = None,
        t: int = None,
    ):
        dev_data = self.dev_data
        node_id = dev_data.node_id
        node_obj: NetworkNode = self.node
        # power demand vs flow rate and pressure difference
        # see eg. doi:10.1016/S0262-1762(07)70434-0
        # P = Q*(p_out-p_in)/eta
        # units: m3/s*MPa = MW
        #
        # Assuming incompressible fluid, so flow rate m3/s=Sm3/s
        # (this approximation may not be very good for multiphase
        # wellstream)
        # assume nominal pressure and keep only flow rate dependence

        eta = dev_data.eta

        if t is None:
            t = 0
        if Q is None:
            Q = pyomo_model.varDeviceFlow[self.id, carrier, "in", t]
        if p1 is None:
            p1 = pyomo_model.varPressure[node_id, carrier, "in", t]
        if p2 is None:
            p2 = pyomo_model.varPressure[node_id, carrier, "out", t]
        if linear:
            # linearised equations around operating point
            # p1=p10, p2=p20, Q=Q0
            if t == 0:
                logger.debug("Node:{}, nominal pressures={}".format(node_id, node_obj.pressure_nominal))
            p10 = node_obj.pressure_nominal[carrier]["in"]
            p20 = node_obj.pressure_nominal[carrier]["out"]
            delta_p = p20 - p10
            P = Q * delta_p / eta
        else:
            # non-linear equation - for computing outside optimisation
            P = Q * (p2 - p1) / eta
        return P

    def _rules_pump(self, model: pyo.Model, t: int, i: int) -> Union[pyo.Expression, pyo.Constraint.Skip]:
        dev = self.id
        if isinstance(self, PumpOil):
            carrier = "oil"
        elif isinstance(self, PumpWater):
            carrier = "water"
        else:
            raise NotImplementedError(f"Device model {self.dev_data.model} has not been implemented")
        if i == 1:
            # flow out = flow in
            lhs = model.varDeviceFlow[dev, carrier, "out", t]
            rhs = model.varDeviceFlow[dev, carrier, "in", t]
            return lhs == rhs  # noqa
        elif i == 2:
            lhs = model.varDeviceFlow[dev, "el", "in", t]
            rhs = self.compute_pump_demand(
                pyomo_model=model,
                carrier=carrier,
                linear=True,
                t=t,
            )
            return lhs == rhs  # noqa
        else:
            raise ValueError(f"Argument i must be 1 or 2. {i} was given.")

    def define_constraints(self, pyomo_model: pyo.Model):
        """Specifies the list of constraints for the device"""

        list_to_reconstruct = super().define_constraints(pyomo_model)

        constr = pyo.Constraint(pyomo_model.setHorizon, pyo.RangeSet(1, 2), rule=self._rules_pump)
        # add constraint to model:
        setattr(pyomo_model, "constr_{}_{}".format(self.id, "misc"), constr)
        return list_to_reconstruct

    def get_flow_var(self, pyomo_model: pyo.Model, t: int):
        return pyomo_model.varDeviceFlow[self.id, "el", "in", t]


class PumpOil(PumpDevice):
    """Oil pump"""

    carrier_in = ["oil", "el"]
    carrier_out = ["oil"]
    serial = ["oil"]

    def __init__(
        self,
        dev_data: dto.DevicePumpOilData,
        carrier_data_dict: Dict[str, Union[dto.CarrierElData, dto.CarrierOilData]],
    ):
        super().__init__(dev_data=dev_data, carrier_data_dict=carrier_data_dict)
        self.dev_data = dev_data
        self.id = dev_data.id
        self.carrier_data = carrier_data_dict


class PumpWellStream(PumpDevice):
    """Wellstream pump"""

    carrier_in = ["wellstream", "el"]
    carrier_out = ["wellstream"]
    serial = ["wellstream"]


class PumpWater(PumpDevice):
    """Water pump"""

    carrier_in = ["water", "el"]
    carrier_out = ["water"]
    serial = ["water"]

    def __init__(
        self,
        dev_data: dto.DevicePumpWaterData,
        carrier_data_dict: Dict[str, Union[dto.CarrierWaterData, dto.CarrierElData]],
    ):
        super().__init__(dev_data=dev_data, carrier_data_dict=carrier_data_dict)
        self.dev_data = dev_data
        self.id = dev_data.id
        self.carrier_data = carrier_data_dict
