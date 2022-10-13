from typing import Dict, List, Optional, Union

import pyomo.environ as pyo

from oogeso import dto
from oogeso.core.devices.base import Device
from oogeso.core.networks.network_node import NetworkNode


class CompressorEl(Device):
    """Electric compressor."""

    carrier_in = ["gas", "el"]
    carrier_out = ["gas"]
    serial = ["gas"]

    def __init__(
        self,
        dev_data: Union[dto.DeviceCompressorGasData, dto.DeviceCompressorElData],
        carrier_data_dict: Dict[str, Union[dto.CarrierElData, dto.CarrierGasData]],
    ):
        super().__init__(dev_data=dev_data, carrier_data_dict=carrier_data_dict)
        self.dev_data = dev_data
        self.id = dev_data.id
        self.carrier_data = carrier_data_dict

    def _rules(self, pyomo_model: pyo.Model, t: int, i: int) -> Union[pyo.Expression, pyo.Constraint.Skip]:
        dev = self.id
        node_obj: NetworkNode = self.node
        gas_data = self.carrier_data["gas"]
        if i == 1:
            """gas flow in equals gas flow out (mass flow)"""
            lhs = pyomo_model.varDeviceFlow[dev, "gas", "in", t]
            rhs = pyomo_model.varDeviceFlow[dev, "gas", "out", t]
            return lhs == rhs  # noqa
        elif i == 2:
            """Device el demand"""
            lhs = pyomo_model.varDeviceFlow[dev, "el", "in", t]
            rhs = compute_compressor_demand(pyomo_model, self, node_obj, gas_data, linear=True, t=t)
            return lhs == rhs

    def define_constraints(self, pyomo_model: pyo.Model) -> List[pyo.Constraint]:
        """Specifies the list of constraints for the device"""

        list_to_reconstruct = super().define_constraints(pyomo_model)

        constr_compressor_el = pyo.Constraint(pyomo_model.setHorizon, pyo.RangeSet(1, 2), rule=self._rules)
        # add constraint to model:
        setattr(
            pyomo_model,
            "constr_{}_{}".format(self.id, "compr"),
            constr_compressor_el,
        )
        return list_to_reconstruct

    def get_flow_var(self, pyomo_model: pyo.Model, t: int) -> pyo.Var:
        return pyomo_model.varDeviceFlow[self.id, "el", "in", t]


class CompressorGas(Device):
    """
    Gas-driven compressor
    """

    carrier_in = ["gas"]
    carrier_out = ["gas"]
    serial = ["gas"]

    def __init__(
        self,
        dev_data: dto.DeviceCompressorGasData,
        carrier_data_dict: Dict[str, dto.CarrierGasData],
    ):
        super().__init__(dev_data=dev_data, carrier_data_dict=carrier_data_dict)
        self.dev_data = dev_data
        self.id = dev_data.id
        self.carrier_data = carrier_data_dict

    def _rules(self, pyomo_model: pyo.Model, t: int) -> Union[pyo.Expression, pyo.Constraint.Skip]:
        dev = self.id
        node_obj: NetworkNode = self.node
        gas_data = self.carrier_data["gas"]
        gas_energy_content = gas_data.energy_value  # MJ/Sm3
        power_demand = compute_compressor_demand(pyomo_model, self, node_obj, gas_data, linear=True, t=t)

        # matter conservation:
        lhs = pyomo_model.varDeviceFlow[dev, "gas", "out", t]
        rhs = pyomo_model.varDeviceFlow[dev, "gas", "in", t] - power_demand / gas_energy_content

        return lhs == rhs

    def define_constraints(self, pyomo_model: pyo.Model):
        """Specifies the list of constraints for the device"""

        list_to_reconstruct = super().define_constraints(pyomo_model)

        constr = pyo.Constraint(pyomo_model.setHorizon, rule=self._rules)
        setattr(pyomo_model, "constr_{}_{}".format(self.id, "compr"), constr)
        return list_to_reconstruct

    # overriding default
    def compute_CO2(self, pyomo_model: pyo.Model, timesteps: List[int]):
        d = self.id
        gas_data = self.carrier_data["gas"]
        gasflow_co2 = gas_data.co2_content  # kg/m3

        return (
            sum(
                (pyomo_model.varDeviceFlow[d, "gas", "in", t] - pyomo_model.varDeviceFlow[d, "gas", "out", t])
                for t in timesteps
            )
            * gasflow_co2
        )

    def get_flow_var(self, pyomo_model: pyo.Model, t: int) -> float:
        return pyomo_model.varDeviceFlow[self.id, "gas", "in", t]


def compute_compressor_demand(
    model,
    device_obj: Union[CompressorEl, CompressorGas],
    node_obj: NetworkNode,
    gas_data: dto.CarrierGasData,
    linear: bool = False,
    Q: Optional[float] = None,
    p1: Optional[float] = None,
    p2: Optional[float] = None,
    t: Optional[float] = None,
):
    """Compute energy demand by compressor as function of pressure and flow"""
    # power demand depends on gas pressure ratio and flow
    # See LowEmission report DSP5_2020_04 for description

    dev_data = device_obj.dev_data
    k = gas_data.k_heat_capacity_ratio
    Z = gas_data.Z_compressibility
    # factor 1e-6 converts R units from J/kgK to MJ/kgK:
    R = gas_data.R_individual_gas_constant * 1e-6
    rho = gas_data.rho_density
    T1 = dev_data.temp_in  # inlet temperature
    eta = dev_data.eta  # isentropic efficiency
    a = (k - 1) / k
    c = rho / eta * k / (k - 1) * Z * R * T1
    node = node_obj.id
    if t is None:
        t = 0
    if Q is None:
        Q = model.varDeviceFlow[device_obj.id, "gas", "out", t]
    if p1 is None:
        p1 = model.varPressure[node, "gas", "in", t]
    if p2 is None:
        p2 = model.varPressure[node, "gas", "out", t]
    if linear:
        # linearised equations around operating point
        # p1=p10, p2=p20, Q=Q0
        p10 = node_obj.pressure_nominal["gas"]["in"]
        p20 = node_obj.pressure_nominal["gas"]["out"]
        Q0 = dev_data.Q0
        P = c * (a * (p20 / p10) ** a * Q0 * (p2 / p20 - p1 / p10) + ((p20 / p10) ** a - 1) * Q)
    else:
        P = c * ((p2 / p1) ** a - 1) * Q
    return P
