import pyomo.environ as pyo

from oogeso.core.networks.network_node import NetworkNode
from . import Device
from oogeso.dto import (
    CarrierData,
    DeviceCompressor_elData,
    DeviceCompressor_gasData,
)
from typing import Union


class Compressor_el(Device):
    "Electric compressor"
    carrier_in = ["gas", "el"]
    carrier_out = ["gas"]
    serial = ["gas"]

    def _rules(self, model, t, i):
        dev = self.id
        node_obj: NetworkNode = self.node
        gas_data = self.carrier_data["gas"]
        if i == 1:
            """gas flow in equals gas flow out (mass flow)"""
            lhs = model.varDeviceFlow[dev, "gas", "in", t]
            rhs = model.varDeviceFlow[dev, "gas", "out", t]
            return lhs == rhs
        elif i == 2:
            """Device el demand"""
            lhs = model.varDeviceFlow[dev, "el", "in", t]
            rhs = compute_compressor_demand(
                model, self, node_obj, gas_data, linear=True, t=t
            )
            return lhs == rhs

    def defineConstraints(self, pyomo_model):
        """Specifies the list of constraints for the device"""

        list_to_reconstruct = super().defineConstraints(pyomo_model)

        constr_compressor_el = pyo.Constraint(
            pyomo_model.setHorizon, pyo.RangeSet(1, 2), rule=self._rules
        )
        # add constraint to model:
        setattr(
            pyomo_model,
            "constr_{}_{}".format(self.id, "compr"),
            constr_compressor_el,
        )
        return list_to_reconstruct

    def getFlowVar(self, pyomo_model, t):
        return pyomo_model.varDeviceFlow[self.id, "el", "in", t]


class Compressor_gas(Device):
    "Gas-driven compressor"
    carrier_in = ["gas"]
    carrier_out = ["gas"]
    serial = ["gas"]

    def _rules(self, model, t):
        dev = self.id
        dev_data: DeviceCompressor_gasData = self.dev_data
        node_obj: NetworkNode = self.node
        gas_data = self.carrier_data["gas"]
        gas_energy_content = gas_data.energy_value  # MJ/Sm3
        powerdemand = compute_compressor_demand(
            model, self, node_obj, gas_data, linear=True, t=t
        )
        # matter conservation:
        lhs = model.varDeviceFlow[dev, "gas", "out", t]
        rhs = (
            model.varDeviceFlow[dev, "gas", "in", t] - powerdemand / gas_energy_content
        )
        return lhs == rhs

    def defineConstraints(self, pyomo_model):
        """Specifies the list of constraints for the device"""

        list_to_reconstruct = super().defineConstraints(pyomo_model)

        constr = pyo.Constraint(pyomo_model.setHorizon, rule=self._rules)
        setattr(pyomo_model, "constr_{}_{}".format(self.id, "compr"), constr)
        return list_to_reconstruct

    # overriding default
    def compute_CO2(self, pyomo_model, timesteps):
        d = self.id
        gas_data = self.carrier_data["gas"]
        gasflow_co2 = gas_data.CO2content  # kg/m3
        thisCO2 = (
            sum(
                (
                    pyomo_model.varDeviceFlow[d, "gas", "in", t]
                    - pyomo_model.varDeviceFlow[d, "gas", "out", t]
                )
                for t in timesteps
            )
            * gasflow_co2
        )
        return thisCO2


def compute_compressor_demand(
    model,
    device_obj: Union[Compressor_el, Compressor_gas],
    node_obj: NetworkNode,
    gas_data: CarrierData,
    linear=False,
    Q=None,
    p1=None,
    p2=None,
    t=None,
):
    """Compute energy demand by compressor as function of pressure and flow"""
    # power demand depends on gas pressure ratio and flow
    # See LowEmission report DSP5_2020_04 for description

    dev_data: Union[
        DeviceCompressor_elData, DeviceCompressor_gasData
    ] = device_obj.dev_data
    k = gas_data.k_heat_capacity_ratio
    Z = gas_data.Z_compressibility
    # factor 1e-6 converts R units from J/kgK to MJ/kgK:
    R = gas_data.R_individual_gas_constant * 1e-6
    rho = gas_data.rho_density
    T1 = dev_data.temp_in  # inlet temperature
    eta = dev_data.eta  # isentropic efficiency
    a = (k - 1) / k
    c = rho / eta * 1 / (k - 1) * Z * R * T1
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
        p10 = node_obj.nominal_pressure["gas"]["in"]
        p20 = node_obj.nominal_pressure["gas"]["out"]
        Q0 = dev_data.Q0
        P = c * (
            a * (p20 / p10) ** a * Q0 * (p2 / p20 - p1 / p10)
            + ((p20 / p10) ** a - 1) * Q
        )
    else:
        P = c * ((p2 / p1) ** a - 1) * Q
    return P
