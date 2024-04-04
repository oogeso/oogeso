import logging
from typing import Dict, Union

import pyomo.environ as pyo

from oogeso import dto
from oogeso.core.devices.base import Device

logger = logging.getLogger(__name__)


class CarbonCapture(Device):
    """Carbon capture device"""

    carrier_in = ["carbon", "el", "heat"]
    carrier_out = ["carbon"]
    serial = ["carbon"]

    def __init__(
        self,
        dev_data: dto.DeviceCarbonCaptureData,
        carrier_data_dict: Dict[str, Union[dto.CarrierGasData, dto.CarrierCarbonData]],
    ):
        super().__init__(dev_data=dev_data, carrier_data_dict=carrier_data_dict)
        self.dev_data = dev_data
        self.id = dev_data.id
        self.carrier_data = carrier_data_dict

    @classmethod
    def co2frac(cls, egr):
        """The %CO2 is a function of the GT load and of the EGR rate selected.
        In a first approximation we can make it only a function of the EGR with
        a piecewise linear relationship:"""
        # TODO: compare this with gas_co2_content - two parameters for the same thing?
        # 3.3% if no recirculation.
        # compare with default co2_content = 2.34 # kg_co2/Sm3_gas (SSB 2016 report)
        if egr < 0.3:
            y = (4.96 * egr + 3.3358) / 100
        else:
            y = (12.88 * egr + 0.7874) / 100
        return y

    @classmethod
    def specific_heat_demand(cls, frac_co2):
        y = 0
        if frac_co2 < 0.0489:
            y = -3.57 * frac_co2 + 4.1551
        else:
            y = -2.11 * frac_co2 + 4.0837
        return y

    @classmethod
    def specific_electricity_demand(cls, frac_co2):
        y = 0
        if frac_co2 < 0.0489:
            y = 0.16 * frac_co2 + 0.0232
        else:
            y = 0.009 * frac_co2 + 0.0305
        return y

    @classmethod
    def required_heat(cls, flow_in_co2, egr, ccr):
        f_co2 = cls.co2frac(egr)
        y = flow_in_co2 * (1 - egr) * ccr * cls.specific_heat_demand(f_co2)
        return y

    @classmethod
    def required_electricity(cls, flow_in_co2, egr, ccr):
        f_co2 = cls.co2frac(egr)
        y = flow_in_co2 * (1 - egr) * ccr * cls.specific_electricity_demand(f_co2)
        return y

    def required_electricity_compressor(self, flow_in_co2):
        carbon_data = self.carrier_data["carbon"]
        dev_data = self.dev_data
        k = carbon_data.k_heat_capacity_ratio
        Z = carbon_data.Z_compressibility
        R = carbon_data.R_individual_gas_constant * 1e-6  # factor 1e-6 converts R units from J/kgK to MJ/kgK
        eta = dev_data.compressor_eta  # isentropic efficiency
        T1 = dev_data.compressor_temp_in  # inlet temperature, Kelvin
        p1 = dev_data.compressor_pressure_in
        p2 = dev_data.compressor_pressure_out
        Q_kg_per_s = flow_in_co2  # kg/s
        a = (k - 1) / k
        c = 1 / eta * k / (k - 1) * Z * R * T1
        P = c * ((p2 / p1) ** a - 1) * Q_kg_per_s
        return P

    def _rules_carbon_capture(self, model: pyo.Model, t: int, i: int) -> Union[pyo.Expression, pyo.Constraint.Skip]:
        dev = self.id
        # gas_data = self.carrier_data["gas"]
        # carbon_data = self.carrier_data["carbon"]
        # gas_co2_content = gas_data.co2_content  # kg/Sm3 = 2.34 kg/Sm3 natural gas = ca 3.9 kg_CO2/kg_gas (>1 because of O2?)

        ccr = self.dev_data.carbon_capture_rate  # 0-1
        egr = self.dev_data.exhaust_gas_recirculation  # 0-0.6

        co2_flow_in = model.varDeviceFlow[dev, "carbon", "in", t]  # kg/s

        if i == 1:
            # heat consumption (heat in) is a linear function of carbon flow
            lhs = model.varDeviceFlow[dev, "heat", "in", t]
            rhs = self.required_heat(co2_flow_in, egr=egr, ccr=ccr)
            return lhs == rhs
        elif i == 2:
            # electricity consumption for carbon capture process and co2 compression
            lhs = model.varDeviceFlow[dev, "el", "in", t]
            electricity_carbon_capture = self.required_electricity(co2_flow_in, egr=egr, ccr=ccr)
            # compression:
            carbon_captured = co2_flow_in * (1 - egr) * ccr
            electricity_compressor = self.required_electricity_compressor(carbon_captured)
            rhs = electricity_carbon_capture + electricity_compressor
            return lhs == rhs
        elif i == 3:
            # carbon emitted
            co2_emitted = co2_flow_in * (1 - egr) * (1 - ccr)
            lhs = model.varDeviceFlow[dev, "carbon", "out", t]
            rhs = co2_emitted
            return lhs == rhs
        else:
            raise ValueError()

    def define_constraints(self, pyomo_model: pyo.Model):
        """Specifies the list of constraints for the device"""

        list_to_reconstruct = super().define_constraints(pyomo_model)

        constr = pyo.Constraint(pyomo_model.setHorizon, pyo.RangeSet(1, 3), rule=self._rules_carbon_capture)
        # add constraint to model:
        setattr(pyomo_model, "constr_{}_{}".format(self.id, "misc"), constr)
        return list_to_reconstruct

    def get_flow_var(self, pyomo_model: pyo.Model, t: int):
        return pyomo_model.varDeviceFlow[self.id, "carbon", "in", t]
