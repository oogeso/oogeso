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
    def required_heat(cls, flow_in_co2, ccr, specific_demand_MW_per_kgCO2):
        # f_co2 = cls.co2frac(egr)
        # y = flow_in_co2 * (1 - egr) * ccr * cls.specific_heat_demand(f_co2)
        y = flow_in_co2 * ccr * specific_demand_MW_per_kgCO2
        return y

    @classmethod
    def required_electricity(self, flow_in_co2, ccr, specific_demand_MW_per_kgCO2):
        # f_co2 = cls.co2frac(egr)
        # y = flow_in_co2 * (1 - egr) * ccr * cls.specific_electricity_demand(f_co2)
        y = flow_in_co2 * ccr * specific_demand_MW_per_kgCO2
        return y

    @classmethod
    def required_electricity_compressor(cls, flow_in_co2, energy_demand_MJ_per_kg):
        Q_kg_per_s = flow_in_co2  # kg/s
        P_MW = Q_kg_per_s * energy_demand_MJ_per_kg
        return P_MW

    def _rules_carbon_capture(self, model: pyo.Model, t: int, i: int) -> Union[pyo.Expression, pyo.Constraint.Skip]:
        dev = self.id
        ccr = self.dev_data.carbon_capture_rate  # 0-1
        ccs_el_demand = self.dev_data.capture_el_demand_MJ_per_kgCO2
        ccs_heat_demand = self.dev_data.capture_heat_demand_MJ_per_kgCO2
        compressor_el_demand = self.dev_data.compressor_el_demand_MJ_per_kgCO2

        co2_flow_in = model.varDeviceFlow[dev, "carbon", "in", t]  # kg/s

        if i == 1:
            # heat consumption (heat in) is a linear function of carbon flow
            lhs = model.varDeviceFlow[dev, "heat", "in", t]
            rhs = self.required_heat(co2_flow_in, ccr=ccr, specific_demand_MW_per_kgCO2=ccs_heat_demand)
            return lhs == rhs
        elif i == 2:
            # electricity consumption for carbon capture process and co2 compression
            lhs = model.varDeviceFlow[dev, "el", "in", t]
            electricity_carbon_capture = self.required_electricity(
                co2_flow_in, ccr=ccr, specific_demand_MW_per_kgCO2=ccs_el_demand
            )
            # compression:
            carbon_captured = co2_flow_in * ccr
            electricity_compressor = self.required_electricity_compressor(carbon_captured, compressor_el_demand)
            rhs = electricity_carbon_capture + electricity_compressor
            return lhs == rhs
        elif i == 3:
            # carbon emitted
            co2_emitted = co2_flow_in * (1 - ccr)
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
