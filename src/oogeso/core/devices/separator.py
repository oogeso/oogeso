from typing import Dict, List, Union

import pyomo.environ as pyo

from oogeso import dto
from oogeso.core.devices.base import Device
from oogeso.core.networks.network_node import NetworkNode


class Separator(Device):
    """Wellstream separation into oil/gas/water."""

    carrier_in = ["wellstream", "heat", "el"]
    carrier_out = ["oil", "gas", "water"]
    serial = []

    def __init__(
        self,
        dev_data: dto.DeviceSeparatorData,
        carrier_data_dict: Dict[str, Union[dto.CarrierElData, dto.CarrierHeatData, dto.CarrierWellStreamData]],
    ):
        super().__init__(dev_data=dev_data, carrier_data_dict=carrier_data_dict)
        self.dev_data = dev_data
        self.id = dev_data.id
        self.carrier_data = carrier_data_dict

    def _rule_separator(self, pyomo_model: pyo.Model, t: int, i: int) -> Union[pyo.Expression, pyo.Constraint.Skip]:
        dev = self.id
        dev_data = self.dev_data
        node = dev_data.node_id
        node_obj: NetworkNode = self.node
        wellstream_prop: dto.CarrierWellStreamData = self.carrier_data["wellstream"]
        GOR = wellstream_prop.gas_oil_ratio
        WC = wellstream_prop.water_cut
        comp_oil = (1 - WC) / (1 + GOR - GOR * WC)
        comp_water = WC / (1 + GOR - GOR * WC)
        comp_gas = GOR * (1 - WC) / (1 + GOR * (1 - WC))
        flow_in = pyomo_model.varDeviceFlow[dev, "wellstream", "in", t]
        if i == 1:
            lhs = pyomo_model.varDeviceFlow[dev, "gas", "out", t]
            rhs = flow_in * comp_gas
            return lhs == rhs
        elif i == 2:
            lhs = pyomo_model.varDeviceFlow[dev, "oil", "out", t]
            rhs = flow_in * comp_oil
            return lhs == rhs
        elif i == 3:
            # return pyo.Constraint.Skip
            lhs = pyomo_model.varDeviceFlow[dev, "water", "out", t]
            rhs = flow_in * comp_water
            return lhs == rhs
        elif i == 4:
            lhs = pyomo_model.varDeviceFlow[dev, "heat", "in", t]
            rhs = flow_in * dev_data.heat_demand_factor
            return lhs == rhs
        elif i == 5:
            # gas pressure out = nominal
            lhs = pyomo_model.varPressure[(node, "gas", "out", t)]
            rhs = node_obj.get_pressure_nominal("gas", "out")
            return lhs == rhs  # noqa
        elif i == 6:
            # oil pressure out = nominal
            lhs = pyomo_model.varPressure[(node, "oil", "out", t)]
            rhs = node_obj.get_pressure_nominal("oil", "out")
            return lhs == rhs  # noqa
        elif i == 7:
            # water pressure out = nominal
            lhs = pyomo_model.varPressure[(node, "water", "out", t)]
            rhs = node_obj.get_pressure_nominal("water", "out")
            return lhs == rhs  # noqa
        else:
            raise ValueError(f"Argument i must be 1, 2, ..., 8. {i} was given.")

    def define_constraints(self, pyomo_model: pyo.Model):
        """Specifies the list of constraints for the device"""

        list_to_reconstruct = super().define_constraints(pyomo_model)

        constr_separator = pyo.Constraint(pyomo_model.setHorizon, pyo.RangeSet(1, 7), rule=self._rule_separator)
        # add constraints to model:
        setattr(
            pyomo_model,
            "constr_{}_{}".format(self.id, "separator"),
            constr_separator,
        )
        return list_to_reconstruct

    def get_flow_var(self, pyomo_model: pyo.Model, t: int) -> float:
        return pyomo_model.varDeviceFlow[self.id, "wellstream", "in", t]


class Separator2(Device):
    """
    Wellstream separation into oil/gas/water

    Alternative separator model - using oil/gas/water input instead of
    """

    carrier_in = ["oil", "gas", "water", "heat"]
    carrier_out = ["oil", "gas", "water"]
    serial = ["oil", "gas", "water"]

    def __init__(
        self,
        dev_data: dto.DeviceSeparator2Data,
        carrier_data_dict: Dict[
            str,
            Union[dto.CarrierGasData, dto.CarrierOilData, dto.CarrierElData, dto.CarrierHeatData, dto.CarrierWaterData],
        ],
    ):
        super().__init__(dev_data=dev_data, carrier_data_dict=carrier_data_dict)
        self.dev_data = dev_data
        self.id = dev_data.id
        self.carrier_data = carrier_data_dict

    def _rule_separator2_flow(
        self, pyomo_model: pyo.Model, fc, t: int, i: int
    ) -> Union[pyo.Expression, pyo.Constraint.Skip]:
        dev = self.id
        node = self.dev_data.node_id
        node_obj: NetworkNode = self.node
        if i == 1:
            # component flow in = flow out
            lhs = pyomo_model.varDeviceFlow[dev, fc, "out", t]
            rhs = pyomo_model.varDeviceFlow[dev, fc, "in", t]
            return lhs == rhs  # noqa
        elif i == 2:
            # pressure out is nominal
            lhs = pyomo_model.varPressure[(node, fc, "out", t)]
            rhs = node_obj.get_pressure_nominal(fc, "out")
            return lhs == rhs  # noqa
        else:
            raise ValueError(f"Argument i must be 1 or 2. {i} was given.")

    def _rule_separator2_heat(self, pyomo_model: pyo.Model, t: int) -> Union[pyo.Expression, pyo.Constraint.Skip]:
        dev = self.id
        dev_data: dto.DeviceSeparator2Data = self.dev_data
        flow_in = sum(pyomo_model.varDeviceFlow[dev, f, "in", t] for f in ["oil", "gas", "water"])

        # heat demand
        lhs = pyomo_model.varDeviceFlow[dev, "heat", "in", t]
        rhs = flow_in * dev_data.heat_demand_factor
        return lhs == rhs

    def define_constraints(self, pyomo_model: pyo.Model) -> List[pyo.Constraint]:
        """Specifies the list of constraints for the device"""
        # No specific constraints, use only generic ones:
        list_to_reconstruct = super().define_constraints(pyomo_model)

        constr_separator2_flow = pyo.Constraint(
            ["oil", "gas", "water"],
            pyomo_model.setHorizon,
            pyo.RangeSet(1, 2),
            rule=self._rule_separator2_flow,
        )
        # add constraints to model:
        setattr(
            pyomo_model,
            "constr_{}_{}".format(self.id, "flow"),
            constr_separator2_flow,
        )

        constr_separator2_energy = pyo.Constraint(
            pyomo_model.setHorizon,
            rule=self._rule_separator2_heat,
        )
        # add constraints to model:
        setattr(
            pyomo_model,
            "constr_{}_{}".format(self.id, "energy"),
            constr_separator2_energy,
        )
        return list_to_reconstruct

    def get_flow_var(self, pyomo_model: pyo.Model, t: int) -> float:
        return (
            pyomo_model.varDeviceFlow[self.id, "gas", "in", t]
            + pyomo_model.varDeviceFlow[self.id, "oil", "in", t]
            + pyomo_model.varDeviceFlow[self.id, "water", "in", t]
        )
