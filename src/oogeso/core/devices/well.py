from typing import Dict, List, Union

import pyomo.environ as pyo

from oogeso import dto
from oogeso.core.devices.base import Device


class WellProduction(Device):
    """Production well (wellstream source)"""

    carrier_in = []
    carrier_out = ["wellstream"]
    serial = []

    def __init__(
        self,
        dev_data: dto.DeviceWellProductionData,
        carrier_data_dict: Dict[str, dto.CarrierWellStreamData],
    ):
        super().__init__(dev_data=dev_data, carrier_data_dict=carrier_data_dict)
        self.dev_data = dev_data
        self.id = dev_data.id
        self.carrier_data = carrier_data_dict

    def _rule_well_production(self, pyomo_model: pyo.Model, t: int) -> Union[pyo.Expression, pyo.Constraint.Skip]:
        dev_data = self.dev_data
        node = dev_data.node_id
        lhs = pyomo_model.varPressure[(node, "wellstream", "out", t)]
        rhs = dev_data.wellhead_pressure
        return lhs == rhs

    def define_constraints(self, pyomo_model: pyo.Model) -> List[pyo.Constraint]:
        """Specifies the list of constraints for the device"""
        # No specific constraints, use only generic ones:
        list_to_reconstruct = super().define_constraints(pyomo_model)

        constr_well = pyo.Constraint(pyomo_model.setHorizon, rule=self._rule_well_production)
        # add constraint to model:
        setattr(
            pyomo_model,
            "constr_{}_{}".format(self.id, "pressure"),
            constr_well,
        )
        return list_to_reconstruct

    def get_flow_var(self, pyomo_model: pyo.Model, t: int) -> float:
        return pyomo_model.varDeviceFlow[self.id, "wellstream", "out", t]


class WellGasLift(Device):
    """Production well with gas lift"""

    carrier_in = ["gas"]
    carrier_out = ["gas", "oil", "water"]
    serial = ["gas"]

    def __init__(
        self,
        dev_data: dto.DeviceWellGasLiftData,
        carrier_data_dict: Dict[str, dto.CarrierGasData],
    ):
        super().__init__(dev_data=dev_data, carrier_data_dict=carrier_data_dict)
        self.dev_data = dev_data
        self.id = dev_data.id
        self.carrier_data = carrier_data_dict

    def _rule_gaslift(
        self, pyomo_model: pyo.Model, carrier: dto.CarrierGasData, t: int, i: int
    ) -> Union[pyo.Expression, pyo.Constraint.Skip]:

        # flow from reservoir (equals flow out minus gas injection)
        dev = self.id
        dev_data = self.dev_data
        Q_reservoir = (
            sum(pyomo_model.varDeviceFlow[dev, c, "out", t] for c in ["gas", "oil", "water"])
            - pyomo_model.varDeviceFlow[dev, "gas", "in", t]
        )
        node = dev_data.node_id
        if i == 1:
            # output pressure is fixed
            lhs = pyomo_model.varPressure[(node, carrier, "out", t)]
            rhs = dev_data.separator_pressure
            return lhs == rhs
        elif i == 2:
            # output flow per comonent determined by GOR and WC
            GOR = dev_data.gas_oil_ratio
            WC = dev_data.water_cut
            comp_oil = (1 - WC) / (1 + GOR - GOR * WC)
            comp_water = WC / (1 + GOR - GOR * WC)
            comp_gas = GOR * (1 - WC) / (1 + GOR * (1 - WC))
            comp = {"oil": comp_oil, "gas": comp_gas, "water": comp_water}
            lhs = pyomo_model.varDeviceFlow[dev, carrier, "out", t]
            if carrier == "gas":
                lhs -= pyomo_model.varDeviceFlow[dev, carrier, "in", t]
            rhs = comp[carrier] * Q_reservoir
            return lhs == rhs
        elif i == 3:
            # gas injection rate vs proudction rate (determines input gas)
            # gas injection rate vs OIL proudction rate (determines input gas)
            if carrier == "gas":
                lhs = pyomo_model.varDeviceFlow[dev, "gas", "in", t]
                # rhs = model.paramDevice[dev]['f_inj']*Q_reservoir
                rhs = dev_data.f_inj * pyomo_model.varDeviceFlow[dev, "oil", "out", t]
                return lhs == rhs
            else:
                return pyo.Constraint.Skip  # noqa
        elif i == 4:
            # gas injection pressure is fixed
            if carrier == "gas":
                lhs = pyomo_model.varPressure[(node, carrier, "in", t)]
                rhs = dev_data.injection_pressure
                return lhs == rhs
            else:
                return pyo.Constraint.Skip  # noqa
        else:
            raise ValueError(f"Argument i must be 1, 2, 3 or 4. {i} was given.")

    def define_constraints(self, pyomo_model: pyo.Model) -> List[pyo.Constraint]:
        """Specifies the list of constraints for the device"""
        # No specific constraints, use only generic ones:
        list_to_reconstruct = super().define_constraints(pyomo_model)

        constr_gaslift = pyo.Constraint(
            self.carrier_out,
            pyomo_model.setHorizon,
            pyo.RangeSet(1, 4),
            rule=self._rule_gaslift,
        )
        # add constraint to model:
        setattr(
            pyomo_model,
            "constr_{}_{}".format(self.id, "gaslift"),
            constr_gaslift,
        )
        return list_to_reconstruct

    def get_flow_var(self, pyomo_model: pyo.Model, t: int) -> float:
        dev = self.id
        # flow from reservoir (out minus in)
        flow = (
            pyomo_model.varDeviceFlow[dev, "oil", "out", t]
            + pyomo_model.varDeviceFlow[dev, "gas", "out", t]
            - pyomo_model.varDeviceFlow[dev, "gas", "in", t]
            + pyomo_model.varDeviceFlow[dev, "water", "out", t]
        )
        return flow
