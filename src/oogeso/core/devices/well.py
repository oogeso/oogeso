import pyomo.environ as pyo
import logging

from oogeso.dto.oogeso_input_data_objects import (
    DeviceWell_gasliftData,
    DeviceWell_productionData,
)
from . import Device


class Well_production(Device):
    "Production well (wellstream source)"
    carrier_in = []
    carrier_out = ["wellstream"]
    serial = []

    def _rule_well_production(self, model, t):
        dev_data: DeviceWell_productionData = self.dev_data
        node = dev_data.node_id
        lhs = model.varPressure[(node, "wellstream", "out", t)]
        rhs = dev_data.wellhead_pressure
        return lhs == rhs

    def defineConstraints(self):
        """Specifies the list of constraints for the device"""
        # No specific constraints, use only generic ones:
        super().defineConstraints()

        constr_well = pyo.Constraint(
            self.pyomo_model.setHorizon, rule=self._rule_well_production
        )
        # add constraint to model:
        setattr(
            self.pyomo_model,
            "constr_{}_{}".format(self.id, "pressure"),
            constr_well,
        )

    def getFlowVar(self, t):
        return self.pyomo_model.varDeviceFlow[self.id, "wellstream", "out", t]


class Well_gaslift(Device):
    "Production well with gas lift"
    carrier_in = ["gas"]
    carrier_out = ["gas", "oil", "water"]
    serial = ["gas"]

    def _rule_gaslift(self, model, carrier, t, i):

        # flow from reservoir (equals flow out minus gas injection)
        dev = self.id
        dev_data: DeviceWell_gasliftData = self.dev_data
        Q_reservoir = (
            sum(model.varDeviceFlow[dev, c, "out", t] for c in ["gas", "oil", "water"])
            - model.varDeviceFlow[dev, "gas", "in", t]
        )
        node = dev_data.node_id
        if i == 1:
            # output pressure is fixed
            lhs = model.varPressure[(node, carrier, "out", t)]
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
            lhs = model.varDeviceFlow[dev, carrier, "out", t]
            if carrier == "gas":
                lhs -= model.varDeviceFlow[dev, carrier, "in", t]
            rhs = comp[carrier] * Q_reservoir
            return lhs == rhs
        elif i == 3:
            # gas injection rate vs proudction rate (determines input gas)
            # gas injection rate vs OIL proudction rate (determines input gas)
            if carrier == "gas":
                lhs = model.varDeviceFlow[dev, "gas", "in", t]
                # rhs = model.paramDevice[dev]['f_inj']*Q_reservoir
                rhs = dev_data.f_inj * model.varDeviceFlow[dev, "oil", "out", t]
                return lhs == rhs
            else:
                return pyo.Constraint.Skip
        elif i == 4:
            # gas injection pressure is fixed
            if carrier == "gas":
                lhs = model.varPressure[(node, carrier, "in", t)]
                rhs = dev_data.injection_pressure
                return lhs == rhs
            else:
                return pyo.Constraint.Skip

    def defineConstraints(self):
        """Specifies the list of constraints for the device"""
        # No specific constraints, use only generic ones:
        super().defineConstraints()

        constr_gaslift = pyo.Constraint(
            self.carrier_out,
            self.pyomo_model.setHorizon,
            pyo.RangeSet(1, 4),
            rule=self._rule_gaslift,
        )
        # add constraint to model:
        setattr(
            self.pyomo_model,
            "constr_{}_{}".format(self.id, "gaslift"),
            constr_gaslift,
        )

    def getFlowVar(self, t):
        dev = self.id
        # flow from reservoir (out minus in)
        flow = (
            self.pyomo_model.varDeviceFlow[dev, "oil", "out", t]
            + self.pyomo_model.varDeviceFlow[dev, "gas", "out", t]
            - self.pyomo_model.varDeviceFlow[dev, "gas", "in", t]
            + self.pyomo_model.varDeviceFlow[dev, "water", "out", t]
        )
        return flow
