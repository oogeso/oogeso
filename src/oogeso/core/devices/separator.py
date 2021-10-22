import pyomo.environ as pyo
from oogeso.core.networks.network_node import NetworkNode

from oogeso.dto.oogeso_input_data_objects import (
    CarrierWellstreamData,
    DeviceSeparator2Data,
    DeviceSeparatorData,
)
from . import Device


class Separator(Device):
    "Wellstream separation into oil/gas/water"
    carrier_in = ["wellstream", "heat", "el"]
    carrier_out = ["oil", "gas", "water"]
    serial = []

    def _rule_separator(self, model, t, i):
        dev = self.id
        dev_data: DeviceSeparatorData = self.dev_data
        node = dev_data.node_id
        node_obj: NetworkNode = self.optimiser.all_nodes[node]
        wellstream_prop: CarrierWellstreamData = self.all_carriers["wellstream"]
        GOR = wellstream_prop.gas_oil_ratio
        WC = wellstream_prop.water_cut
        comp_oil = (1 - WC) / (1 + GOR - GOR * WC)
        comp_water = WC / (1 + GOR - GOR * WC)
        comp_gas = GOR * (1 - WC) / (1 + GOR * (1 - WC))
        flow_in = model.varDeviceFlow[dev, "wellstream", "in", t]
        if i == 1:
            lhs = model.varDeviceFlow[dev, "gas", "out", t]
            rhs = flow_in * comp_gas
            return lhs == rhs
        elif i == 2:
            lhs = model.varDeviceFlow[dev, "oil", "out", t]
            rhs = flow_in * comp_oil
            return lhs == rhs
        elif i == 3:
            # return pyo.Constraint.Skip
            lhs = model.varDeviceFlow[dev, "water", "out", t]
            rhs = flow_in * comp_water
            return lhs == rhs
        elif i == 4:
            # electricity demand
            lhs = model.varDeviceFlow[dev, "el", "in", t]
            rhs = flow_in * dev_data.el_demand_factor
            return lhs == rhs
        elif i == 5:
            lhs = model.varDeviceFlow[dev, "heat", "in", t]
            rhs = flow_in * dev_data.heat_demand_factor
            return lhs == rhs
        elif i == 6:
            # gas pressure out = nominal
            lhs = model.varPressure[(node, "gas", "out", t)]
            rhs = node_obj.get_nominal_pressure("gas", "out")
            return lhs == rhs
        elif i == 7:
            # oil pressure out = nominal
            lhs = model.varPressure[(node, "oil", "out", t)]
            rhs = node_obj.get_nominal_pressure("oil", "out")
            return lhs == rhs
        elif i == 8:
            # water pressure out = nominal
            lhs = model.varPressure[(node, "water", "out", t)]
            rhs = node_obj.get_nominal_pressure("water", "out")
            return lhs == rhs

    def defineConstraints(self, pyomo_model):
        """Specifies the list of constraints for the device"""

        list_to_reconstruct = super().defineConstraints(pyomo_model)

        constr_separator = pyo.Constraint(
            pyomo_model.setHorizon, pyo.RangeSet(1, 8), rule=self._rule_separator
        )
        # add constraints to model:
        setattr(
            pyomo_model,
            "constr_{}_{}".format(self.id, "separator"),
            constr_separator,
        )
        return list_to_reconstruct


class Separator2(Device):
    "Wellstream separation into oil/gas/water"
    carrier_in = ["oil", "gas", "water", "heat", "el"]
    carrier_out = ["oil", "gas", "water"]
    serial = ["oil", "gas", "water"]

    # Alternative separator model - using oil/gas/water input instead of
    # wellstream
    def _rule_separator2_flow(self, model, fc, t, i):
        dev = self.id
        node = self.dev_data.node_id
        node_obj: NetworkNode = self.node
        # wellstream_prop=self.pyomo_model.all_carriers['wellstream']
        # flow_in = sum(model.varDeviceFlow[dev,f,'in',t]
        #                for f in['oil','gas','water'])
        if i == 1:
            # component flow in = flow out
            lhs = model.varDeviceFlow[dev, fc, "out", t]
            rhs = model.varDeviceFlow[dev, fc, "in", t]
            return lhs == rhs
        elif i == 2:
            # pressure out is nominal
            lhs = model.varPressure[(node, fc, "out", t)]
            rhs = node_obj.get_nominal_pressure(fc, "out")
            return lhs == rhs

    def _rule_separator2_energy(self, model, t, i):
        dev = self.id
        dev_data: DeviceSeparator2Data = self.dev_data
        flow_in = sum(
            model.varDeviceFlow[dev, f, "in", t] for f in ["oil", "gas", "water"]
        )

        if i == 1:
            # electricity demand
            lhs = model.varDeviceFlow[dev, "el", "in", t]
            rhs = flow_in * dev_data.el_demand_factor
            return lhs == rhs
        elif i == 2:
            # heat demand
            lhs = model.varDeviceFlow[dev, "heat", "in", t]
            rhs = flow_in * dev_data.heat_demand_factor
            return lhs == rhs

    def defineConstraints(self, pyomo_model):
        """Specifies the list of constraints for the device"""
        # No specific constraints, use only generic ones:
        list_to_reconstruct = super().defineConstraints(pyomo_model)

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
            pyo.RangeSet(1, 2),
            rule=self._rule_separator2_energy,
        )
        # add constraints to model:
        setattr(
            pyomo_model,
            "constr_{}_{}".format(self.id, "energy"),
            constr_separator2_energy,
        )
        return list_to_reconstruct
