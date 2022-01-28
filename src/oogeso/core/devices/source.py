from typing import Dict, List, Union

import pyomo.environ as pyo

from oogeso import dto
from oogeso.core.devices.base import Device


class SourceEl(Device):
    """Generic external source for electricity (e.g. cable or wind turbine)"""

    dev_data: dto.DeviceSourceElData

    carrier_in = []
    carrier_out = ["el"]
    serial = []

    def __init__(
        self,
        dev_data: dto.DeviceSourceElData,
        carrier_data_dict: Dict[str, dto.CarrierElData],
    ):
        super().__init__(dev_data=dev_data, carrier_data_dict=carrier_data_dict)
        self.dev_data = dev_data
        self.id = dev_data.id
        self.carrier_data = carrier_data_dict

    def get_flow_var(self, pyomo_model: pyo.Model, t: int):
        return pyomo_model.varDeviceFlow[self.id, "el", "out", t]

    # overriding default
    def compute_CO2(self, pyomo_model: pyo.Model, timesteps: List[int]):
        # co2 content in fuel combustion
        # co2em is kgCO2/MWh_el, deltaT is seconds, deviceFlow is MW
        # need to convert co2em to kgCO2/(MW*s)
        if self.dev_data.co2em is not None:
            return (
                sum(pyomo_model.varDeviceFlow[self.id, "el", "out", t] * self.dev_data.co2em for t in timesteps)
                * 1
                / 3600
            )
        return 0.0


class PowerSource(Device):
    """Generic external source for electricity (e.g. cable or wind turbine)."""

    dev_data: dto.DevicePowerSourceData

    carrier_in = []
    carrier_out = ["el"]
    serial = []

    def __init__(
        self,
        dev_data: dto.DevicePowerSourceData,
        carrier_data_dict: Dict[str, dto.CarrierElData],
    ):
        super().__init__(dev_data=dev_data, carrier_data_dict=carrier_data_dict)
        self.dev_data = dev_data
        self.id = dev_data.id
        self.carrier_data = carrier_data_dict

    def define_constraints(self, pyomo_model: pyo.Model):
        """Specifies the list of constraints for the device"""

        list_to_reconstruct = super().define_constraints(pyomo_model)

        if self.dev_data.penalty_function is not None:
            constr_penalty = self._penalty_constraint(pyomo_model)
            setattr(
                pyomo_model,
                "constrPW_{}_{}".format(self.id, "penalty"),
                constr_penalty,
            )
        return list_to_reconstruct

    def _penalty_constraint(self, pyomo_model: pyo.Model):
        # Piecewise constraints require independent variable to be bounded:
        # ub = self.dev_data.flow_max
        ub = self.get_flow_upper_bound()
        pyomo_model.varDeviceFlow[self.id, "el", "out", :].setlb(0)
        pyomo_model.varDeviceFlow[self.id, "el", "out", :].setub(ub)
        lookup_table = self.dev_data.penalty_function
        pw_x = lookup_table[0]
        pw_y = lookup_table[1]
        var_x = pyomo_model.varDeviceFlow  # [self.dev_id, "el", "out", :]
        var_y = pyomo_model.varDevicePenalty  # [self.dev_id, "el", "out", :]
        pw_repn = pyomo_model.optimisation_parameters.piecewise_repn
        constr_penalty = pyo.Piecewise(
            [self.id],
            ["el"],
            ["out"],
            pyomo_model.setHorizon,
            var_y,
            var_x,
            pw_repn=pw_repn,  # default "SOS2" does not work with CBC solver
            pw_constr_type="EQ",
            pw_pts=pw_x,
            f_rule=pw_y,  # self._penaltyfunction,
        )
        return constr_penalty

    def get_flow_var(self, pyomo_model: pyo.Model, t: int):
        return pyomo_model.varDeviceFlow[self.id, "el", "out", t]


class SourceGas(Device):
    """Generic external source for gas."""

    dev_data: dto.DeviceSourceGasData

    carrier_in = []
    carrier_out = ["gas"]
    serial = []

    def __init__(
        self,
        dev_data: dto.DeviceSourceGasData,
        carrier_data_dict: Dict[str, dto.CarrierGasData],
    ):
        super().__init__(dev_data=dev_data, carrier_data_dict=carrier_data_dict)
        self.dev_data = dev_data
        self.id = dev_data.id
        self.carrier_data = carrier_data_dict

    def _rules(self, model: pyo.Model, t: int) -> Union[pyo.Expression, pyo.Constraint.Skip]:
        node = self.dev_data.node_id
        lhs = model.varPressure[(node, "gas", "out", t)]
        rhs = self.dev_data.naturalpressure
        return lhs == rhs

    def define_constraints(self, pyomo_model: pyo.Model):
        """Specifies the list of constraints for the device"""
        # add generic device constraints:
        list_to_reconstruct = super().define_constraints(pyomo_model)

        constr_well = pyo.Constraint(pyomo_model.setHorizon, rule=self._rules)
        # add constraint to model:
        setattr(
            pyomo_model,
            "constr_{}_{}".format(self.id, "pressure"),
            constr_well,
        )
        return list_to_reconstruct

    def get_flow_var(self, pyomo_model: pyo.Model, t: int):
        return pyomo_model.varDeviceFlow[self.id, "gas", "out", t]


class SourceOil(Device):
    """Generic external source for oil."""

    dev_data: dto.DeviceSourceOilData

    carrier_in = []
    carrier_out = ["oil"]
    serial = []

    def _rules(self, model: pyo.Model, t: int) -> Union[pyo.Expression, pyo.Constraint.Skip]:
        node = self.dev_data.node_id
        lhs = model.varPressure[(node, "oil", "out", t)]
        rhs = self.dev_data.naturalpressure
        return lhs == rhs

    def define_constraints(self, pyomo_model: pyo.Model):
        """Specifies the list of constraints for the device"""
        # add generic device constraints:
        list_to_reconstruct = super().define_constraints(pyomo_model)

        constr_well = pyo.Constraint(pyomo_model.setHorizon, rule=self._rules)
        # add constraint to model:
        setattr(
            pyomo_model,
            "constr_{}_{}".format(self.id, "pressure"),
            constr_well,
        )
        return list_to_reconstruct

    def get_flow_var(self, pyomo_model: pyo.Model, t: int):
        return pyomo_model.varDeviceFlow[self.id, "oil", "out", t]


class SourceWater(Device):
    """Generic external source for water."""

    dev_data: dto.DeviceSourceWaterData

    carrier_in = []
    carrier_out = ["water"]
    serial = []

    def __init__(
        self,
        dev_data: dto.DeviceSourceWaterData,
        carrier_data_dict: Dict[str, dto.CarrierWaterData],
    ):
        super().__init__(dev_data=dev_data, carrier_data_dict=carrier_data_dict)
        self.dev_data = dev_data
        self.id = dev_data.id
        self.carrier_data = carrier_data_dict

    def _rules(self, model: pyo.Model, t: int) -> Union[pyo.Expression, pyo.Constraint.Skip]:
        node = self.dev_data.node_id
        lhs = model.varPressure[(node, "water", "out", t)]
        rhs = self.dev_data.naturalpressure
        return lhs == rhs

    def define_constraints(self, pyomo_model: pyo.Model):
        """Specifies the list of constraints for the device"""
        # add generic device constraints:
        list_to_reconstruct = super().define_constraints(pyomo_model)

        constr_well = pyo.Constraint(pyomo_model.setHorizon, rule=self._rules)
        # add constraint to model:
        setattr(
            pyomo_model,
            "constr_{}_{}".format(self.id, "pressure"),
            constr_well,
        )
        return list_to_reconstruct

    def get_flow_var(self, pyomo_model: pyo.Model, t: int):
        return pyomo_model.varDeviceFlow[self.id, "water", "out", t]
