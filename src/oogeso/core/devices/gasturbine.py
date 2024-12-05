from typing import Dict, Union

import pyomo.environ as pyo

from oogeso import dto
from oogeso.core.devices.base import Device


class GasTurbine(Device):
    """Gas turbine generator"""

    carrier_in = ["gas", "hydrogen"]
    carrier_out = ["el", "heat", "carbon"]
    serial = list()

    def __init__(
        self,
        dev_data: dto.DeviceGasTurbineData,
        carrier_data_dict: Dict[str, dto.CarrierGasData],
    ):
        super().__init__(dev_data=dev_data, carrier_data_dict=carrier_data_dict)
        self.dev_data = dev_data
        self.id = dev_data.id
        self.carrier_data = carrier_data_dict

    def _rules_misc(self, model: pyo.Model, t: int, i: int) -> Union[pyo.Expression, pyo.Constraint.Skip]:
        dev = self.id
        gas_data = self.carrier_data["gas"]
        gas_energy_content = gas_data.energy_value  # MJ/Sm3
        has_hydrogen = "hydrogen" in model.setCarrier
        if has_hydrogen:
            hydrogen_data = self.carrier_data["hydrogen"]
            hydrogen_energy_content = hydrogen_data.energy_value  # MJ/Sm3
        if i == 1:
            """turbine el power out vs gas fuel in"""
            # fuel consumption (gas+hydrogen in) is a linear function of el power output
            # fuel = B + A*power
            # => efficiency = power/(A+B*power)
            A = self.dev_data.fuel_A
            B = self.dev_data.fuel_B
            P_max = self.dev_data.flow_max
            lhs = model.varDeviceFlow[dev, "gas", "in", t] * gas_energy_content / P_max
            if has_hydrogen:
                lhs += model.varDeviceFlow[dev, "hydrogen", "in", t] * hydrogen_energy_content / P_max
            rhs = (
                B * (model.varDeviceIsOn[dev, t] + model.varDeviceIsPrep[dev, t])
                + A * model.varDeviceFlow[dev, "el", "out", t] / P_max
            )
            return lhs == rhs
        elif i == 2:
            """heat output = (energy in - el power out)* heat efficiency"""
            energy_in = model.varDeviceFlow[dev, "gas", "in", t] * gas_energy_content
            if has_hydrogen:
                energy_in += model.varDeviceFlow[dev, "hydrogen", "in", t] * hydrogen_energy_content
            lhs = model.varDeviceFlow[dev, "heat", "out", t]
            rhs = (energy_in - model.varDeviceFlow[dev, "el", "out", t]) * self.dev_data.eta_heat
            return lhs == rhs
        elif i == 3:
            """carbon emitted given by gas consumption and EGR. The value is co2 after recirculation"""
            gasflow_co2 = gas_data.co2_content  # kgCO2/Sm3gas
            egr = self.dev_data.exhaust_gas_recirculation
            if egr == 0:
                y_egr = 1  # to not break with previous test cases
            elif 0 <= egr <= 0.3:
                y_egr = 1.411 * egr + 0.9838
            elif 0.3 < egr <= 0.6:
                y_egr = 3.5142 * egr + 0.3089
            else:
                raise ValueError("EGR parameter invalid.")
            lhs = model.varDeviceFlow[dev, "carbon", "out", t]
            # multiply by (1-egr) to give flow after gas for recycling has been taken out:
            rhs = gasflow_co2 * y_egr * (1 - egr) * (model.varDeviceFlow[dev, "gas", "in", t])
            return lhs == rhs
        elif (i == 4) and (has_hydrogen):
            """hydrogen blend max"""
            flow_total = (
                model.varDeviceFlow[dev, "gas", "in", t] + model.varDeviceFlow[dev, "hydrogen", "in", t]
            )  # Sm3/s
            return model.varDeviceFlow[dev, "hydrogen", "in", t] <= self.dev_data.hydrogen_blend_max * flow_total
        elif (i == 5) and (has_hydrogen):
            """hydrogen blend min"""
            flow_total = (
                model.varDeviceFlow[dev, "gas", "in", t] + model.varDeviceFlow[dev, "hydrogen", "in", t]
            )  # Sm3/s
            return model.varDeviceFlow[dev, "hydrogen", "in", t] >= self.dev_data.hydrogen_blend_min * flow_total
        else:
            return pyo.Constraint.Skip

    def define_constraints(self, pyomo_model: pyo.Model):
        """Specifies the list of constraints for the device"""
        list_to_reconstruct = super().define_constraints(pyomo_model)

        constr = pyo.Constraint(pyomo_model.setHorizon, pyo.RangeSet(1, 5), rule=self._rules_misc)
        setattr(pyomo_model, "constr_{}_{}".format(self.id, "misc"), constr)
        return list_to_reconstruct

    def get_flow_var(self, pyomo_model: pyo.Model, t: int):
        return pyomo_model.varDeviceFlow[self.id, "el", "out", t]


class SteamCycle(Device):
    """Steam cycle generator"""

    carrier_in = ["heat"]
    carrier_out = ["el", "heat"]
    serial = ["heat"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    #    def __init__(
    #        self,
    #        dev_data: dto.DeviceSteamCycleData,
    #        carrier_data_dict: Dict[str, dto.CarrierData],
    #    ):
    #        super().__init__(dev_data=dev_data, carrier_data_dict=carrier_data_dict)
    #        self.dev_data = dev_data
    #        self.id = dev_data.id
    #        self.carrier_data = carrier_data_dict

    def _rules_misc(self, model: pyo.Model, t: int, i: int) -> Union[pyo.Expression, pyo.Constraint.Skip]:
        dev = self.id
        alpha = self.dev_data.alpha
        egr = self.dev_data.exhaust_gas_recirculation
        p_sc_nominal = self.dev_data.flow_max
        if i == 1:
            """power out vs heat in, expressed in normalised variables"""
            # heat consumption is a linear function of el power output
            # fuel = B + A*power
            # => efficiency = power/(A+B*power)

            # TODO make this user input
            (linA, linB) = (1.96, -0.96)

            heat_input_norm = model.varDeviceFlow[dev, "heat", "in", t] / p_sc_nominal
            heat_extracted = model.varDeviceFlow[dev, "heat", "out", t]
            el_output = model.varDeviceFlow[dev, "el", "out", t]

            # normalised equation
            y_heat = linA * heat_input_norm + linB * (model.varDeviceIsOn[dev, t] + model.varDeviceIsPrep[dev, t])
            # egr dependency (piecewise linear factor):
            if egr < 0.1:
                y_egr = 0.1763 * egr + 1.0
            else:
                y_egr = 0.0225 * egr + 1.0153
            power_computed = y_heat * y_egr * p_sc_nominal
            # power equivalent of steam extracted for CCS (1/alpha >1 since 1 unit heat gives < 1 unit el)
            power_extracted = heat_extracted / alpha
            return el_output == power_computed - power_extracted
        elif i == 2:
            # TODO: this should not be necessary
            # make sure that energy out < energy in
            # heat out <= heat in
            return pyo.Constraint.Skip
            max_efficiency = 1.0
            energy_in = model.varDeviceFlow[dev, "heat", "in", t]
            energy_out = model.varDeviceFlow[dev, "heat", "out", t] + model.varDeviceFlow[dev, "el", "out", t]
            return energy_out <= energy_in * max_efficiency
        else:
            return pyo.Constraint.Skip

    def define_constraints(self, pyomo_model: pyo.Model):
        """Specifies the list of constraints for the device"""
        list_to_reconstruct = super().define_constraints(pyomo_model)

        constr = pyo.Constraint(pyomo_model.setHorizon, pyo.RangeSet(1, 2), rule=self._rules_misc)
        setattr(pyomo_model, "constr_{}_{}".format(self.id, "misc"), constr)
        return list_to_reconstruct

    def get_flow_var(self, pyomo_model: pyo.Model, t: int):
        return pyomo_model.varDeviceFlow[self.id, "el", "out", t]
