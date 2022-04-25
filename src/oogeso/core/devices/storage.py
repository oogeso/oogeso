from abc import abstractmethod
from typing import Dict, List, Optional, Union

import pyomo.environ as pyo
import numpy as np
from zmq import device

from oogeso import dto
from oogeso.core.devices.base import Device


class StorageDevice(Device):
    @abstractmethod
    def get_flow_var(self, pyomo_model: pyo.Model, t: int):
        pass


class StorageEl(StorageDevice):
    """Electric storage (battery)"""

    carrier_in = ["el"]
    carrier_out = ["el"]
    serial = []

    def __init__(
        self,
        dev_data: dto.DeviceStorageElData,
        carrier_data_dict: Dict[str, dto.CarrierElData],
    ):
        super().__init__(dev_data=dev_data, carrier_data_dict=carrier_data_dict)
        self.dev_data = dev_data
        self.id = dev_data.id
        self.carrier_data = carrier_data_dict

    def _rules(self, pyomo_model: pyo.Model, t: int, i: int) -> Union[pyo.Expression, pyo.Constraint.Skip]:
        dev = self.id
        dev_data: dto.DeviceStorageElData = self.dev_data
        time_delta_minutes = pyomo_model.paramTimestepDeltaMinutes
        el_data = self.carrier_data["el"]
        time_reserve_minutes = el_data.reserve_storage_minutes
        if i == 1:
            # energy balance
            # (el_in*eta - el_out/eta)*dt = delta storage
            # eta = efficiency charging and discharging
            delta_t = time_delta_minutes / 60  # hours
            lhs = (
                pyomo_model.varDeviceFlow[dev, "el", "in", t] * dev_data.eta
                - pyomo_model.varDeviceFlow[dev, "el", "out", t] / dev_data.eta
            ) * delta_t
            if t > 0:
                E_prev = pyomo_model.varDeviceStorageEnergy[dev, t - 1]
            else:
                E_prev = pyomo_model.paramDeviceEnergyInitially[dev]
            rhs = pyomo_model.varDeviceStorageEnergy[dev, t] - E_prev
            return lhs == rhs
        elif i == 2:
            # energy storage limit
            ub = dev_data.E_max
            lb = dev_data.E_min
            # Specifying Emin may be useful e.g. to impose that battery
            # should only be used for reserve
            # BUT - probably better to specify an energy depletion cost
            # instead (Ecost) - to allow using battery (at a cost) if required
            return pyo.inequality(lb, pyomo_model.varDeviceStorageEnergy[dev, t], ub)
        elif i == 3:
            # return pyo.Constraint.Skip # unnecessary -> generic Pmax/min constraints
            # discharging power limit
            ub = dev_data.flow_max
            return pyomo_model.varDeviceFlow[dev, "el", "out", t] <= ub
        elif i == 4:
            # charging power limit - required because the generic flow max/min constraint
            # concerns dis-charging [dev,"el","out",t], cf get_flow_var()
            if dev_data.flow_min is not None:
                ub = -dev_data.flow_min
            else:
                # assume max charging power is the same as discharging power (Pmax)
                ub = dev_data.flow_max  # <- see generic Pmax/min constr
            return pyomo_model.varDeviceFlow[dev, "el", "in", t] <= ub
        elif i == 5:
            # Constraint 5-8: varDeviceStoragePmax = min{Pmax,E/dt}
            # ref: https://or.stackexchange.com/a/1174
            lhs = pyomo_model.varDeviceStoragePmax[dev, t]
            rhs = dev_data.flow_max
            return lhs <= rhs
        elif i == 6:
            lhs = pyomo_model.varDeviceStoragePmax[dev, t]
            # Parameter specifying for how long the power needs to be
            # sustained to count as reserve (e.g. similar to GT startup time)
            dt_hours = time_reserve_minutes / 60
            rhs = pyomo_model.varDeviceStorageEnergy[dev, t] / dt_hours
            return lhs <= rhs
        elif i == 7:
            big_M = 10 * dev_data.flow_max
            lhs = pyomo_model.varDeviceStoragePmax[dev, t]
            rhs = dev_data.flow_max - big_M * (1 - pyomo_model.varStorY[dev, t])
            return lhs >= rhs
        elif i == 8:
            dt_hours = time_reserve_minutes / 60
            big_M = 10 * dev_data.E_max / dt_hours
            lhs = pyomo_model.varDeviceStoragePmax[dev, t]
            rhs = pyomo_model.varDeviceStorageEnergy[dev, t] / dt_hours - big_M * pyomo_model.varStorY[dev, t]
            return lhs >= rhs
        elif i == 9:
            # constraint on storage end vs start
            # Adding this does not seem to improve result (not lower CO2)
            if (dev_data.E_end is not None) and (t == pyomo_model.setHorizon.last()):
                lhs = pyomo_model.varDeviceStorageEnergy[dev, t]
                # rhs = model.varDeviceStorageEnergy[dev,0] # end=start
                rhs = dev_data.E_end
                return lhs == rhs
            else:
                return pyo.Constraint.Skip  # noqa
        elif i == 10:
            # Constraint 5-7: varDeviceFlow[dev, "hydrogen", "in", t] and varDeviceFlow[dev, "hydrogen", "out", t] cannot both be nonzero
            # ref: https://stackoverflow.com/questions/71372177/
            big_M = dev_data.E_max
            lhs = pyomo_model.varDeviceFlow[dev, "el", "in", t]
            rhs = pyomo_model.varStorIn[dev, t] * big_M
            return lhs <= rhs
        elif i == 11:
            big_M = dev_data.E_max
            lhs = pyomo_model.varDeviceFlow[dev, "el", "out", t]
            rhs = pyomo_model.varStorOut[dev, t] * big_M
            return lhs <= rhs
        elif i == 12:
            lhs = pyomo_model.varStorIn[dev, t] + pyomo_model.varStorOut[dev, t]
            rhs = 1
            return lhs <= rhs

    def define_constraints(self, pyomo_model: pyo.Model) -> List[pyo.Constraint]:
        """Specifies the list of constraints for the device"""

        list_to_reconstruct = super().define_constraints(pyomo_model)

        constr = pyo.Constraint(pyomo_model.setHorizon, pyo.RangeSet(1, 12), rule=self._rules)
        # add constraints to model:
        setattr(pyomo_model, "constr_{}_{}".format(self.id, "misc"), constr)
        return list_to_reconstruct

    def get_flow_var(self, pyomo_model: pyo.Model, t: int) -> float:
        return pyomo_model.varDeviceFlow[self.id, "el", "out", t]

    def get_max_flow(self, pyomo_model: pyo.Model, t: int) -> float:
        # available power may be limited by energy in the storage
        # charging also contributes (can be reversed)
        # (it can go to e.g. -2 MW to +2MW => 4 MW,
        # even if Pmax=2 MW)
        return pyomo_model.varDeviceStoragePmax[self.id, t] + pyomo_model.varDeviceFlow[self.id, "el", "in", t]

    def compute_cost_for_depleted_storage(
        self, pyomo_model: pyo.Model, timesteps: Optional[Union[List[int], pyo.Set]] = None
    ):

        stor_cost = 0
        dev_data = self.dev_data
        if dev_data.E_cost is not None:
            stor_cost = dev_data.E_cost
        E_max = dev_data.E_max
        store_cost = 0
        for t in timesteps:
            var_E = pyomo_model.varDeviceStorageEnergy[self.id, t]
            store_cost += stor_cost * (E_max - var_E)
        return store_cost

    # Overriding default
    def compute_operating_costs(self, pyomo_model: pyo.Model, timesteps: Union[pyo.Set, List[int]]) -> float:
        """average operating cost within selected timespan"""
        sum_cost = 0
        if self.dev_data.op_cost is not None and self.dev_data.op_cost_in is None:
            self.dev_data.op_cost_in = self.dev_data.op_cost
        if self.dev_data.op_cost_in is not None:
            op_cost = self.dev_data.op_cost_in
            for t in timesteps: #pyomo_model.setHorizon:
                var_P = pyomo_model.varDeviceFlow[self.id, "el", "in", t]
                sum_cost += op_cost * var_P
        if self.dev_data.op_cost_out is not None:
            op_cost = self.dev_data.op_cost_out
            for t in timesteps: #pyomo_model.setHorizon:
                var_P = pyomo_model.varDeviceFlow[self.id, "el", "out", t]
                sum_cost += op_cost * var_P
        # average per sec (simulation timestep drops out)
        return sum_cost / len(timesteps)

class StorageHydrogen(StorageDevice):
    """Hydrogen storage"""

    carrier_in = ["hydrogen"]
    carrier_out = ["hydrogen"]
    serial = []

    def __init__(
        self,
        dev_data: dto.DeviceStorageHydrogenData,
        carrier_data_dict: Dict[str, dto.CarrierHydrogenData],
    ):
        super().__init__(dev_data=dev_data, carrier_data_dict=carrier_data_dict)
        self.dev_data = dev_data
        self.id = dev_data.id
        self.carrier_data = carrier_data_dict

    def _rules(self, pyomo_model: pyo.Model, t: int, i: int) -> Union[pyo.Expression, pyo.Constraint.Skip]:
        dev = self.id
        dev_data: dto.DeviceStorageHydrogenData = self.dev_data
        # param_hydrogen = self.optimiser.all_carriers["hydrogen"]
        time_delta_minutes = pyomo_model.paramTimestepDeltaMinutes
        if i == 1:
            # energy balance (delta E = in - out) (energy in Sm3)
            delta_t = time_delta_minutes * 60  # seconds
            eta = 1
            if dev_data.eta is not None:
                eta = dev_data.eta
            lhs = (
                pyomo_model.varDeviceFlow[dev, "hydrogen", "in", t] * eta
                - pyomo_model.varDeviceFlow[dev, "hydrogen", "out", t] / eta
            ) * delta_t
            if t > 0:
                E_prev = pyomo_model.varDeviceStorageEnergy[dev, t - 1]
            else:
                E_prev = pyomo_model.paramDeviceEnergyInitially[dev]
            rhs = pyomo_model.varDeviceStorageEnergy[dev, t] - E_prev
            return lhs == rhs
        elif i == 2:
            # energy storage limit
            ub = dev_data.E_max
            lb = dev_data.E_min
            return pyo.inequality(lb, pyomo_model.varDeviceStorageEnergy[dev, t], ub)
        elif i == 3:
            # Constraint 3 and 4: to represent absolute value in obj.function
            # see e.g. http://lpsolve.sourceforge.net/5.1/absolute.htm
            #
            # deviation from target and absolute value at the end of horizon
            if t != pyomo_model.setHorizon.last():
                return pyo.Constraint.Skip  # noqa
            X_prime = pyomo_model.varDeviceStorageDeviationFromTarget[dev]
            # profile = model.paramDevice[dev]['target_profile']
            target_value = pyomo_model.paramDeviceEnergyTarget[dev]
            deviation = pyomo_model.varDeviceStorageEnergy[dev, t] - target_value
            return X_prime >= deviation  # noqa
        elif i == 4:
            # deviation from target and absolute value at the end of horizon
            if t != pyomo_model.setHorizon.last():
                return pyo.Constraint.Skip  # noqa
            X_prime = pyomo_model.varDeviceStorageDeviationFromTarget[dev]
            # profile = model.paramDevice[dev]['target_profile']
            target_value = pyomo_model.paramDeviceEnergyTarget[dev]
            deviation = pyomo_model.varDeviceStorageEnergy[dev, t] - target_value
            return X_prime >= -deviation  # noqa
        else:
            raise ValueError(f"Argument i must be 1, 2, 3 or 4. {i} was given.")

    def define_constraints(self, pyomo_model: pyo.Model) -> List[pyo.Constraint]:
        """Specifies the list of constraints for the device"""

        list_to_reconstruct = super().define_constraints(pyomo_model)

        constr = pyo.Constraint(pyomo_model.setHorizon, pyo.RangeSet(1, 4), rule=self._rules)
        # add constraints to model:
        setattr(pyomo_model, "constr_{}_{}".format(self.id, "misc"), constr)
        return list_to_reconstruct

    def get_flow_var(self, pyomo_model: pyo.Model, t: int):
        return pyomo_model.varDeviceFlow[self.id, "hydrogen", "out", t]

    def compute_cost_for_depleted_storage(
        self, pyomo_model: pyo.Model, timesteps: Optional[Union[List[int], pyo.Set]] = None
    ):

        return self.compute_cost_for_depleted_storage_alt2(pyomo_model, timesteps)

    def compute_cost_for_depleted_storage_alt1(self, pyomo_model: pyo.Model):
        # cost if storage level at end of optimisation deviates from
        # target profile (user input based on expectations)
        # absolute value of deviation (filling too much also a cost)
        # - avoids over-filling storage
        dev = self.id
        dev_data = self.dev_data
        deviation = pyomo_model.varDeviceStorageDeviationFromTarget[dev]
        stor_cost = dev_data.E_cost * deviation
        return stor_cost

    def compute_cost_for_depleted_storage_alt2(self, pyomo_model: pyo.Model, timesteps: pyo.Set):
        # cost rate kr/s
        # Cost associated with deviation from target value
        # below target = cost, above target = benefit   => gives signal to fill storage
        dev_data = self.dev_data
        # E_target = dev_data.E_max
        E_target = pyomo_model.paramDeviceEnergyTarget[self.id]
        t_end = timesteps[-1]
        var_E = pyomo_model.varDeviceStorageEnergy[self.id, t_end]
        storage_cost = dev_data.E_cost * (E_target - var_E)
        # from cost (kr) to cost rate (kr/s):
        storage_cost = storage_cost / len(timesteps)
        return storage_cost


class StorageHydrogenCompressor(StorageDevice):
    """Hydrogen storage with compressor"""

    carrier_in = ["hydrogen", "el"]
    carrier_out = ["hydrogen", "heat"]
    serial = []

    def __init__(
        self,
        dev_data: dto.DeviceStorageHydrogenCompressorData,
        carrier_data_dict: Dict[str, dto.CarrierHydrogenData],
    ):
        super().__init__(dev_data=dev_data, carrier_data_dict=carrier_data_dict)
        self.dev_data = dev_data
        self.id = dev_data.id
        self.carrier_data = carrier_data_dict

        if dev_data.compressor.eta_heat == 0 or dev_data.compressor.isothermal_adiabatic == 0:
            self.carrier_out = ["hydrogen"]

        #self.compressibility_factor: float = 1
        self.Z_max = compressibility_factor(dev_data.compressor.p_max, dev_data.compressor.temperature)
        self.Z_min = compressibility_factor(dev_data.compressor.p_in, dev_data.compressor.temperature)
        self.p_init = tank_pressure(self, dev_data.E_init, linear = False)
        self.Z_init = compressibility_factor(self.p_init, dev_data.compressor.temperature)
        self.T_out_ad = dev_data.compressor.temperature * (self.p_init / dev_data.compressor.p_in)**((self.carrier_data["hydrogen"].gamma - 1) / self.carrier_data["hydrogen"].gamma) 
        self.Z_out_ad = compressibility_factor(self.p_init, self.T_out_ad)
        #self.ad_T_end_a, self.ad_T_end_b = T_ad_out_coefficients(self)

    def _rules(self, pyomo_model: pyo.Model, t: int, i: int) -> Union[pyo.Expression, pyo.Constraint.Skip]:
        dev = self.id
        dev_data: dto.DeviceStorageHydrogenData = self.dev_data
        #hydrogen_data: dto.CarrierHydrogenData = self.carrier_data
        # param_hydrogen = self.optimiser.all_carriers["hydrogen"]
        time_delta_minutes = pyomo_model.paramTimestepDeltaMinutes
        if i == 1:
            # energy balance (delta E = in - out) (energy in Sm3)
            delta_t = time_delta_minutes * 60  # seconds
            eta = 1
            if dev_data.eta is not None:
                eta = dev_data.eta
            lhs = (
                pyomo_model.varDeviceFlow[dev, "hydrogen", "in", t] * eta
                - pyomo_model.varDeviceFlow[dev, "hydrogen", "out", t] / eta
            ) * delta_t
            if t > 0:
                E_prev = pyomo_model.varDeviceStorageEnergy[dev, t - 1]
            else:
                E_prev = pyomo_model.paramDeviceEnergyInitially[dev]
            rhs = pyomo_model.varDeviceStorageEnergy[dev, t] - E_prev
            return lhs == rhs
        elif i == 2:
            # energy storage limit
            ub = dev_data.E_max
            lb = dev_data.E_min
            return pyo.inequality(lb, pyomo_model.varDeviceStorageEnergy[dev, t], ub)
        elif i == 3:
            """Device compressor demand"""
            if t > 0:
                #p_prev = tank_pressure(self, pyomo_model.varDeviceStorageEnergy[dev, t-1])
                #p_prev = tank_pressure(self, pyomo_model.paramDeviceEnergyInitially[dev])
                self.p_init = tank_pressure(self, pyomo_model.paramDeviceEnergyInitially[dev], linear = False)
                self.Z_init = compressibility_factor(self.p_init, dev_data.compressor.temperature)
                if dev_data.compressor.isothermal_adiabatic != 0:
                    self.T_out_ad = dev_data.compressor.temperature * (self.p_init / dev_data.compressor.p_in)**((self.carrier_data["hydrogen"].gamma - 1) / self.carrier_data["hydrogen"].gamma) 
                    self.Z_out_ad = compressibility_factor(self.p_init, self.T_out_ad)
            #else:
                #p_prev = tank_pressure(self, pyomo_model.paramDeviceEnergyInitially[dev])
            #self.compressibility_factor = compressibility_factor(p_prev, dev_data.compressor.temperature)
            lhs = pyomo_model.varDeviceCompressorEnergy[dev, t]
            rhs = compute_hydrogen_compressor_demand(pyomo_model, self, 
                isothermal_adiabatic = dev_data.compressor.isothermal_adiabatic, t=t) / dev_data.compressor.eta
            return lhs == rhs
        elif i == 4:
            # Constraint 4-7: varDeviceFlow[dev, "el", "in", t] = max{varDeviceCompressorEnergy,0}
            # ref: https://or.stackexchange.com/a/1174
            lhs = pyomo_model.varDeviceCompressorEnergy[dev, t]
            rhs = pyomo_model.varDeviceFlow[dev, "el", "in", t]
            return lhs <= rhs
        elif i == 5:
            lhs = 0
            rhs = pyomo_model.varDeviceFlow[dev, "el", "in", t]
            return lhs <= rhs
        # elif i == 6:
        #     big_M = dev_data.E_max / 10 # Taking out 10 % of max storage in one timestep is quite big
        #     lhs = pyomo_model.varDeviceFlow[dev, "el", "in", t]
        #     rhs = big_M #pyomo_model.varDeviceCompressorEnergy[dev, t] + big_M * (1 - pyomo_model.varStorY[dev, t])
        #     return lhs <= rhs
        elif i == 6:
            #big_M = dev_data.E_max / 10 # Taking out 10 % of max storage in one timestep is quite big
            lhs = pyomo_model.varDeviceFlow[dev, "el", "in", t]
            rhs = pyomo_model.varDeviceCompressorEnergy[dev, t] * pyomo_model.varStorY[dev,t] #0 + big_M * pyomo_model.varStorY[dev, t]
            return lhs <= rhs
        # elif i == 4:
        #     """Set the el demand equal to the compressor demand"""
        #     lhs = pyomo_model.varDeviceFlow[dev, "el", "in", t]
        #     rhs = pyomo_model.varDeviceCompressorEnergy[dev, t]
        #     return lhs == rhs
        elif i == 7:
            # Constraint 8-10: varDeviceFlow[dev, "hydrogen", "in", t] and varDeviceFlow[dev, "hydrogen", "out", t] cannot both be nonzero
            # ref: https://stackoverflow.com/questions/71372177/
            big_M = dev_data.E_max / 10 # Taking out 10 % of max storage in one timestep is quite big
            lhs = pyomo_model.varDeviceFlow[dev, "hydrogen", "in", t]
            rhs = pyomo_model.varStorIn[dev, t] * big_M
            return lhs <= rhs
        elif i == 8:
            big_M = dev_data.E_max / 10 # Taking out 10 % of max storage in one timestep is quite big
            lhs = pyomo_model.varDeviceFlow[dev, "hydrogen", "out", t]
            rhs = pyomo_model.varStorOut[dev, t] * big_M
            return lhs <= rhs
        elif i == 9:
            lhs = pyomo_model.varStorIn[dev, t] + pyomo_model.varStorOut[dev, t]
            rhs = 1
            return lhs <= rhs
        elif i == 10:
            """Device isothermal compressor demand"""
            #if t > 0:
                #p_prev = tank_pressure(self, pyomo_model.varDeviceStorageEnergy[dev, t-1])
                #p_prev = tank_pressure(self, pyomo_model.paramDeviceEnergyInitially[dev])
                #self.p_init = tank_pressure(self, pyomo_model.paramDeviceEnergyInitially[dev], linear = False)
                #self.Z_init = compressibility_factor(self.p_init, dev_data.compressor.temperature)
            #else:
                #p_prev = tank_pressure(self, pyomo_model.paramDeviceEnergyInitially[dev])
            #self.compressibility_factor = compressibility_factor(p_prev, dev_data.compressor.temperature)
            lhs = pyomo_model.varDeviceCompressorEnergyIso[dev, t]
            rhs = compute_hydrogen_compressor_demand(pyomo_model, self, 
                isothermal_adiabatic = 0, t=t) / self.dev_data.compressor.eta
            return lhs == rhs
        elif i == 11:
            # Constraint 12-15: varDeviceCompressorPIso[dev, t] = max{varDeviceCompressorEnergyIso,0}
            # ref: https://or.stackexchange.com/a/1174
            lhs = pyomo_model.varDeviceCompressorEnergyIso[dev, t]
            rhs = pyomo_model.varDeviceCompressorPIso[dev, t]
            return lhs <= rhs
        elif i == 12:
            lhs = 0
            rhs = pyomo_model.varDeviceCompressorPIso[dev, t]
            return lhs <= rhs
        # elif i == 14:
        #     big_M = dev_data.E_max / 10 # Taking out 10 % of max storage in one timestep is quite big
        #     lhs = pyomo_model.varDeviceCompressorPIso[dev, t]
        #     rhs = pyomo_model.varDeviceCompressorEnergyIso[dev, t] + big_M * (1 - pyomo_model.varStorY2[dev, t])
        #     return lhs <= rhs
        elif i == 13:
            big_M = dev_data.E_max / 10 # Taking out 10 % of max storage in one timestep is quite big
            lhs = pyomo_model.varDeviceCompressorPIso[dev, t]
            rhs = pyomo_model.varDeviceCompressorEnergyIso[dev, t] * pyomo_model.varStorY2[dev,t] #0 + big_M * pyomo_model.varStorY2[dev, t]
            return lhs <= rhs
        elif i == 14:
            """Device heat production"""
            """heat output = (el energy in - isothermal el energy needed) * heat efficiency"""
            lhs = pyomo_model.varDeviceFlow[dev, "heat", "out", t]
            rhs = (
                pyomo_model.varDeviceFlow[dev, "el", "in", t] 
                - pyomo_model.varDeviceCompressorPIso[dev, t]
                ) * self.dev_data.compressor.eta_heat
            return lhs == rhs
        # elif i == 8:
        #     """Device isothermal compressor demand"""
        #     if t > 0:
        #         p_prev = tank_pressure(self, pyomo_model.paramDeviceEnergyInitially[dev])
        #         self.p_init = tank_pressure(self, pyomo_model.paramDeviceEnergyInitially[dev], linear = False)
        #         self.Z_init = compressibility_factor(self.p_init, dev_data.compressor.temperature)
        #     else:
        #         p_prev = tank_pressure(self, pyomo_model.paramDeviceEnergyInitially[dev])
        #     self.compressibility_factor = compressibility_factor(p_prev, dev_data.compressor.temperature)
        #     lhs = pyomo_model.varDeviceCompressorEnergyIso[dev, t]
        #     rhs = compute_hydrogen_compressor_demand(pyomo_model, self, 
        #         isothermal_adiabatic = 0, t=t, linear = True) / self.dev_data.compressor.eta
        #     return lhs == rhs
        # elif i == 9:
        #     # Constraint 9-12: varDeviceFlow[dev, "heat", "out", t] = max{varDeviceCompressorEnergyIso,0}
        #     # ref: https://or.stackexchange.com/a/1174
        #     lhs = pyomo_model.varDeviceCompressorEnergyIso[dev, t]
        #     rhs = pyomo_model.varDeviceFlow[dev, "heat", "out", t]
        #     return lhs <= rhs
        # elif i == 10:
        #     lhs = 0
        #     rhs = pyomo_model.varDeviceFlow[dev, "el", "in", t]
        #     return lhs <= rhs
        # elif i == 11:
        #     big_M = dev_data.E_max / 10 # Taking out 10 % of max storage in one timestep is quite big
        #     lhs = pyomo_model.varDeviceFlow[dev, "el", "in", t]
        #     rhs = pyomo_model.varDeviceCompressorEnergy[dev, t] + big_M * (1 - pyomo_model.varStorY[dev, t])
        #     return lhs <= rhs
        # elif i == 12:
        #     big_M = dev_data.E_max / 10 # Taking out 10 % of max storage in one timestep is quite big
        #     lhs = pyomo_model.varDeviceFlow[dev, "el", "in", t]
        #     rhs = 0 + big_M * (1 - pyomo_model.varStorY[dev, t])
        #     return lhs >= rhs

        # elif i == 5:
        #     # Constraint 5 and 6: to represent absolute value in obj.function
        #     # see e.g. http://lpsolve.sourceforge.net/5.1/absolute.htm
        #     #
        #     # deviation from target and absolute value at the end of horizon
        #     if t != pyomo_model.setHorizon.last():
        #         return pyo.Constraint.Skip  # noqa
        #     X_prime = pyomo_model.varDeviceStorageDeviationFromTarget[dev]
        #     # profile = model.paramDevice[dev]['target_profile']
        #     target_value = pyomo_model.paramDeviceEnergyTarget[dev]
        #     deviation = pyomo_model.varDeviceStorageEnergy[dev, t] - target_value
        #     return X_prime >= deviation  # noqa
        # elif i == 6:
        #     # deviation from target and absolute value at the end of horizon
        #     if t != pyomo_model.setHorizon.last():
        #         return pyo.Constraint.Skip  # noqa
        #     X_prime = pyomo_model.varDeviceStorageDeviationFromTarget[dev]
        #     # profile = model.paramDevice[dev]['target_profile']
        #     target_value = pyomo_model.paramDeviceEnergyTarget[dev]
        #     deviation = pyomo_model.varDeviceStorageEnergy[dev, t] - target_value
        #     return X_prime >= -deviation  # noqa
        else:
            raise ValueError(f"Argument i must be 1, 2, 3 or 4. {i} was given.")

    def define_constraints(self, pyomo_model: pyo.Model) -> List[pyo.Constraint]:
        """Specifies the list of constraints for the device"""
        list_to_reconstruct = super().define_constraints(pyomo_model)

        range_end = 14
        if self.dev_data.compressor.eta_heat == 0 or self.dev_data.compressor.isothermal_adiabatic == 0:
            range_end = 9
        constr = pyo.Constraint(pyomo_model.setHorizon, pyo.RangeSet(1, range_end), rule=self._rules)
        # add constraints to model:
        setattr(pyomo_model, "constr_{}_{}".format(self.id, "misc"), constr)
        return list_to_reconstruct

    def get_flow_var(self, pyomo_model: pyo.Model, t: int):
        return pyomo_model.varDeviceFlow[self.id, "hydrogen", "out", t]
    
    # Overriding default
    def compute_operating_costs(self, pyomo_model: pyo.Model, timesteps: Union[pyo.Set, List[int]]) -> float:
        """average operating cost within selected timespan"""
        sum_cost = 0
        if self.dev_data.op_cost is not None and self.dev_data.op_cost_in is None:
            self.dev_data.op_cost_in = self.dev_data.op_cost
        if self.dev_data.op_cost_in is not None:
            op_cost_in = self.dev_data.op_cost_in
            for t in timesteps: #pyomo_model.setHorizon:
                h_in = pyomo_model.varDeviceFlow[self.id, "hydrogen", "in", t]
                sum_cost += op_cost_in * h_in
        if self.dev_data.op_cost_out is not None:
            op_cost_out = self.dev_data.op_cost_out
            for t in timesteps: #pyomo_model.setHorizon:
                h_out = pyomo_model.varDeviceFlow[self.id, "hydrogen", "out", t]
                sum_cost += op_cost_out * h_out
        # average per sec (simulation timestep drops out)
        return sum_cost / len(timesteps)

def compute_hydrogen_compressor_demand(
    model: pyo.Model,
    device_obj: StorageHydrogenCompressor,
    isothermal_adiabatic: Optional[float] = 0,
    t: Optional[int] = 0,
    #linear: Optional[bool] = True,
) -> float:
    """Compute energy demand by compressor as function of pressure and flow"""
    # If no hydrogen is added, the compressor need not run
    flow_in = model.varDeviceFlow[device_obj.id, "hydrogen", "in", t]
    # if t > 0:
    #     E_prev = model.varDeviceStorageEnergy[device_obj.id, t - 1]
    # else:
    #E_prev = model.paramDeviceEnergyInitially[device_obj.id]
    # if pyo.value(model.varDeviceFlow[device_obj.id, "hydrogen", "in", t]) == 0:
    #     print(flow_in)
    #     return 0

    #flow_in = 3
    #E_prev = 1.5*10**6
    # Find pressure in tank after this timestep
    time_delta_minutes = model.paramTimestepDeltaMinutes
    delta_t = time_delta_minutes * 60  # seconds
    dev_data = device_obj.dev_data
    eta = 1
    if dev_data.eta is not None:
        eta = dev_data.eta
    #old_pressure = tank_pressure(model, device_obj, t, model.varDeviceStorageEnergy[device_obj.id, t-1])
    extra_stored_energy = flow_in * delta_t * eta #model.varDeviceFlow[device_obj.id, "hydrogen", "in", t] * delta_t * eta
    #final_stored_energy = E_prev + extra_stored_energy #model.varDeviceStorageEnergy[device_obj.id, t] + extra_stored_energy
    #final_pressure = tank_pressure(device_obj, final_stored_energy, linear=linear)

    #return 0.1 * flow_in

    # Isothermal compression
    if isothermal_adiabatic == 0:
        W = isothermal_work(device_obj, extra_stored_energy) #, final_pressure, linear=linear)
        P = W / delta_t * 10**(-6) # Change from work (J) to power (MW)
        # print(pyo.value(P))
        # if pyo.value(P) < 0: return 0
        return P
    elif isothermal_adiabatic == 1:
        W = adiabatic_work(device_obj, extra_stored_energy)
        P = W / delta_t * 10**(-6) # Change from work (J) to power (MW)
        return P
    else:
        W_iso = isothermal_work(device_obj, extra_stored_energy) #, final_pressure, linear=linear)
        W_ad = adiabatic_work(device_obj, extra_stored_energy)
        W = (1 - isothermal_adiabatic) * W_iso + isothermal_adiabatic * W_ad
        P = W / delta_t * 10**(-6) # Change from work (J) to power (MW)
        return P

def tank_pressure(
    device_obj: StorageHydrogenCompressor,
    stored_energy: Optional[float] = 0,
    linear: Optional[bool] = True,
    ):
    # if pyo.value(stored_energy) == 0:
    #     return 0
    hydrogen_data = device_obj.carrier_data["hydrogen"]
    T = device_obj.dev_data.compressor.temperature # temperature of compressed gas at storage conditions
    n = stored_energy * hydrogen_data.rho_density * 10**3 / hydrogen_data.molecular_weight
    R = 8.31446261815324 # Universal gas constant
    V = tank_volume(device_obj)
    p0 = n*R*T/V * 10**(-6) # ideal gas pressure in MPa
    if linear:
        Z_i = device_obj.Z_init
        p = p0*Z_i
    else:
        Z = compressibility_factor(p0, T)
        p = p0*Z #n*Z*R*T/V * 10**(-6) # MPa
        if pyo.value(p) == 0:
            return p
        # Iterate towards the actual pressure
        while abs(pyo.value(p)-pyo.value(p0))/pyo.value(p0) > 0.01: # Cannot evaluate boolean expressions with pyomo variables
        # for i in range(5):
            p0 = p
            Z = compressibility_factor(p0, T)
            p = n*Z*R*T/V * 10**(-6)
    return p

def tank_volume(device_obj: StorageHydrogenCompressor):
    # rho_density is in kg/Nm3, molecular_weight is in g/mol, p_max is in MPa
    hydrogen_data = device_obj.carrier_data["hydrogen"]
    n = device_obj.dev_data.E_max * hydrogen_data.rho_density * 10**3 / hydrogen_data.molecular_weight
    T = device_obj.dev_data.compressor.temperature
    p_max = device_obj.dev_data.compressor.p_max
    Z = device_obj.Z_max #compressibility_factor(p_max, T)
    R = 8.31446261815324 # Universal gas constant
    V = n * Z * R * T / (p_max * 10**6)
    return V

def compressibility_factor(p: float, T: float):
    """Return the compressibility factor of hydrogen given the pressure (in MPa) and temperature (in K)"""
    a = [0.05888460, -0.06136111, -0.002650473, 0.002731125, 0.001802374, -0.001150707, 0.9588528*10**(-4),	-0.1109040*10**(-6), 0.1264403*10**(-9)]
    b = [1.325, 1.87, 2.5, 2.8, 2.938, 3.14, 3.37, 3.75, 4.0]
    c = [1.0, 1.0, 2.0, 2.0, 2.42, 2.63, 3.0, 4.0, 5.0]
    Z = 1.0
    for i in range(len(a)):
        Z += a[i] * (100/T)**b[i] * p**c[i]
    return Z

# def compressibility_factor(p: float, T: float):
#     """Return the compressibility factor of hydrogen given the pressure (in MPa) and temperature (in K)"""
#     v = np.array([[1.0000045, 3.0544750*10**(-4), -1.4584529*10**(-3), 2.3446090*10**(-3), 5.2038693*10**(-4)],
#         [-4.2784574*10**(-4), 2.4749386*10**(-2), -8.7630507*10**(-3), -3.2005290*10**(-2), 1.3585895*10**(-2)],
#         [-5.0333548*10**(-6), 7.6023745*10**(-5), -6.1296001*10**(-4), 1.5893975*10**(-3), 2.7810851*10**(-4)],
#         [3.2594090*10**(-7), -3.7712118*10**(-6), 1.0613715*10**(-5), 2.4601259*10**(-5), -7.6911218*10**(-5)],
#         [-3.2471019*10**(-9), -1.0412400*10**(-8), 5.5294504*10**(-7), -3.1209636*10**(-6), 3.7161086*10**(-6)],
#         [-8.7250210*10**(-11), 2.5607442*10**(-9), -2.3926081*10**(-8), 8.8101225*10**(-8), -8.6078265*10**(-8)],
#         [2.3036183*10**(-12), -4.8093034*10**(-11),	3.6649684*10**(-10), -1.1668258*10**(-9), 1.0473949*10**(-9)],
#         [-1.9359481*10**(-14), 3.6475193*10**(-13), -2.5499310*10**(-12), 7.5339340*10**(-12), -6.4501480*10**(-12)],
#         [5.6844096*10**(-17), -1.0179270*10**(-15), 6.7813004*10**(-15), -1.9146950*10**(-14), 1.5897260*10**(-14)]])
#     Z = 0
#     for i in range(len(v)):
#         for j in range(len(v[0])):
#             Z += v[i,j] * p**i * (100/T)**j
#     return Z

# def compressibility_factor(p: float, T: float):
#     """Return the compressibility factor of hydrogen given the pressure (in MPa) and temperature (in K)"""
#     v = np.array([[1.00018, -0.0022546, 0.01053, -0.013205],
#         [-0.00067291, 0.028051, -0.024126, -0.0058663],
#         [0.000010817, -0.00012653, 0.00019788, 0.00085677],
#         [-1.4368E-07, 1.2171E-06, 7.7563E-07, -1.7418E-05],
#         [1.2441E-09, -8.965E-09, -1.6711E-08, 1.4697E-07],
#         [-4.4709E-12, 3.0271E-11, 6.3329E-11, -4.6974E-10]])
#     Z = 0
#     for i in range(len(v)):
#         for j in range(len(v[0])):
#             Z += v[i,j] * p**i * (100/T)**j
#     return Z

# def compressibility_factor_linear(device_obj: StorageHydrogenCompressor, p: float):
#     """Assume Z is linear for the (constant) temperature and pressure range"""
#     Z_max = device_obj.Z_max
#     Z_min = device_obj.Z_min
#     p_max = device_obj.dev_data.compressor.p_max
#     p_min = device_obj.dev_data.compressor.p_in
#     return (Z_max - Z_min) / (p_max - p_min) * (p - p_min) + Z_min

# def T_ad_out_coefficients(
#     device_obj: StorageHydrogenCompressor,
#     ):
#     """Compute the coefficients for the line T_out = a*T_in + b for adiabatic end
#     temperatures based on beginning temperature"""
#     p_in = device_obj.dev_data.compressor.p_in
#     p_max = device_obj.dev_data.compressor.p_max
#     T_min = device_obj.dev_data.compressor.T_min
#     T_max = device_obj.dev_data.compressor.T_max
#     gamma = device_obj.carrier_data["hydrogen"].gamma
#     T_in_range = np.linspace(T_min, T_max)
#     p_range = np.linspace(p_in, p_max)
#     T_ad = np.zeros((len(T_in_range),len(p_range)))
#     for i in range(len(T_in_range)):
#         T_in = T_in_range[i]
#         T_out = T_in_range[i]
#         T_ad[i][0] = T_out
#         for j in range(len(p_range)):
#             if j > 0:
#                 T_in = T_out
#                 T_out = T_in * (p_range[j] / p_range[j-1])**((gamma - 1) / gamma)
#                 T_ad[i][j] = T_out
#     a = ((T_ad[:,-1][-1] - T_ad[:,-1][0]) / (T_in_range[-1] - T_in_range[0]))
#     b = T_out = a * (- T_in_range[0]) + T_ad[:,-1][0]
#     return a, b

def isothermal_work(
    device_obj: StorageHydrogenCompressor,
    E_extra: float,
    #linear: Optional[float] = True,
    ):
    """Return the work required to compress the E_extra isothermally from p_initial to p_final"""
    # If the initial pressure is greater than the needed final pressure, the compressor need not run
    p_in = device_obj.dev_data.compressor.p_in
    #p_max = device_obj.dev_data.compressor.p_max
    p_init = device_obj.p_init
    #print(p_init)
    #print(pyo.value(p_init))
    #print(p_in)
    #print(pyo.value(p_in))
    # if pyo.value(p_init) < pyo.value(p_in):
    #     return 0
    Z_in = device_obj.Z_min
    #Z_max = device_obj.Z_max
    Z_init = device_obj.Z_init
    T = device_obj.dev_data.compressor.temperature
    R = 8.31446261815324 # Universal gas constant
    hydrogen_data = device_obj.carrier_data["hydrogen"]
    # if p_initial > pyo.value(p_final):
    #     return 0
    # if linear:
    #     Z_final = compressibility_factor_linear(device_obj, p_final)
    # else:
    #     Z_final = compressibility_factor(p_final, T)
    #Z_avg = 0.5*(Z_in + Z_final)
    n = E_extra * hydrogen_data.rho_density * 10**3 / hydrogen_data.molecular_weight
    #W = (Z_in - ((Z_max - Z_in ) / (p_max - p_in)) * p_in) * n * R * T * pyo.log(((Z_in) / (p_in) - (Z_max - Z_in) / (p_max - p_in)) / ((Z_init) / (p_init) - (Z_max - Z_in) / (p_max - p_in)))
    W = n * R * T * 0.5 * (Z_in + Z_init) * pyo.log((p_init * Z_in) / (p_in * Z_init))
    # if linear:
    #     W = n * Z_avg * R * T * pyo.log((Z_in * device_obj.p_init) / (device_obj.Z_init * p_in))
    # else:
    #     W = n * Z_avg * R * T * pyo.log((Z_in * p_final) / (Z_final * p_in))
    #W = R * T * pyo.log(Z_initial/p_initial) * pyo.prod([Z_final, p_final])
    return W

def adiabatic_work(
    device_obj: StorageHydrogenCompressor,
    E_extra: float,
    #linear: Optional[float] = True,
    ):
    hydrogen_data = device_obj.carrier_data["hydrogen"]
    gamma = hydrogen_data.gamma
    p_in = device_obj.dev_data.compressor.p_in
    #p_max = device_obj.dev_data.compressor.p_max
    #T_min = device_obj.dev_data.compressor.T_min
    #T_max = device_obj.dev_data.compressor.T_max
    p_init = device_obj.p_init
    #Z_in = device_obj.Z_min
    #Z_out = device_obj.Z_out_ad
    #Z_max = device_obj.Z_max
    #Z_init = device_obj.Z_init
    T_in = device_obj.dev_data.compressor.temperature
    #T_out = device_obj.T_out_ad #T_in * (p_init / p_in)**((gamma - 1) / gamma) 
    R = 8.31446261815324 # Universal gas constant
    n = E_extra * hydrogen_data.rho_density * 10**3 / hydrogen_data.molecular_weight
    #W = n * R * T * (((Z_max - Z_in) / (p_max - p_in)) * (p_init - p_in - p_in * pyo.log(p_init / p_in)) + Z_in * pyo.log(p_init / p_in))
    #W = - n * R * gamma / (1 - gamma) * (Z_out * T_out - Z_in * T_in)
    
    #W = - n * R * T_in * gamma / (1 - gamma) * 0.5 * (Z_in + Z_out) * ((p_init / p_in)**((gamma - 1) / gamma) - 1)
    W = - n * R * T_in * gamma / (1 - gamma) * ((p_init / p_in)**((gamma - 1) / gamma) - 1)

    return W