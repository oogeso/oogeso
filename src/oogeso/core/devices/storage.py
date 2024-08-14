from abc import abstractmethod
from typing import Dict, List, Optional, Union

import pyomo.environ as pyo

from oogeso import dto
from oogeso.core.devices.base import Device
from oogeso.core.networks.network_node import NetworkNode


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

    def define_constraints(self, pyomo_model: pyo.Model) -> List[pyo.Constraint]:
        """Specifies the list of constraints for the device"""

        list_to_reconstruct = super().define_constraints(pyomo_model)

        constr = pyo.Constraint(pyomo_model.setHorizon, pyo.RangeSet(1, 9), rule=self._rules)
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

    def compute_cost_for_depleted_storage(self, pyomo_model: pyo.Model, timesteps: Optional[pyo.Set] = None):
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

    def compute_cost_for_depleted_storage(self, pyomo_model: pyo.Model, timesteps: Optional[pyo.Set] = None):
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
        t_end = timesteps.at(-1)
        var_E = pyomo_model.varDeviceStorageEnergy[self.id, t_end]
        storage_cost = dev_data.E_cost * (E_target - var_E)
        # from cost (kr) to cost rate (kr/s):
        storage_cost = storage_cost / len(timesteps)
        return storage_cost


class StorageGasLinepack(StorageDevice):
    """Gas storage (line pack)"""

    carrier_in = ["gas"]
    carrier_out = ["gas"]
    serial = []  # not serial, so pressure_out = pressure_in is guaranteed

    def __init__(
        self,
        dev_data: dto.DeviceStorageGasLinepackData,
        carrier_data_dict: Dict[str, dto.CarrierGasData],
    ):
        super().__init__(dev_data=dev_data, carrier_data_dict=carrier_data_dict)
        self.dev_data = dev_data
        self.id = dev_data.id
        self.carrier_data = carrier_data_dict

    def get_flow_var(self, pyomo_model: pyo.Model, t: int):
        return pyomo_model.varDeviceFlow[self.id, "gas", "out", t]

    def define_constraints(self, pyomo_model: pyo.Model) -> List[pyo.Constraint]:
        """Specifies the list of constraints for the device"""

        list_to_reconstruct = super().define_constraints(pyomo_model)

        # set initial condition for storage
        R_individual = self.carrier_data["gas"].R_individual_gas_constant
        rho = self.carrier_data["gas"].rho_density  # kg/Sm3
        T_kelvin = self.carrier_data["gas"].Tb_basetemp_K
        vol = self.dev_data.volume_m3
        self.E_vs_p = vol / (R_individual * T_kelvin * rho) * 1e6  # Pa to MPa
        # Stored matter when pressure equals nominal pressure ( mass= rho E = pV/RT):
        self.pressure_nominal = self.node.get_pressure_nominal("gas", "in")
        self.E_sm3_nominal = self.pressure_nominal * self.E_vs_p
        # print(f"E_vs_p = {self.E_vs_p}")
        # print(f"E_sm3_nominal = {self.E_sm3_nominal}")
        # print(f"E_init = {self.dev_data.E_init}")
        # print(f"Pipeline volume = {vol} m3")

        constr = pyo.Constraint(pyomo_model.setHorizon, pyo.RangeSet(1, 2), rule=self._rules)
        # add constraints to model:
        setattr(pyomo_model, "constr_{}_{}".format(self.id, "misc"), constr)
        return list_to_reconstruct

    def _rules(self, pyomo_model: pyo.Model, t: int, i: int) -> Union[pyo.Expression, pyo.Constraint.Skip]:
        dev = self.id
        time_delta_minutes = pyomo_model.paramTimestepDeltaMinutes
        node_obj: NetworkNode = self.node
        node = node_obj.id

        if i == 1:
            # matter balance (delta E = in - out) (matter in Sm3)
            delta_t = time_delta_minutes * 60  # seconds
            net_inflow = (
                pyomo_model.varDeviceFlow[dev, "gas", "in", t] - pyomo_model.varDeviceFlow[dev, "gas", "out", t]
            ) * delta_t
            if t > 0:
                E_prev = pyomo_model.varDeviceStorageEnergy[dev, t - 1]
            else:
                E_prev = pyomo_model.paramDeviceEnergyInitially[dev]
            delta_storage = pyomo_model.varDeviceStorageEnergy[dev, t] - E_prev
            return net_inflow == delta_storage
        elif i == 2:

            # matter storage vs pressure (deviation from nominal)
            pressure = pyomo_model.varPressure[node, "gas", "in", t]
            mass_stored = pyomo_model.varDeviceStorageEnergy[dev, t]  # Sm3
            if self.E_vs_p == 0:
                # no storage if pipe volume is set to zero.
                return mass_stored == 0
            return pressure == self.pressure_nominal + mass_stored / self.E_vs_p

        else:
            raise ValueError(f"Argument i must be 1 or 2. {i} was given.")
