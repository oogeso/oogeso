from abc import abstractmethod
from typing import Dict, List, Union

import pyomo.environ as pyo

from oogeso import dto
from oogeso.core.devices.base import Device
from oogeso.core.networks.network_node import NetworkNode


class StorageDevice(Device):
    @abstractmethod
    def get_flow_var(self, pyomo_model: pyo.Model, t: int):
        pass

    def compute_cost_for_depleted_storage(self, pyomo_model: pyo.Model, timesteps: pyo.Set):
        # cost rate CURRENCY/s
        # Cost associated with deviation from max value at end of horizon
        dev_data = self.dev_data
        if dev_data.E_cost is None:
            return 0
        if dev_data.E_max is None:
            return 0
        t_end = timesteps.at(-1)
        var_E = pyomo_model.varDeviceStorageEnergy[self.id, t_end]
        storage_cost = dev_data.E_cost * (dev_data.E_max - var_E)
        # from cost (CURRENCY) to cost rate (CURRENCY/s):
        # storage_cost = storage_cost / len(timesteps)  # 2024-11-01: No, don't divide - that means E_cost parameter must be changed if horizon length is changed.
        return storage_cost


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


class StorageHydrogen(StorageDevice):
    """Hydrogen storage"""

    carrier_in = ["hydrogen", "el"]
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
            # electricity demand for hydrogen compression
            lhs = pyomo_model.varDeviceFlow[dev, "el", "in", t]
            if not dev_data.compressor_include:
                # No compressor el demand
                rhs = 0
            else:
                isothermal_adiabatic = dev_data.compressor_isothermal_adiabatic
                power_demand = compute_hydrogen_compressor_demand(pyomo_model, self, isothermal_adiabatic, t=t)
                rhs = power_demand / dev_data.compressor_eta
            return lhs == rhs
        else:
            raise ValueError(f"Argument i must be 1, 2, 3. {i} was given.")

    def define_constraints(self, pyomo_model: pyo.Model) -> List[pyo.Constraint]:
        """Specifies the list of constraints for the device"""

        list_to_reconstruct = super().define_constraints(pyomo_model)

        constr = pyo.Constraint(pyomo_model.setHorizon, pyo.RangeSet(1, 3), rule=self._rules)
        # add constraints to model:
        setattr(pyomo_model, "constr_{}_{}".format(self.id, "misc"), constr)
        return list_to_reconstruct

    def get_flow_var(self, pyomo_model: pyo.Model, t: int):
        return pyomo_model.varDeviceFlow[self.id, "hydrogen", "out", t]

    def get_tank_state(self, pyomo_model, t=None):
        hydrogen_data = self.carrier_data["hydrogen"]
        Z = hydrogen_data.Z_compressibility
        Rspec = hydrogen_data.R_individual_gas_constant
        rho = hydrogen_data.rho_density
        T_in = self.dev_data.compressor_temperature

        tank_p_max_MPa = self.dev_data.compressor_pressure_max
        tank_E_max_Sm3 = self.dev_data.E_max
        tank_volume_m3 = tank_E_max_Sm3 * rho * Rspec * T_in * Z / (tank_p_max_MPa * 1e6)

        if t is not None:
            tank_E_Sm3 = pyomo_model.varDeviceStorageEnergy[self.id, t]
        else:
            tank_E_Sm3 = pyomo_model.paramDeviceEnergyInitially[self.id]

        p_tank_MPa = tank_E_Sm3 * rho * Rspec * T_in * Z / tank_volume_m3 * 1e-6
        dct = {}
        dct["pressure_MPa"] = pyo.value(p_tank_MPa)
        dct["volume_m3"] = tank_volume_m3
        dct["E_Sm3"] = pyo.value(tank_E_Sm3)
        dct["E_max_Sm3"] = tank_E_max_Sm3
        dct["filling_level"] = pyo.value(tank_E_Sm3 / tank_E_max_Sm3)
        return dct


def compute_hydrogen_compressor_demand(
    model: pyo.Model, device_obj: StorageHydrogen, isothermal_adiabatic: float, t: int
) -> float:
    """Compute hydrogen compressor energy demand

    NOTE: We use the initial storage level to compute the pressure. This is an approzimation since the pressure
    changes during the planning horizon. It works OK if the storage is large compared to the planning horizon.
    """

    hydrogen_data = device_obj.carrier_data["hydrogen"]
    gamma = hydrogen_data.adiabatic_index
    Z = hydrogen_data.Z_compressibility
    Rspec = hydrogen_data.R_individual_gas_constant
    rho = hydrogen_data.rho_density  # kg/Sm3
    p_in = device_obj.dev_data.compressor_pressure_in  # MPa
    T_in = device_obj.dev_data.compressor_temperature  # K

    tank_p_max_MPa = device_obj.dev_data.compressor_pressure_max
    tank_E_max_Sm3 = device_obj.dev_data.E_max
    # tank_volume_m3 = tank_E_max_Sm3 * rho * Rspec * T_in * Z / (tank_p_max_MPa*1e6)

    # H2 tank is derived from stored energy
    # tank_E_Sm3 = model.varDeviceStorageEnergy[device_obj.id,t] #variable -> nonlinear terms
    tank_E_Sm3 = model.paramDeviceEnergyInitially[device_obj.id]  # parameter
    # pressure_init_MPa = tank_E_Sm3 * rho * Rspec * T_in * Z / tank_volume  # this is the same as next line
    pressure_init_MPa = tank_p_max_MPa * tank_E_Sm3 / tank_E_max_Sm3
    # print(f"storage.py:356 tank_p_max={tank_p_max_MPa} MPa, tank_vol={tank_volume_m3:g} m3, tank_E_Sm3={pyo.value(tank_E_Sm3)}, t={t}")

    dE = model.varDeviceFlow[device_obj.id, "hydrogen", "in", t]

    if (isothermal_adiabatic > 1) or (isothermal_adiabatic < 0):
        raise ValueError("isothermal_adiabatic paramater must be in the range 0-1")
    if isothermal_adiabatic == 0:
        # isothermal
        dW = isothermal_work(dE, pressure_init_MPa, rho, Rspec, T_in, Z, p_in)
    elif isothermal_adiabatic == 1:
        # adiabatic
        dW = adiabatic_work(dE, pressure_init_MPa, rho, Rspec, T_in, Z, p_in, gamma)
    else:
        # interpolate between the two:
        dW_iso = isothermal_work(dE, pressure_init_MPa, rho, Rspec, T_in, Z, p_in)
        dW_ad = adiabatic_work(dE, pressure_init_MPa, rho, Rspec, T_in, Z, p_in, gamma)
        dW = (1 - isothermal_adiabatic) * dW_iso + isothermal_adiabatic * dW_ad
    P = dW * 1e-6  # W to MW
    return P


def isothermal_work(dE_extra, pressure_init, rho, Rspec, T_in, Z, p_in):
    """Return the work required to compress the E_extra isothermally from p_initial to p_final

    Ref Sondre Wennberg master thesis, 2022 (eq.2.36)
    Here: Ignoring changes in compressibility with pressure
    """
    dW = dE_extra * rho * Rspec * T_in * Z * pyo.log(pressure_init / p_in)
    return dW


def adiabatic_work(dE_extra, pressure_init, rho, Rspec, T_in, Z, p_in, gamma):
    """Return the work required to compress the E_extra adiabatically from p_initial to p_final

    Ref Sondre Wennberg master thesis, 2022 (eq.2.36)
    Here: Ignoring changes in compressibility with pressure
    """
    dW = dE_extra * rho * Rspec * T_in * Z * gamma / (gamma - 1) * ((pressure_init / p_in) ** ((gamma - 1) / gamma) - 1)
    return dW


def NOT_USED_compressibility_factor(p: float, T: float):
    """Return the compressibility factor of hydrogen given the pressure (in MPa) and temperature (in K)

    ref Sondre Wennberg master thesis, 2022
    """
    a = [
        0.05888460,
        -0.06136111,
        -0.002650473,
        0.002731125,
        0.001802374,
        -0.001150707,
        0.9588528e-4,
        -0.1109040e-6,
        0.1264403e-9,
    ]
    b = [1.325, 1.87, 2.5, 2.8, 2.938, 3.14, 3.37, 3.75, 4.0]
    c = [1.0, 1.0, 2.0, 2.0, 2.42, 2.63, 3.0, 4.0, 5.0]
    Z = 1.0
    for i in range(len(a)):
        Z += a[i] * (100 / T) ** b[i] * p ** c[i]
    return Z


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


class WaterInjection(StorageDevice):
    """
    Water injection - flexible water sink
    """

    carrier_in = ["water"]
    carrier_out = []
    serial = []

    def __init__(
        self,
        dev_data: dto.DeviceWaterInjectionData,
        carrier_data_dict: Dict[str, dto.CarrierWaterData],
    ):
        super().__init__(dev_data=dev_data, carrier_data_dict=carrier_data_dict)
        self.dev_data = dev_data
        self.id = dev_data.id
        self.carrier_data = carrier_data_dict

    def rule_devmodel_sink_water(self, model: pyo.Model, t: int, i: int):
        dev = self.id
        dev_data = self.dev_data
        time_delta_minutes = model.paramTimestepDeltaMinutes

        if dev_data.flow_avg is None:
            return pyo.Constraint.Skip
        if dev_data.E_max is None:
            return pyo.Constraint.Skip
        if dev_data.E_max == 0:
            return pyo.Constraint.Skip
        if i == 1:
            # FLEXIBILITY
            # (water_in-water_avg)*dt = delta buffer
            delta_t = time_delta_minutes / 60  # hours
            lhs = (model.varDeviceFlow[dev, "water", "in", t] - dev_data.flow_avg) * delta_t
            if t > 0:
                E_prev = model.varDeviceStorageEnergy[dev, t - 1]
            else:
                E_prev = model.paramDeviceEnergyInitially[dev]
            rhs = model.varDeviceStorageEnergy[dev, t] - E_prev
            return lhs == rhs
        elif i == 2:
            # energy buffer limit
            E_max = dev_data.E_max
            E_min = dev_data.E_min
            return pyo.inequality(E_min, model.varDeviceStorageEnergy[dev, t], E_max)
        else:
            raise Exception("impossible")

    def define_constraints(self, pyomo_model: pyo.Model):
        """Specifies the list of constraints for the device"""
        list_to_reconstruct = super().define_constraints(pyomo_model)

        constr = pyo.Constraint(
            pyomo_model.setHorizon,
            pyo.RangeSet(1, 2),
            rule=self.rule_devmodel_sink_water,
        )
        # add constraints to model:
        setattr(pyomo_model, "constr_{}_{}".format(self.id, "flex"), constr)
        return list_to_reconstruct

    def get_flow_var(self, pyomo_model: pyo.Model, t: int):
        return pyomo_model.varDeviceFlow[self.id, "water", "in", t]
