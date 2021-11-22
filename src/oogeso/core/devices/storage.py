import pyomo.environ as pyo
from oogeso.dto import (
    DeviceStorage_elData,
    DeviceStorage_hydrogenData,
)
from . import Device


class _StorageDevice(Device):
    # pass
    def __init__(self, dev_data, carrier_data_dict):
        """Device object constructor"""
        super().__init__(dev_data, carrier_data_dict)


class Storage_el(_StorageDevice):
    "Electric storage (battery)"
    carrier_in = ["el"]
    carrier_out = ["el"]
    serial = []

    def _rules(self, model, t, i):
        dev = self.id
        dev_data: DeviceStorage_elData = self.dev_data
        time_delta_minutes = model.paramTimestepDeltaMinutes
        time_reserve_minutes = model.paramTimeStorageReserveMinutes

        if i == 1:
            # energy balance
            # (el_in*eta - el_out/eta)*dt = delta storage
            # eta = efficiency charging and discharging
            delta_t = time_delta_minutes / 60  # hours
            lhs = (
                model.varDeviceFlow[dev, "el", "in", t] * dev_data.eta
                - model.varDeviceFlow[dev, "el", "out", t] / dev_data.eta
            ) * delta_t
            if t > 0:
                Eprev = model.varDeviceStorageEnergy[dev, t - 1]
            else:
                Eprev = model.paramDeviceEnergyInitially[dev]
            rhs = model.varDeviceStorageEnergy[dev, t] - Eprev
            return lhs == rhs
        elif i == 2:
            # energy storage limit
            ub = dev_data.E_max
            lb = dev_data.E_min
            # Specifying Emin may be useful e.g. to impose that battery
            # should only be used for reserve
            # BUT - probably better to specify an energy depletion cost
            # instead (Ecost) - to allow using battery (at a cost) if required
            return pyo.inequality(lb, model.varDeviceStorageEnergy[dev, t], ub)
        elif i == 3:
            # return pyo.Constraint.Skip # unnecessary -> generic Pmax/min constraints
            # discharging power limit
            ub = dev_data.flow_max
            return model.varDeviceFlow[dev, "el", "out", t] <= ub
        elif i == 4:
            # charging power limit - required because the generic flow max/min constraint
            # concerns dis-charging [dev,"el","out",t], cf getFlowVar()
            if dev_data.flow_min is not None:
                ub = -dev_data.flow_min
            else:
                # assume max charging power is the same as discharging power (Pmax)
                ub = dev_data.flow_max  # <- see generic Pmax/min constr
            return model.varDeviceFlow[dev, "el", "in", t] <= ub
        elif i == 5:
            # Constraint 5-8: varDeviceStoragePmax = min{Pmax,E/dt}
            # ref: https://or.stackexchange.com/a/1174
            lhs = model.varDeviceStoragePmax[dev, t]
            rhs = dev_data.flow_max
            return lhs <= rhs
        elif i == 6:
            lhs = model.varDeviceStoragePmax[dev, t]
            # Parameter specifying for how long the power needs to be
            # sustained to count as reserve (e.g. similar to GT startup time)
            dt_hours = time_reserve_minutes / 60
            rhs = model.varDeviceStorageEnergy[dev, t] / dt_hours
            return lhs <= rhs
        elif i == 7:
            bigM = 10 * dev_data.flow_max
            lhs = model.varDeviceStoragePmax[dev, t]
            rhs = dev_data.flow_max - bigM * (1 - model.varStorY[dev, t])
            return lhs >= rhs
        elif i == 8:
            dt_hours = time_reserve_minutes / 60
            bigM = 10 * dev_data.E_max / dt_hours
            lhs = model.varDeviceStoragePmax[dev, t]
            rhs = (
                model.varDeviceStorageEnergy[dev, t] / dt_hours
                - bigM * model.varStorY[dev, t]
            )
            return lhs >= rhs
        elif i == 9:
            # constraint on storage end vs start
            # Adding this does not seem to improve result (not lower CO2)
            if (dev_data.E_end is not None) and (t == model.setHorizon.last()):
                lhs = model.varDeviceStorageEnergy[dev, t]
                # rhs = model.varDeviceStorageEnergy[dev,0] # end=start
                rhs = dev_data.E_end
                return lhs == rhs
            else:
                return pyo.Constraint.Skip

    def defineConstraints(self, pyomo_model):
        """Specifies the list of constraints for the device"""

        list_to_reconstruct = super().defineConstraints(pyomo_model)

        constr = pyo.Constraint(
            pyomo_model.setHorizon, pyo.RangeSet(1, 9), rule=self._rules
        )
        # add constraints to model:
        setattr(pyomo_model, "constr_{}_{}".format(self.id, "misc"), constr)
        return list_to_reconstruct

    def getFlowVar(self, pyomo_model, t):
        return pyomo_model.varDeviceFlow[self.id, "el", "out", t]

    def getMaxFlow(self, pyomo_model, t):
        # available power may be limited by energy in the storage
        # charging also contributes (can be reversed)
        # (it can go to e.g. -2 MW to +2MW => 4 MW,
        # even if Pmax=2 MW)
        maxValue = (
            pyomo_model.varDeviceStoragePmax[self.id, t]
            + pyomo_model.varDeviceFlow[self.id, "el", "in", t]
        )
        return maxValue

    def compute_costForDepletedStorage(self, pyomo_model, timesteps):
        stor_cost = 0
        dev_data = self.dev_data
        if dev_data.E_cost is not None:
            stor_cost = dev_data.E_cost
        Emax = dev_data.E_max
        storCost = 0
        for t in timesteps:
            varE = pyomo_model.varDeviceStorageEnergy[self.id, t]
            storCost += stor_cost * (Emax - varE)
        return storCost


class Storage_hydrogen(_StorageDevice):
    "Hydrogen storage"
    carrier_in = ["hydrogen"]
    carrier_out = ["hydrogen"]
    serial = []

    def _rules(self, model, t, i):
        dev = self.id
        dev_data: DeviceStorage_hydrogenData = self.dev_data
        # param_hydrogen = self.optimiser.all_carriers["hydrogen"]
        time_delta_minutes = model.paramTimestepDeltaMinutes
        if i == 1:
            # energy balance (delta E = in - out) (energy in Sm3)
            delta_t = time_delta_minutes * 60  # seconds
            eta = 1
            if dev_data.eta is not None:
                eta = dev_data.eta
            lhs = (
                model.varDeviceFlow[dev, "hydrogen", "in", t] * eta
                - model.varDeviceFlow[dev, "hydrogen", "out", t] / eta
            ) * delta_t
            if t > 0:
                Eprev = model.varDeviceStorageEnergy[dev, t - 1]
            else:
                Eprev = model.paramDeviceEnergyInitially[dev]
            rhs = model.varDeviceStorageEnergy[dev, t] - Eprev
            return lhs == rhs
        elif i == 2:
            # energy storage limit
            ub = dev_data.E_max
            lb = dev_data.E_min
            return pyo.inequality(lb, model.varDeviceStorageEnergy[dev, t], ub)
        elif i == 3:
            # Constraint 3 and 4: to represent absolute value in obj.function
            # see e.g. http://lpsolve.sourceforge.net/5.1/absolute.htm
            #
            # deviation from target and absolute value at the end of horizon
            # TODO: Harald: is there any reason to penalise _positive_ deviation from the target?
            # Xprime>(E_end-E_target)
            # should we instead use
            # Xprime >= 0 (we still need the lower limit (or bound) to avoid negative cost)
            if t != model.setHorizon.last():
                return pyo.Constraint.Skip
            Xprime = model.varDeviceStorageDeviationFromTarget[dev]
            # profile = model.paramDevice[dev]['target_profile']
            target_value = model.paramDeviceEnergyTarget[dev]
            deviation = model.varDeviceStorageEnergy[dev, t] - target_value
            return Xprime >= deviation
        elif i == 4:
            # deviation from target and absolute value at the end of horizon
            if t != model.setHorizon.last():
                return pyo.Constraint.Skip
            Xprime = model.varDeviceStorageDeviationFromTarget[dev]
            # profile = model.paramDevice[dev]['target_profile']
            target_value = model.paramDeviceEnergyTarget[dev]
            deviation = model.varDeviceStorageEnergy[dev, t] - target_value
            return Xprime >= -deviation

    def defineConstraints(self, pyomo_model):
        """Specifies the list of constraints for the device"""

        list_to_reconstruct = super().defineConstraints(pyomo_model)

        constr = pyo.Constraint(
            pyomo_model.setHorizon, pyo.RangeSet(1, 4), rule=self._rules
        )
        # add constraints to model:
        setattr(pyomo_model, "constr_{}_{}".format(self.id, "misc"), constr)
        return list_to_reconstruct

    def getFlowVar(self, pyomo_model, t):
        return pyomo_model.varDeviceFlow[self.id, "hydrogen", "out", t]

    def compute_costForDepletedStorage(self, pyomo_model, timesteps):
        return self.compute_costForDepletedStorage_alt2(pyomo_model, timesteps)

    def compute_costForDepletedStorage_alt1(self, pyomo_model, timesteps):
        # cost if storage level at end of optimisation deviates from
        # target profile (user input based on expectations)
        # absolute value of deviation (filling too much also a cost)
        # - avoids over-filling storage
        dev = self.id
        dev_data = self.dev_data
        deviation = pyomo_model.varDeviceStorageDeviationFromTarget[dev]
        stor_cost = dev_data.E_cost * deviation
        return stor_cost

    def compute_costForDepletedStorage_alt2(self, pyomo_model, timesteps):
        # cost rate kr/s
        # Cost associated with deviation from target value
        # below target = cost, above target = benefit   => gives signal to fill storage
        dev_data = self.dev_data
        E_target = dev_data.E_max
        E_target = pyomo_model.paramDeviceEnergyTarget[self.id]
        t_end = timesteps.last()
        varE = pyomo_model.varDeviceStorageEnergy[self.id, t_end]
        stor_cost = dev_data.E_cost * (E_target - varE)
        # from cost (kr) to cost rate (kr/s):
        stor_cost = stor_cost / len(timesteps)
        return stor_cost
