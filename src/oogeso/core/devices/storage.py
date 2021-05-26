import pyomo.environ as pyo
import logging
from . import Device


class _StorageDevice(Device):
    def __init__(self, pyomo_model, dev_id, dev_data, optimiser):
        """Device object constructor"""
        super().__init__(pyomo_model, dev_id, dev_data, optimiser)
        # Add this device to global list of storage devices:
        optimiser.devices_with_storage.append(self)


class Storage_el(_StorageDevice):
    "Electric storage (battery)"
    carrier_in = ["el"]
    carrier_out = ["el"]
    serial = []

    def _rules(self, model, t, i):
        dev = self.dev_id
        generic_params = self.optimiser.optimisation_parameters
        if i == 1:
            # energy balance
            # (el_in*eta - el_out/eta)*dt = delta storage
            # eta = efficiency charging and discharging
            delta_t = generic_params["time_delta_minutes"] / 60  # hours
            lhs = (
                model.varDeviceFlow[dev, "el", "in", t] * self.params["eta"]
                - model.varDeviceFlow[dev, "el", "out", t] / self.params["eta"]
            ) * delta_t
            if t > 0:
                Eprev = model.varDeviceStorageEnergy[dev, t - 1]
            else:
                Eprev = model.paramDeviceEnergyInitially[dev]
            rhs = model.varDeviceStorageEnergy[dev, t] - Eprev
            return lhs == rhs
        elif i == 2:
            # energy storage limit
            ub = self.params["Emax"]
            lb = 0
            # Specifying Emin may be useful e.g. to impose that battery
            # should only be used for reserve
            # BUT - probably better to specify an energy depletion cost
            # instead (Ecost) - to allow using battery (at a cost) if required
            if "Emin" in self.params:
                lb = self.params["Emin"]
            return pyo.inequality(lb, model.varDeviceStorageEnergy[dev, t], ub)
        elif i == 3:
            # return pyo.Constraint.Skip # unnecessary -> generic Pmax/min constraints
            # discharging power limit
            ub = self.params["Pmax"]
            return model.varDeviceFlow[dev, "el", "out", t] <= ub
        elif i == 4:
            # return pyo.Constraint.Skip # unnecessary -> generic Pmax/min constraints
            # charging power limit
            # ub = model.paramDevice[dev]['Pmax']
            if "Pmin" in self.params:
                ub = -self.params["Pmin"]
            else:
                # assume max charging power is the same as discharging power (Pmax)
                ub = self.params["Pmax"]  # <- see generic Pmax/min constr
            return model.varDeviceFlow[dev, "el", "in", t] <= ub
        elif i == 5:
            # Constraint 5-8: varDeviceStoragePmax = min{Pmax,E/dt}
            # ref: https://or.stackexchange.com/a/1174
            lhs = model.varDeviceStoragePmax[dev, t]
            rhs = self.params["Pmax"]
            return lhs <= rhs
        elif i == 6:
            lhs = model.varDeviceStoragePmax[dev, t]
            # Parameter specifying for how long the power needs to be
            # sustained to count as reserve (e.g. similar to GT startup time)
            dt_hours = generic_params["time_reserve_minutes"] / 60
            rhs = model.varDeviceStorageEnergy[dev, t] / dt_hours
            return lhs <= rhs
        elif i == 7:
            bigM = 10 * self.params["Pmax"]
            lhs = model.varDeviceStoragePmax[dev, t]
            rhs = self.params["Pmax"] - bigM * (1 - model.varStorY[dev, t])
            return lhs >= rhs
        elif i == 8:
            dt_hours = generic_params["time_reserve_minutes"] / 60
            bigM = 10 * self.params["Emax"] / dt_hours
            lhs = model.varDeviceStoragePmax[dev, t]
            rhs = (
                model.varDeviceStorageEnergy[dev, t] / dt_hours
                - bigM * model.varStorY[dev, t]
            )
            return lhs >= rhs
        elif i == 9:
            # constraint on storage end vs start
            # Adding this does not seem to improve result (not lower CO2)
            if ("E_end" in self.params) and (t == model.setHorizon[-1]):
                lhs = model.varDeviceStorageEnergy[dev, t]
                # rhs = model.varDeviceStorageEnergy[dev,0] # end=start
                rhs = self.params["E_end"]
                return lhs == rhs
            else:
                return pyo.Constraint.Skip

    def defineConstraints(self):
        """Specifies the list of constraints for the device"""

        super().defineConstraints()

        constr = pyo.Constraint(
            self.pyomo_model.setHorizon, pyo.RangeSet(1, 9), rule=self._rules
        )
        # add constraints to model:
        setattr(self.pyomo_model, "constr_{}_{}".format(self.dev_id, "misc"), constr)

    def getPowerVar(self, t):
        return self.pyomo_model.varDeviceFlow[self.dev_id, "el", "out", t]

    def getMaxP(self, t):
        # available power may be limited by energy in the storage
        # charging also contributes (can be reversed)
        # (it can go to e.g. -2 MW to +2MW => 4 MW,
        # even if Pmax=2 MW)
        model = self.pyomo_model
        maxValue = (
            model.varDeviceStoragePmax[self.dev_id, t]
            + model.varDeviceFlow[self.dev_id, "el", "in", t]
        )
        return maxValue

    def compute_costForDepletedStorage(self, timesteps):
        stor_cost = 0
        if "Ecost" in self.params:
            stor_cost = self.params["Ecost"]
        Emax = self.params["Emax"]
        storCost = 0
        for t in timesteps:
            varE = self.pyomo_model.varDeviceStorageEnergy[self.dev_id, t]
            storCost += stor_cost * (Emax - varE)
        return storCost


class Storage_hydrogen(_StorageDevice):
    "Hydrogen storage"
    carrier_in = ["hydrogen"]
    carrier_out = ["hydrogen"]
    serial = []

    def _rules(self, model, t, i):
        dev = self.dev_id
        param_dev = self.params
        param_hydrogen = self.optimiser.all_carriers["hydrogen"].params
        generic_params = self.optimiser.optimisation_parameters
        if i == 1:
            # energy balance (delta E = in - out) (energy in Sm3)
            delta_t = generic_params["time_delta_minutes"] * 60  # seconds
            eta = 1
            if "eta" in param_dev:
                eta = param_dev["eta"]
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
            ub = param_dev["Emax"]
            lb = 0
            if "Emin" in model.paramDevice[dev]:
                lb = param_dev["Emin"]
            return pyo.inequality(lb, model.varDeviceStorageEnergy[dev, t], ub)
        elif i == 3:
            # Constraint 3 and 4: to represent absolute value in obj.function
            # see e.g. http://lpsolve.sourceforge.net/5.1/absolute.htm
            #
            # deviation from target and absolute value at the end of horizon
            if t != model.setHorizon[-1]:
                return pyo.Constraint.Skip
            Xprime = model.varDeviceStorageDeviationFromTarget[dev]
            # profile = model.paramDevice[dev]['target_profile']
            target_value = model.paramDeviceEnergyTarget[dev]
            deviation = model.varDeviceStorageEnergy[dev, t] - target_value
            return Xprime >= deviation
        elif i == 4:
            # deviation from target and absolute value at the end of horizon
            if t != model.setHorizon[-1]:
                return pyo.Constraint.Skip
            Xprime = model.varDeviceStorageDeviationFromTarget[dev]
            # profile = model.paramDevice[dev]['target_profile']
            target_value = model.paramDeviceEnergyTarget[dev]
            deviation = model.varDeviceStorageEnergy[dev, t] - target_value
            return Xprime >= -deviation

    def defineConstraints(self):
        """Specifies the list of constraints for the device"""

        super().defineConstraints()

        constr = pyo.Constraint(
            self.pyomo_model.setHorizon, pyo.RangeSet(1, 4), rule=self._rules
        )
        # add constraints to model:
        setattr(self.pyomo_model, "constr_{}_{}".format(self.dev_id, "misc"), constr)

    def getFlowVar(self, t):
        return self.pyomo_model.varDeviceFlow[self.dev_id, "hydrogen", "out", t]

    def compute_costForDepletedStorage(self, timesteps):
        # cost if storage level at end of optimisation deviates from
        # target profile (user input based on expectations)
        d = self.dev_id
        deviation = self.pyomo_model.varDeviceStorageDeviationFromTarget[d]
        stor_cost = self.params["Ecost"] * deviation
        return stor_cost
