import pyomo.environ as pyo
import logging
from . import Device


class Gasturbine(Device):
    "Gas turbine generator"

    carrier_in = ["gas"]
    carrier_out = ["el", "heat"]
    serial = []

    # overriding generic Pmax constraint because of start/stop logic
    # for gas turbine generators:
    def _rule_devicePmax(self, model, t):
        # max/min power (zero if device is not on)
        maxValue = self.params["Pmax"]
        if "profile" in self.params:
            # use an availability profile if provided
            extprofile = self.params["profile"]
            maxValue = maxValue * model.paramProfiles[extprofile, t]
        ison = model.varDeviceIsOn[self.dev_id, t]
        power = self.getPowerVar(t)
        expr = power <= ison * maxValue
        return expr

    def _rule_devicePmin(self, model, t):
        params_dev = self.params
        minValue = params_dev["Pmin"]
        if "profile" in params_dev:
            # use an availability profile if provided
            extprofile = params_dev["profile"]
            minValue = minValue * model.paramProfiles[extprofile, t]
        ison = 1
        ison = model.varDeviceIsOn[self.dev_id, t]
        power = self.getPowerVar(t)
        expr = power >= ison * minValue
        return expr

    def _rule_startup_shutdown(self, model, t):
        """startup/shutdown constraint
        connecting starting, stopping, preparation, online stages of GTs"""
        dev = self.dev_id
        param_dev = self.params
        param_generic = self.optimiser.optimisation_parameters
        Tdelay = 0
        if "startupDelay" in param_dev:
            Tdelay_min = param_dev["startupDelay"]
            # Delay in timesteps, rounding down.
            # example: time_delta = 5 min, startupDelay= 8 min => Tdelay=1
            Tdelay = int(Tdelay_min / param_generic["time_delta_minutes"])
        prevPart = 0
        if t >= Tdelay:
            # prevPart = sum( model.varDeviceStarting[dev,t-tau]
            #    for tau in range(0,Tdelay) )
            prevPart = model.varDeviceStarting[dev, t - Tdelay]
        else:
            # OBS: for this to work as intended, need to reconstruct constraint
            # pyo.value(...) not needed
            # prepInit = pyo.value(model.paramDevicePrepTimestepsInitially[dev])
            prepInit = model.paramDevicePrepTimestepsInitially[dev]
            if prepInit + t == Tdelay:
                prevPart = 1
        if t > 0:
            ison_prev = model.varDeviceIsOn[dev, t - 1]
        else:
            ison_prev = model.paramDeviceIsOnInitially[dev]
        lhs = model.varDeviceIsOn[dev, t] - ison_prev
        rhs = prevPart - model.varDeviceStopping[dev, t]
        return lhs == rhs

    def _rule_startup_delay(self, model, t):
        """startup delay/preparation for GTs"""
        dev = self.dev_id
        param_dev = self.params
        param_generic = self.optimiser.optimisation_parameters
        Tdelay = 0
        if "startupDelay" in param_dev:
            Tdelay_min = param_dev["startupDelay"]
            # Delay in timesteps, rounding down.
            # example: time_delta = 5 min, startupDelay= 8 min => Tdelay=1
            Tdelay = int(Tdelay_min / param_generic["time_delta_minutes"])
        else:
            return pyo.Constraint.Skip
        # determine if was in preparation previously
        # dependend on value - so must reconstruct constraint each time
        stepsPrevPrep = pyo.value(model.paramDevicePrepTimestepsInitially[dev])
        if stepsPrevPrep > 0:
            prevIsPrep = 1
        else:
            prevIsPrep = 0

        prevPart = 0
        if t < Tdelay - stepsPrevPrep:
            prevPart = prevIsPrep
        tau_range = range(0, min(t + 1, Tdelay))
        lhs = model.varDeviceIsPrep[dev, t]
        rhs = sum(model.varDeviceStarting[dev, t - tau] for tau in tau_range) + prevPart
        return lhs == rhs

    def _rules_misc(self, model, t, i):
        dev = self.dev_id
        param_gas = self.optimiser.all_carriers["gas"].params
        elpower = model.varDeviceFlow[dev, "el", "out", t]
        gas_energy_content = param_gas["energy_value"]  # MJ/Sm3
        if i == 1:
            """turbine el power out vs gas fuel in"""
            # fuel consumption (gas in) is a linear function of el power output
            # fuel = B + A*power
            # => efficiency = power/(A+B*power)
            A = self.params["fuelA"]
            B = self.params["fuelB"]
            Pmax = self.params["Pmax"]
            lhs = model.varDeviceFlow[dev, "gas", "in", t] * gas_energy_content / Pmax
            rhs = (
                B * (model.varDeviceIsOn[dev, t] + model.varDeviceIsPrep[dev, t])
                + A * model.varDeviceFlow[dev, "el", "out", t] / Pmax
            )
            return lhs == rhs
        elif i == 2:
            """heat output = (gas energy in - el power out)* heat efficiency"""
            lhs = model.varDeviceFlow[dev, "heat", "out", t]
            rhs = (
                model.varDeviceFlow[dev, "gas", "in", t] * gas_energy_content
                - model.varDeviceFlow[dev, "el", "out", t]
            ) * self.params["eta_heat"]
            return lhs == rhs

    def defineConstraints(self):
        """Specifies the list of constraints for the device"""
        super().defineConstraints()

        constr = pyo.Constraint(
            self.pyomo_model.setHorizon, pyo.RangeSet(1, 2), rule=self._rules_misc
        )
        setattr(self.pyomo_model, "constr_{}_{}".format(self.dev_id, "misc"), constr)

        constrDevice_startup_shutdown = pyo.Constraint(
            self.pyomo_model.setHorizon, rule=self._rule_startup_shutdown
        )
        setattr(
            self.pyomo_model,
            "constr_{}_{}".format(self.dev_id, "startstop"),
            constrDevice_startup_shutdown,
        )

        constrDevice_startup_delay = pyo.Constraint(
            self.pyomo_model.setHorizon, rule=self._rule_startup_delay
        )
        setattr(
            self.pyomo_model,
            "constr_{}_{}".format(self.dev_id, "startdelay"),
            constrDevice_startup_delay,
        )

        # Because of logic that needs to be re-evalued, these constraints need
        # to be reconstructed each optimisation:
        self.optimiser.constraints_to_reconstruct.append(constrDevice_startup_shutdown)
        self.optimiser.constraints_to_reconstruct.append(constrDevice_startup_delay)

    def getPowerVar(self, t):
        return self.pyomo_model.varDeviceFlow[self.dev_id, "el", "out", t]

    def getMaxP(self, t):
        """get available capacity"""
        maxValue = super().getMaxP(t)
        # maxValue = Device.getMaxP(self,model,t) # Works
        ison = self.pyomo_model.varDeviceIsOn[self.dev_id, t]
        maxValue = ison * maxValue
        return maxValue

    # overriding default
    def compute_CO2(self, timesteps):
        model = self.pyomo_model
        param_gas = self.optimiser.all_carriers["gas"].params
        gasflow_co2 = param_gas["CO2content"]  # kg/m3
        thisCO2 = (
            sum(model.varDeviceFlow[self.dev_id, "gas", "in", t] for t in timesteps)
            * gasflow_co2
        )
        return thisCO2

    def compute_startup_costs(self, timesteps):
        start_stop_costs = 0
        model = self.pyomo_model
        if "startupCost" in self.params:
            startupcost = self.params["startupCost"]
            thisCost = (
                sum(model.varDeviceStarting[self.dev_id, t] for t in timesteps)
                * startupcost
            )
            start_stop_costs += thisCost
        if "shutdownCost" in self.params:
            shutdowncost = self.params["shutdownCost"]
            thisCost = (
                sum(model.varDeviceStopping[self.dev_id, t] for t in timesteps)
                * shutdowncost
            )
            start_stop_costs += thisCost
        return start_stop_costs
