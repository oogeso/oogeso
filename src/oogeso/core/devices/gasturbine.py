import pyomo.environ as pyo
import logging
from . import Device


class Gasturbine(Device):
    "Gas turbine generator"

    carrier_in = ["gas"]
    carrier_out = ["el", "heat"]
    serial = []

    # def __init__(self, data):
    #    super().__init__()

    # def setInitValues(self):
    #     """Initial value in pyomo optimisation model"""
    #     dev_id = self.dev_id
    #     pyomo_model = self.pyomo_model
    #     if self.isOn_init is not None:
    #         pyomo_model.paramDeviceIsOnInitially[dev_id] = self.isOn_init
    #     if "P_init" in dev_data:
    #         pyomo_model.paramDevicePowerInitially[dev_id] = dev_data["P_init"]

    # overriding generic Pmax constraint because of start/stop logic
    # for gas turbine generators:
    def _rule_devicePmax(self, model, t):
        # max/min power (zero if device is not on)
        maxValue = self.dev_data.flow_max
        if self.dev_data.profile is not None:
            # use an availability profile if provided
            extprofile = self.dev_data.profile
            maxValue = maxValue * model.paramProfiles[extprofile, t]
        ison = model.varDeviceIsOn[self.id, t]
        power = self.getFlowVar(t)
        expr = power <= ison * maxValue
        return expr

    def _rule_devicePmin(self, model, t):
        minValue = self.dev_data.flow_min
        if self.dev_data.profile is not None:
            # use an availability profile if provided
            extprofile = self.dev_data.profile
            minValue = minValue * model.paramProfiles[extprofile, t]
        ison = 1
        ison = model.varDeviceIsOn[self.id, t]
        power = self.getFlowVar(t)
        expr = power >= ison * minValue
        return expr

    def _rule_startup_shutdown(self, model, t):
        """startup/shutdown constraint
        connecting starting, stopping, preparation, online stages of GTs"""
        dev = self.id
        param_generic = self.optimiser.optimisation_parameters
        Tdelay = 0
        if self.dev_data.startup_delay is not None:
            Tdelay_min = self.dev_data.startup_delay
            # Delay in timesteps, rounding down.
            # example: time_delta = 5 min, startupDelay= 8 min => Tdelay=1
            Tdelay = int(Tdelay_min / param_generic.time_delta_minutes)
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
        dev = self.id
        param_generic = self.optimiser.optimisation_parameters
        Tdelay = 0
        if self.dev_data.startup_delay is not None:
            Tdelay_min = self.dev_data.startup_delay
            # Delay in timesteps, rounding down.
            # example: time_delta = 5 min, startupDelay= 8 min => Tdelay=1
            Tdelay = int(Tdelay_min / param_generic.time_delta_minutes)
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
        dev = self.id
        param_gas = self.optimiser.all_carriers["gas"]
        # elpower = model.varDeviceFlow[dev, "el", "out", t]
        gas_energy_content = param_gas.energy_value  # MJ/Sm3
        if i == 1:
            """turbine el power out vs gas fuel in"""
            # fuel consumption (gas in) is a linear function of el power output
            # fuel = B + A*power
            # => efficiency = power/(A+B*power)
            A = self.dev_data.fuel_A
            B = self.dev_data.fuel_B
            Pmax = self.dev_data.flow_max
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
            ) * self.dev_data.eta_heat
            return lhs == rhs

    def defineConstraints(self):
        """Specifies the list of constraints for the device"""
        super().defineConstraints()

        constr = pyo.Constraint(
            self.pyomo_model.setHorizon, pyo.RangeSet(1, 2), rule=self._rules_misc
        )
        setattr(self.pyomo_model, "constr_{}_{}".format(self.id, "misc"), constr)

        constrDevice_startup_shutdown = pyo.Constraint(
            self.pyomo_model.setHorizon, rule=self._rule_startup_shutdown
        )
        setattr(
            self.pyomo_model,
            "constr_{}_{}".format(self.id, "startstop"),
            constrDevice_startup_shutdown,
        )

        constrDevice_startup_delay = pyo.Constraint(
            self.pyomo_model.setHorizon, rule=self._rule_startup_delay
        )
        setattr(
            self.pyomo_model,
            "constr_{}_{}".format(self.id, "startdelay"),
            constrDevice_startup_delay,
        )

        # Because of logic that needs to be re-evalued, these constraints need
        # to be reconstructed each optimisation:
        self.optimiser.constraints_to_reconstruct.append(constrDevice_startup_shutdown)
        self.optimiser.constraints_to_reconstruct.append(constrDevice_startup_delay)

    def getFlowVar(self, t):
        return self.pyomo_model.varDeviceFlow[self.id, "el", "out", t]

    def getMaxP(self, t):
        """get available capacity"""
        maxValue = super().getMaxP(t)
        # maxValue = Device.getMaxP(self,model,t) # Works
        ison = self.pyomo_model.varDeviceIsOn[self.id, t]
        maxValue = ison * maxValue
        return maxValue

    # overriding default
    def compute_CO2(self, timesteps):
        model = self.pyomo_model
        param_gas = self.optimiser.all_carriers["gas"]
        gasflow_co2 = param_gas.co2_content  # kg/m3
        thisCO2 = (
            sum(model.varDeviceFlow[self.id, "gas", "in", t] for t in timesteps)
            * gasflow_co2
        )
        return thisCO2

    def compute_startup_costs(self, timesteps):
        start_stop_costs = 0
        model = self.pyomo_model
        if self.dev_data.startup_cost is not None:
            startupcost = self.dev_data.startup_cost
            thisCost = (
                sum(model.varDeviceStarting[self.id, t] for t in timesteps)
                * startupcost
            )
            start_stop_costs += thisCost
        if self.dev_data.shutdown_cost is not None:
            shutdowncost = self.dev_data.shutdown_cost
            thisCost = (
                sum(model.varDeviceStopping[self.id, t] for t in timesteps)
                * shutdowncost
            )
            start_stop_costs += thisCost
        return start_stop_costs
