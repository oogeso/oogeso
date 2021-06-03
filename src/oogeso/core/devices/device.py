import pyomo.environ as pyo
import logging


class Device:
    "Parent class from which all device types derive"

    # Common class parameters:
    carrier_in = []
    carrier_out = []
    serial = []

    def __init__(self, pyomo_model, optimiser, dev_data):
        """Device object constructor"""
        self.dev_data = dev_data
        self.id = dev_data.id
        self.pyomo_model = pyomo_model
        self.optimiser = optimiser
        self.dev_constraints = None

    def setInitValues(self):
        # Method invoked to specify optimisation problem initial value parameters
        # TODO: move this to each subclass instead
        dev_id = self.id
        pyomo_model = self.pyomo_model
        dev_data = self.dev_data
        if hasattr(dev_data, "E_init"):
            pyomo_model.paramDeviceEnergyInitially[dev_id] = dev_data.E_init
        if hasattr(dev_data, "isOn_init"):
            pyomo_model.paramDeviceIsOnInitially[dev_id] = dev_data.isOn_init
        if hasattr(dev_data, "P_init"):
            pyomo_model.paramDevicePowerInitially[dev_id] = dev_data.P_init

    def _rule_devicePmax(self, model, t):
        maxValue = self.dev_data.flow_max
        if self.dev_data.profile is not None:
            # use an availability profile if provided
            extprofile = self.dev_data.profile
            maxValue = maxValue * model.paramProfiles[extprofile, t]
        power = self.getFlowVar(t)
        if power is not None:
            expr = power <= maxValue
        else:
            expr = pyo.Constraint.Skip
        return expr

    def _rule_devicePmin(self, model, t):
        minValue = self.dev_data.flow_min
        if self.dev_data.profile is not None:
            # use an availability profile if provided
            extprofile = self.dev_data.profile
            minValue = minValue * model.paramProfiles[extprofile, t]
        power = self.getFlowVar(t)
        if power is not None:
            expr = power >= minValue
        else:
            expr = pyo.Constraint.Skip
        return expr

    def _rule_ramprate(self, model, t):
        """power ramp rate limit"""
        dev = self.id
        dev_data = self.dev_data
        param_generic = self.optimiser.optimisation_parameters

        # If no ramp limits have been specified, skip constraint
        if self.dev_data.max_ramp_up is None:
            return pyo.Constraint.Skip
        if t > 0:
            p_prev = self.getFlowVar(t - 1)
        else:
            p_prev = model.paramDevicePowerInitially[dev]
        p_this = self.getFlowVar(t)
        deltaP = p_this - p_prev
        delta_t = param_generic.time_delta_minutes
        maxP = dev_data.flow_max
        max_neg = -dev_data.max_ramp_down * maxP * delta_t
        max_pos = dev_data.max_ramp_up * maxP * delta_t
        expr = pyo.inequality(max_neg, deltaP, max_pos)
        return expr

    def defineConstraints(self):
        """Returns a set of constraints for the device."""

        # Generic constraints common for all device types:

        if self.dev_data.flow_max is not None:
            constrDevicePmax = pyo.Constraint(
                self.pyomo_model.setHorizon, rule=self._rule_devicePmax
            )
            setattr(
                self.pyomo_model,
                "constr_{}_{}".format(self.id, "Pmax"),
                constrDevicePmax,
            )
        if self.dev_data.flow_min is not None:
            constrDevicePmin = pyo.Constraint(
                self.pyomo_model.setHorizon, rule=self._rule_devicePmin
            )
            setattr(
                self.pyomo_model,
                "constr_{}_{}".format(self.id, "Pmin"),
                constrDevicePmin,
            )
        # if "Qmax" in self.params:
        #     constrDeviceQmax = pyo.Constraint(
        #         self.pyomo_model.setHorizon, rule=self._rule_deviceQmax
        #     )
        #     setattr(
        #         self.pyomo_model,
        #         "constr_{}_{}".format(self.id, "Qmax"),
        #         constrDeviceQmax,
        #     )
        # if "Qmin" in self.params:
        #     constrDeviceQmin = pyo.Constraint(
        #         self.pyomo_model.setHorizon, rule=self._rule_deviceQmin
        #     )
        #     setattr(
        #         self.pyomo_model,
        #         "constr_{}_{}".format(self.id, "Qmin"),
        #         constrDeviceQmin,
        #     )

        if (self.dev_data.max_ramp_up is not None) or (
            self.dev_data.max_ramp_down is not None
        ):
            constrDevice_ramprate = pyo.Constraint(
                self.pyomo_model.setHorizon, rule=self._rule_ramprate
            )
            setattr(
                self.pyomo_model,
                "constr_{}_{}".format(self.id, "ramprate"),
                constrDevice_ramprate,
            )

    def getFlowVar(self, t):
        logging.error("Device: no getFlowVar defined for {}".format(self.id))
        raise NotImplementedError()

    def getMaxP(self, t):
        model = self.pyomo_model
        maxValue = self.dev_data.flow_max
        if self.dev_data.profile is not None:
            extprofile = self.dev_data.profile
            maxValue = maxValue * model.paramProfiles[extprofile, t]
        return maxValue

    def compute_CO2(self, timesteps):
        return 0

    def compute_export(self, value, carriers, timesteps):
        """Compute average export (volume or revenue)

        Parameters:
        -----------
        value : str
            "revenue" (€/s) or "volume" (Sm3oe/s)
        carriers : list of carriers ("gas","oil","el")
        timesteps : list of timesteps
        """
        carriers_in = self.carrier_in
        carriers_incl = [v for v in carriers if v in carriers_in]
        sumValue = 0
        if not hasattr(self.dev_data, "price"):
            return 0
        for carrier in carriers_incl:
            # flow in m3/s, price in $/m3
            if self.dev_data.price is not None:
                if carrier in self.dev_data.price:
                    inflow = sum(
                        self.pyomo_model.varDeviceFlow[self.id, carrier, "in", t]
                        for t in timesteps
                    )
                    if value == "revenue":
                        sumValue += inflow * self.dev_data.price[carrier]
                    elif value == "volume":
                        volumefactor = 1
                        if carrier == "gas":
                            volumefactor = 1 / 1000  # Sm3 to Sm3oe
                        sumValue += inflow * volumefactor
        return sumValue

    def compute_elReserve(self, t):
        """Compute available reserve power from this device

        device parameter "reserve_factor" specifies how large part of the
        available capacity should count towards the reserve (1=all, 0=none)
        """
        model = self.pyomo_model
        rf = 1
        loadreduction = 0
        cap_avail = 0
        p_generating = 0
        if "el" in self.carrier_out:
            # Generators and storage
            maxValue = self.getMaxP(t)
            if self.dev_data.reserve_factor is not None:
                # safety margin - only count a part of the forecast power
                # towards the reserve, relevant for wind power
                # (equivalently, this may be seen as increaseing the
                # reserve margin requirement)
                reserve_factor = self.dev_data.reserve_factor
                maxValue = maxValue * reserve_factor
                if reserve_factor == 0:
                    # no reserve contribution
                    rf = 0
            cap_avail = rf * maxValue
            p_generating = rf * model.varDeviceFlow[self.id, "el", "out", t]
        elif "el" in self.carrier_in:
            # Loads (only consider if resere factor has been set)
            if self.dev_data.reserve_factor is not None:
                # load reduction possible
                f_lr = self.dev_data.reserve_factor
                loadreduction = f_lr * model.varDeviceFlow[self.id, "el", "in", t]
        reserve = {
            "capacity_available": cap_avail,
            "capacity_used": p_generating,
            "loadreduction_available": loadreduction,
        }
        return reserve

    # only gas turbine has non-zero start/stop costs
    def compute_startup_costs(self, timesteps):
        return 0

    def compute_operatingCosts(self, timesteps):
        """average operating cost within selected timespan"""
        sumCost = 0
        if self.dev_data.op_cost is not None:
            opcost = self.dev_data.op_cost
            for t in self.pyomo_model.setHorizon:
                varP = self.getFlowVar(t)
                sumCost += opcost * varP
        # average per sec (simulation timestep drops out)
        avgCost = sumCost / len(timesteps)
        return avgCost

    def compute_costForDepletedStorage(self, timesteps):
        return 0

    def getProfile(self):
        """Get device profile as list of values, or None if no profile is used"""
        profile = None
        if self.dev_data.profile is not None:
            prof_id = self.dev_data.profile
            profile = self.pyomo_model.paramProfiles[prof_id, :].value
        return profile
