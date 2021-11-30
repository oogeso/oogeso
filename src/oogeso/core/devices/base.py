import logging
from typing import Dict, List, Optional

import pyomo.environ as pyo
from pyomo.core import Constraint

from oogeso.core.networks.network_node import NetworkNode
from oogeso.dto import DeviceData

logger = logging.getLogger(__name__)


class Device:
    """
    Parent class from which all device types derive
    """

    # Common class parameters:
    carrier_in = list()
    carrier_out = list()
    serial = list()

    def __init__(self, dev_data: DeviceData, carrier_data_dict: Dict):
        """Device object constructor"""
        self.dev_data = dev_data
        self.id = dev_data.id
        self.carrier_data = carrier_data_dict
        self.node: Optional[NetworkNode] = None
        self._flow_upper_bound: Optional[float] = None

    def add_node(self, node: NetworkNode):
        """associate node with device"""
        self.node = node

    def set_init_values(self, pyomo_model: pyo.Model):
        """Method invoked to specify optimisation problem initial value parameters"""
        # TODO: move this to each subclass instead?
        dev_id = self.id
        dev_data = self.dev_data
        if hasattr(dev_data, "E_init"):
            pyomo_model.paramDeviceEnergyInitially[dev_id] = dev_data.E_init
        if dev_data.start_stop is not None:
            pyomo_model.paramDeviceIsOnInitially[dev_id] = dev_data.start_stop.is_on_init
        if hasattr(dev_data, "P_init"):
            pyomo_model.paramDevicePowerInitially[dev_id] = dev_data.P_init

    def _rule_device_flow_max(self, pyomo_model: pyo.Model, t):
        power = self.get_flow_var(pyomo_model, t)
        if power is None:
            return pyo.Constraint.Skip
        maxValue = self.get_max_flow(pyomo_model, t)
        expr = power <= maxValue
        return expr

    def _rule_device_flow_min(self, pyomo_model: pyo.Model, t):
        power = self.get_flow_var(pyomo_model, t)
        if power is None:
            return pyo.Constraint.Skip
        minValue = self.dev_data.flow_min
        if self.dev_data.profile is not None:
            # use an availability profile if provided
            extprofile = self.dev_data.profile
            minValue = minValue * pyomo_model.paramProfiles[extprofile, t]
        ison = 1
        if self.dev_data.start_stop is not None:
            ison = pyomo_model.varDeviceIsOn[self.id, t]
        expr = power >= ison * minValue
        return expr

    def _rule_ramp_rate(self, pyomo_model, t):
        """power ramp rate limit"""
        dev = self.id
        dev_data = self.dev_data

        # If no ramp limits have been specified, skip constraint
        if self.dev_data.max_ramp_up is None:
            return pyo.Constraint.Skip
        if t > 0:
            p_prev = self.get_flow_var(pyomo_model, t - 1)
        else:
            p_prev = pyomo_model.paramDevicePowerInitially[dev]
        p_this = self.get_flow_var(pyomo_model, t)
        deltaP = p_this - p_prev
        delta_t = pyomo_model.paramTimestepDeltaMinutes
        maxP = dev_data.flow_max
        max_neg = -dev_data.max_ramp_down * maxP * delta_t
        max_pos = dev_data.max_ramp_up * maxP * delta_t
        expr = pyo.inequality(max_neg, deltaP, max_pos)
        return expr

    def _rule_startup_shutdown(self, model, t):
        """startup/shutdown constraint
        connecting starting, stopping, preparation, online stages of GTs"""
        dev = self.id
        Tdelay_min = self.dev_data.start_stop.delay_start_minutes
        time_delta_minutes = model.paramTimestepDeltaMinutes
        # Delay in timesteps, rounding down.
        # example: time_delta = 5 min, delay_start_minutes= 8 min => Tdelay=1
        Tdelay = int(Tdelay_min / time_delta_minutes)
        prevPart = 0
        if t >= Tdelay:
            # prevPart = sum( model.varDeviceStarting[dev,t-tau]
            #    for tau in range(0,Tdelay) )
            prevPart = model.varDeviceStarting[dev, t - Tdelay]
        else:
            # NOTE: for this to work as intended, may need to reconstruct constraint
            # pyo.value(...) needed in pyomo v6
            prepInit = pyo.value(model.paramDevicePrepTimestepsInitially[dev])
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
        time_delta_minutes = model.paramTimestepDeltaMinutes
        Tdelay_min = self.dev_data.start_stop.delay_start_minutes
        # Delay in timesteps, rounding down.
        # example: time_delta = 5 min, startupDelay= 8 min => Tdelay=1
        Tdelay = int(Tdelay_min / time_delta_minutes)
        if Tdelay == 0:
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

    def define_constraints(self, pyomo_model) -> List[Constraint]:
        """Build constraints for the device and add to pyomo model.
        Returns list of constraints that need to be reconstructed between each
        optimisation
        """

        list_to_reconstruct = []  # Default

        if self.dev_data.flow_max is not None:
            constrDevicePmax = pyo.Constraint(pyomo_model.setHorizon, rule=self._rule_device_flow_max)
            setattr(
                pyomo_model,
                "constr_{}_{}".format(self.id, "flowMax"),
                constrDevicePmax,
            )
        if self.dev_data.flow_min is not None:
            constrDevicePmin = pyo.Constraint(pyomo_model.setHorizon, rule=self._rule_device_flow_min)
            setattr(
                pyomo_model,
                "constr_{}_{}".format(self.id, "flowMin"),
                constrDevicePmin,
            )
        if (self.dev_data.max_ramp_up is not None) or (self.dev_data.max_ramp_down is not None):
            constrDevice_ramprate = pyo.Constraint(pyomo_model.setHorizon, rule=self._rule_ramp_rate)
            setattr(
                pyomo_model,
                "constr_{}_{}".format(self.id, "ramprate"),
                constrDevice_ramprate,
            )
        if self.dev_data.start_stop is not None:
            constrDevice_startup_shutdown = pyo.Constraint(pyomo_model.setHorizon, rule=self._rule_startup_shutdown)
            setattr(
                pyomo_model,
                "constr_{}_{}".format(self.id, "startstop"),
                constrDevice_startup_shutdown,
            )
            constrDevice_startup_delay = pyo.Constraint(pyomo_model.setHorizon, rule=self._rule_startup_delay)
            setattr(
                pyomo_model,
                "constr_{}_{}".format(self.id, "startdelay"),
                constrDevice_startup_delay,
            )

            # TODO: Add constraints for minimum up and down-time

            # return list of constraints that need to be reconstructed:
            list_to_reconstruct = [
                constrDevice_startup_shutdown,
                constrDevice_startup_delay,
            ]
        return list_to_reconstruct

    def get_flow_var(self, pyomo_model, t) -> float:
        logger.error("Device: no get_flow_var defined for {}".format(self.id))
        raise NotImplementedError()

    def get_max_flow(self, pyomo_model, t):
        """Return available capacity at given timestep.
        This is given by the "flow_max" input parameter, profile value (if any), and
        whether devie is on/off."""
        maxValue = self.dev_data.flow_max
        if self.dev_data.profile is not None:
            extprofile = self.dev_data.profile
            maxValue = maxValue * pyomo_model.paramProfiles[extprofile, t]
        if self.dev_data.start_stop is not None:
            ison = pyomo_model.varDeviceIsOn[self.id, t]
            maxValue = ison * maxValue
        return maxValue

    def set_flow_upper_bound(self, profiles):
        """Maximum flow value through entire profile.
        Given as the product of the "flow_max" parameter and profile values."""
        ub = self.dev_data.flow_max
        if self.dev_data.profile is not None:
            extprofile = self.dev_data.profile
            prof_max = None
            for prof in profiles:
                if prof.id == extprofile:
                    prof_max = max(prof.data)
                    if prof.data_nowcast is not None:
                        prof_nowcast_max = max(prof.data_nowcast)
                        prof_max = max(prof_max, prof_nowcast_max)
                    ub = ub * prof_max
                    break
            if prof_max is None:
                logger.warning(
                    "Profile (%s) defined for device %s was not found",
                    extprofile,
                    self.dev_data.id,
                )
        self._flow_upper_bound = ub

    def get_flow_upper_bound(self):
        """Returns the maximum possible flow given capacity and profile"""
        # Used by piecewise linear constraints
        return self._flow_upper_bound

    def compute_CO2(self, pyomo_model, timesteps):
        return 0

    def compute_export(self, pyomo_model, value, carriers, timesteps):
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
                    inflow = sum(pyomo_model.varDeviceFlow[self.id, carrier, "in", t] for t in timesteps)
                    if value == "revenue":
                        sumValue += inflow * self.dev_data.price[carrier]
                    elif value == "volume":
                        volumefactor = 1
                        if carrier == "gas":
                            volumefactor = 1 / 1000  # Sm3 to Sm3oe
                        sumValue += inflow * volumefactor
        return sumValue

    def compute_el_reserve(self, model, t):
        """Compute available reserve power from this device

        device parameter "reserve_factor" specifies how large part of the
        available capacity should count towards the reserve (1=all, 0=none)
        """
        rf = 1
        loadreduction = 0
        cap_avail = 0
        p_generating = 0
        if "el" in self.carrier_out:
            # Generators and storage
            maxValue = self.get_max_flow(model, t)
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

    # start/stop penalty - generalised from gas turbine startup cost
    def compute_startup_penalty(self, pyomo_model, timesteps):
        penalty = 0
        model = pyomo_model
        if self.dev_data.start_stop is not None:
            penalty = (
                sum(model.varDeviceStarting[self.id, t] for t in timesteps) * self.dev_data.start_stop.penalty_start
                + sum(model.varDeviceStopping[self.id, t] for t in timesteps) * self.dev_data.start_stop.penalty_stop
            )
        return penalty

    def compute_operating_costs(self, pyomo_model, timesteps):
        """average operating cost within selected timespan"""
        sumCost = 0
        if self.dev_data.op_cost is not None:
            opcost = self.dev_data.op_cost
            for t in pyomo_model.setHorizon:
                varP = self.get_flow_var(pyomo_model, t)
                sumCost += opcost * varP
        # average per sec (simulation timestep drops out)
        avgCost = sumCost / len(timesteps)
        return avgCost

    def compute_cost_for_depleted_storage(self, pyomo_model, timesteps):
        return 0

    def compute_penalty(self, pyomo_model, timesteps: List[int]):
        """Compute average penalty rate (cost, emission or similar per second)
        as defined by penalty functions and start/stop penalties"""
        penalty_rate = 0
        if hasattr(self.dev_data, "penalty_function") and self.dev_data.penalty_function is not None:
            if not hasattr(self, "_penaltyConstraint"):
                logger.warning("Penalty function constraint not impelemented for %s", self.id)
            # Since the penalty function may be nonzero at Pel=0 we need to split up so computed
            # penalty for Pel > 0 only when device is actually online (penalty should be zero when
            # device is offline)
            penalty_offset = 0
            if self.dev_data.start_stop is not None:
                # penalty_offset = penalty(Pel=0)
                penalty_offset = self.dev_data.penalty_function[1][0]
            this_penalty = sum(
                pyomo_model.varDevicePenalty[self.id, "el", "out", t]
                + (pyomo_model.varDeviceIsOn[self.id, t] - 1) * penalty_offset
                for t in timesteps
            )
            # divide by number of timesteps to get average penalty rate (penalty per sec):
            penalty_rate = this_penalty / len(timesteps)

        start_stop_penalty_rate = 0
        if self.dev_data.start_stop is not None:
            # this sums up penalty over all timesteps in horizon:
            start_stop_penalty = self.compute_startup_penalty(pyomo_model, timesteps)
            # get average per second:
            time_delta_minutes = pyomo_model.paramTimestepDeltaMinutes
            timestep_duration_sec = time_delta_minutes * 60
            time_interval_sec = len(timesteps) * timestep_duration_sec
            start_stop_penalty_rate = start_stop_penalty / time_interval_sec

        sum_penalty_rate = penalty_rate + start_stop_penalty_rate
        return sum_penalty_rate
