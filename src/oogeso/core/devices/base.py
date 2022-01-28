import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import pyomo.environ as pyo
from pyomo.core import Constraint

from oogeso import dto
from oogeso.core.networks.network_node import NetworkNode

logger = logging.getLogger(__name__)


class Device(ABC):
    """
    Parent class from which all device types derive
    """

    carrier_in: List
    carrier_out: List
    serial: List

    def __init__(self, dev_data: dto.DeviceData, carrier_data_dict: Dict[str, dto.CarrierData]):
        """Device object constructor"""
        self.dev_data = dev_data
        self.id = dev_data.id
        self.carrier_data = carrier_data_dict
        self.node: Optional[NetworkNode] = None
        self._flow_upper_bound: Optional[pyo.Constraint] = None

    def add_node(self, node: NetworkNode) -> None:
        """associate node with device"""
        self.node = node

    def set_init_values(self, pyomo_model: pyo.Model) -> None:
        """Method invoked to specify optimisation problem initial value parameters"""
        dev_id = self.id
        dev_data = self.dev_data
        if hasattr(dev_data, "E_init"):
            pyomo_model.paramDeviceEnergyInitially[dev_id] = dev_data.E_init
        if dev_data.start_stop is not None:
            pyomo_model.paramDeviceIsOnInitially[dev_id] = dev_data.start_stop.is_on_init
        if hasattr(dev_data, "P_init"):
            pyomo_model.paramDevicePowerInitially[dev_id] = dev_data.P_init

    def _rule_device_flow_max(self, pyomo_model: pyo.Model, t: int) -> Union[pyo.Expression, pyo.Constraint.Skip]:
        power = self.get_flow_var(pyomo_model, t)
        if power is None:
            return pyo.Constraint.Skip  # noqa
        max_value = self.get_max_flow(pyomo_model, t)

        return power <= max_value  # noqa

    def _rule_device_flow_min(self, pyomo_model: pyo.Model, t: int) -> Union[pyo.Expression, pyo.Constraint.Skip]:
        power = self.get_flow_var(pyomo_model, t)
        if power is None:
            return pyo.Constraint.Skip  # noqa
        min_value = self.dev_data.flow_min
        if self.dev_data.profile is not None:
            # use an availability profile if provided
            ext_profile = self.dev_data.profile
            min_value = min_value * pyomo_model.paramProfiles[ext_profile, t]
        ison = 1
        if self.dev_data.start_stop is not None:
            ison = pyomo_model.varDeviceIsOn[self.id, t]

        return power >= ison * min_value

    def _rule_ramp_rate(self, pyomo_model: pyo.Model, t: int) -> Union[pyo.Expression, pyo.Constraint.Skip]:
        """power ramp rate limit"""
        dev = self.id
        dev_data = self.dev_data

        # If no ramp limits have been specified, skip constraint
        if self.dev_data.max_ramp_up is None:
            return pyo.Constraint.Skip  # noqa
        if t > 0:
            p_prev = self.get_flow_var(pyomo_model, t - 1)
        else:
            p_prev = pyomo_model.paramDevicePowerInitially[dev]
        p_this = self.get_flow_var(pyomo_model, t)
        delta_P = p_this - p_prev
        delta_t = pyomo_model.paramTimestepDeltaMinutes
        max_P = dev_data.flow_max
        max_neg = -dev_data.max_ramp_down * max_P * delta_t
        max_pos = dev_data.max_ramp_up * max_P * delta_t

        return pyo.inequality(max_neg, delta_P, max_pos)

    def _rule_startup_shutdown(self, model: pyo.Model, t: int) -> Union[pyo.Expression, pyo.Constraint.Skip]:
        """startup/shutdown constraint
        connecting starting, stopping, preparation, online stages of GTs"""
        dev = self.id
        T_delay_min = self.dev_data.start_stop.delay_start_minutes
        time_delta_minutes = model.paramTimestepDeltaMinutes
        # Delay in time-steps, rounding down.
        # example: time_delta = 5 min, delay_start_minutes= 8 min => T_delay=1
        T_delay = int(T_delay_min / time_delta_minutes)
        prev_part = 0
        if t >= T_delay:
            # prev_part = sum( model.varDeviceStarting[dev,t-tau]
            #    for tau in range(0,T_delay) )
            prev_part = model.varDeviceStarting[dev, t - T_delay]
        else:
            # NOTE: for this to work as intended, may need to reconstruct constraint
            # pyo.value(...) needed in pyomo v6
            prep_init = pyo.value(model.paramDevicePrepTimestepsInitially[dev])
            if prep_init + t == T_delay:
                prev_part = 1
        if t > 0:
            is_on_prev = model.varDeviceIsOn[dev, t - 1]
        else:
            is_on_prev = model.paramDeviceIsOnInitially[dev]
        lhs = model.varDeviceIsOn[dev, t] - is_on_prev
        rhs = prev_part - model.varDeviceStopping[dev, t]
        return lhs == rhs

    def _rule_startup_delay(self, pyomo_model: pyo.Model, t: int) -> Union[bool, pyo.Constraint, pyo.Constraint.Skip]:
        """startup delay/preparation for GTs"""
        dev = self.id
        time_delta_minutes = pyomo_model.paramTimestepDeltaMinutes
        T_delay_min = self.dev_data.start_stop.delay_start_minutes
        # Delay in time-steps, rounding down.
        # example: time_delta = 5 min, startupDelay= 8 min => T_delay=1
        T_delay = int(T_delay_min / time_delta_minutes)
        if T_delay == 0:
            return pyo.Constraint.Skip  # noqa
        # determine if was in preparation previously
        # dependent on value - so must reconstruct constraint each time
        steps_prev_prep = pyo.value(pyomo_model.paramDevicePrepTimestepsInitially[dev])
        if steps_prev_prep > 0:
            prev_is_prep = 1
        else:
            prev_is_prep = 0

        prev_part = 0
        if t < T_delay - steps_prev_prep:
            prev_part = prev_is_prep
        tau_range = range(0, min(t + 1, T_delay))
        lhs = pyomo_model.varDeviceIsPrep[dev, t]
        rhs = sum(pyomo_model.varDeviceStarting[dev, t - tau] for tau in tau_range) + prev_part
        return lhs == rhs

    def define_constraints(self, pyomo_model: pyo.Model) -> List[Constraint]:
        """Build constraints for the device and add to pyomo model.
        Returns list of constraints that need to be reconstructed between each
        optimisation

        """

        list_to_reconstruct = []  # Default

        if self.dev_data.flow_max is not None:
            constr_device_P_max = pyo.Constraint(pyomo_model.setHorizon, rule=self._rule_device_flow_max)
            setattr(
                pyomo_model,
                f"constr_{self.id}_flowMax",
                constr_device_P_max,
            )
        if self.dev_data.flow_min is not None:
            constr_device_P_min = pyo.Constraint(pyomo_model.setHorizon, rule=self._rule_device_flow_min)
            setattr(
                pyomo_model,
                f"constr_{self.id}_flowMin",
                constr_device_P_min,
            )
        if (self.dev_data.max_ramp_up is not None) or (self.dev_data.max_ramp_down is not None):
            constr_device_ramprate = pyo.Constraint(pyomo_model.setHorizon, rule=self._rule_ramp_rate)
            setattr(
                pyomo_model,
                f"constr_{self.id}_ramprate",
                constr_device_ramprate,
            )
        if self.dev_data.start_stop is not None:
            constr_device_startup_shutdown = pyo.Constraint(pyomo_model.setHorizon, rule=self._rule_startup_shutdown)
            setattr(
                pyomo_model,
                f"constr_{self.id}_startstop",
                constr_device_startup_shutdown,
            )
            constr_device_startup_delay = pyo.Constraint(pyomo_model.setHorizon, rule=self._rule_startup_delay)
            setattr(
                pyomo_model,
                f"constr_{self.id}_startdelay",
                constr_device_startup_delay,
            )

            # return list of constraints that need to be reconstructed:
            list_to_reconstruct = [
                constr_device_startup_shutdown,
                constr_device_startup_delay,
            ]
        return list_to_reconstruct

    @abstractmethod
    def get_flow_var(self, pyomo_model: pyo.Model, t: int) -> float:
        pass

    def get_max_flow(self, pyomo_model: pyo.Model, t: int) -> float:
        """
        Return available capacity at given time-step.
        This is given by the "flow_max" input parameter, profile value (if any), and
        whether device is on/off.
        """
        max_value = self.dev_data.flow_max
        if self.dev_data.profile is not None:
            ext_profile = self.dev_data.profile
            max_value = max_value * pyomo_model.paramProfiles[ext_profile, t]
        if self.dev_data.start_stop is not None:
            is_on = pyomo_model.varDeviceIsOn[self.id, t]
            max_value = is_on * max_value
        return max_value

    def set_flow_upper_bound(self, profiles: List[dto.TimeSeriesData]) -> None:
        """
        Maximum flow value through entire profile.

        Given as the product of the "flow_max" parameter and profile values.
        """
        ub = self.dev_data.flow_max
        if self.dev_data.profile is not None:
            ext_profile = self.dev_data.profile
            prof_max = None
            for prof in profiles:
                if prof.id == ext_profile:
                    prof_max = max(prof.data)
                    if prof.data_nowcast is not None:
                        prof_nowcast_max = max(prof.data_nowcast)
                        prof_max = max(prof_max, prof_nowcast_max)
                    ub = ub * prof_max
                    break
            if prof_max is None:
                logger.warning(
                    "Profile (%s) defined for device %s was not found",
                    ext_profile,
                    self.dev_data.id,
                )
        self._flow_upper_bound = ub

    def get_flow_upper_bound(self) -> Optional[pyo.Constraint]:
        """Returns the maximum possible flow given capacity and profile"""
        # Used by piecewise linear constraints
        return self._flow_upper_bound

    def compute_export(self, pyomo_model: pyo.Model, value: str, carriers: List[str], timesteps: List[int]) -> float:
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
        sum_value = 0
        if hasattr(self.dev_data, "price"):
            for carrier in carriers_incl:
                # flow in m3/s, price in $/m3
                if self.dev_data.price is not None:
                    if carrier in self.dev_data.price:
                        inflow = sum(pyomo_model.varDeviceFlow[self.id, carrier, "in", t] for t in timesteps)
                        if value == "revenue":
                            sum_value += inflow * self.dev_data.price[carrier]
                        elif value == "volume":
                            volume_factor = 1
                            if carrier == "gas":
                                volume_factor = 1 / 1000  # Sm3 to Sm3oe
                            sum_value += inflow * volume_factor
        return sum_value

    def compute_el_reserve(self, pyomo_model: pyo.Model, t: int) -> Dict[str, float]:
        """Compute available reserve power from this device

        device parameter "reserve_factor" specifies how large part of the
        available capacity should count towards the reserve (1=all, 0=none)
        """
        rf = 1
        load_reduction = 0
        cap_avail = 0
        p_generating = 0
        if "el" in self.carrier_out:
            # Generators and storage
            max_value = self.get_max_flow(pyomo_model=pyomo_model, t=t)
            if self.dev_data.reserve_factor is not None:
                # safety margin - only count a part of the forecast power
                # towards the reserve, relevant for wind power
                # (equivalently, this may be seen as increaseing the
                # reserve margin requirement)
                reserve_factor = self.dev_data.reserve_factor
                max_value = max_value * reserve_factor
                if reserve_factor == 0:
                    # no reserve contribution
                    rf = 0
            cap_avail = rf * max_value
            p_generating = rf * pyomo_model.varDeviceFlow[self.id, "el", "out", t]
        elif "el" in self.carrier_in:
            # Loads (only consider if resere factor has been set)
            if self.dev_data.reserve_factor is not None:
                # load reduction possible
                f_lr = self.dev_data.reserve_factor
                load_reduction = f_lr * pyomo_model.varDeviceFlow[self.id, "el", "in", t]
        reserve = {
            "capacity_available": cap_avail,
            "capacity_used": p_generating,
            "loadreduction_available": load_reduction,
        }
        return reserve

    def compute_startup_penalty(self, pyomo_model: pyo.Model, timesteps: List[int]) -> float:
        """start/stop penalty - generalised from gas turbine startup cost."""
        penalty = 0
        if self.dev_data.start_stop is not None:
            penalty = (
                sum(pyomo_model.varDeviceStarting[self.id, t] for t in timesteps)
                * self.dev_data.start_stop.penalty_start
                + sum(pyomo_model.varDeviceStopping[self.id, t] for t in timesteps)
                * self.dev_data.start_stop.penalty_stop
            )
        return penalty

    def compute_operating_costs(self, pyomo_model: pyo.Model, timesteps: Union[pyo.Set, List[int]]) -> float:
        """average operating cost within selected timespan"""
        sum_cost = 0
        if self.dev_data.op_cost is not None:
            op_cost = self.dev_data.op_cost
            for t in pyomo_model.setHorizon:
                var_P = self.get_flow_var(pyomo_model=pyomo_model, t=t)
                sum_cost += op_cost * var_P
        # average per sec (simulation timestep drops out)
        return sum_cost / len(timesteps)

    def compute_cost_for_depleted_storage(
        self, pyomo_model: pyo.Model, timesteps: Optional[Union[List[int], pyo.Set]] = None
    ):
        return 0

    def compute_penalty(self, pyomo_model: pyo.Model, timesteps: List[int]):
        """Compute average penalty rate (cost, emission or similar per second)
        as defined by penalty functions and start/stop penalties"""
        penalty_rate = 0

        if self.dev_data.penalty_function is not None:
            if not hasattr(self, "_penalty_constraint"):
                logger.warning(f"Penalty function constraint is not implemented for {self.id}")
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

    def compute_CO2(self, pyomo_model: pyo.Model, timesteps: List[int]) -> float:
        """Standard for all devices unless overridden."""
        return 0
