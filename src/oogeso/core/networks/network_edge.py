from __future__ import annotations
import pyomo.environ as pyo
import logging
import numpy as np
import scipy
from . import electricalsystem as el_calc
import typing

if typing.TYPE_CHECKING:
    from oogeso.core.networks.network_node import NetworkNode
    from ...dto.oogeso_input_data_objects import EdgeData, EdgeFluidData


class NetworkEdge:
    "Network edge"

    def __init__(self, pyomo_model, optimiser, edge_data_object: EdgeData):
        self.id = edge_data_object.id
        self.edge_data = edge_data_object  # Edge data object as defined in the DTO
        self.pyomo_model = pyomo_model
        self.optimiser = optimiser

    def _ruleEdgeFlowEquations(self, model, t):
        """Flow as a function of node values (voltage/pressure)"""
        edge = self.id
        carrier = self.edge_data.carrier
        n_from = self.edge_data.node_from
        n_to = self.edge_data.node_to
        print_log = True if t == 0 else False

        if carrier == "el":
            """Depending on method, power flow depends on nodal voltage angles (dc-pf)
            or is unconstrained. DC-PF refers to the linearised power flow equations"""
            carrier_data = self.optimiser.all_carriers["el"]
            flowmethod = carrier_data.powerflow_method
            if flowmethod is None:
                return pyo.Constraint.Skip
            elif flowmethod == "transport":
                return pyo.Constraint.Skip
            elif flowmethod == "dc-pf":
                base_mva = el_calc.elbase["baseMVA"]
                base_angle = el_calc.elbase["baseAngle"]
                lhs = model.varEdgeFlow[edge, t]
                lhs = lhs / base_mva
                rhs = 0
                # TODO speed up creatioin of constraints - remove for loop
                n2s = [
                    k[1] for k in self.optimiser.elFlowCoeffDA.keys() if k[0] == edge
                ]
                for n2 in n2s:
                    rhs += self.optimiser.elFlowCoeffDA[edge, n2] * (
                        model.varElVoltageAngle[n2, t] * base_angle
                    )
                return lhs == rhs
            else:
                raise Exception(
                    "Power flow method must be None, 'transport' or 'dc-pf'"
                )

        elif carrier in ["gas", "wellstream", "oil", "water"]:
            p1 = model.varPressure[(n_from, carrier, "out", t)]
            p2 = model.varPressure[(n_to, carrier, "in", t)]
            Q = model.varEdgeFlow[edge, t]
            if self.edge_data.num_pipes is not None:
                num_pipes = self.edge_data.num_pipes
                if print_log:
                    logging.debug("{},{}: {} parallel pipes".format(edge, t, num_pipes))
                Q = Q / num_pipes
            p2_computed = self.compute_edge_pressuredrop(
                p1=p1, Q=Q, linear=True, print_log=print_log
            )
            return p2 == p2_computed
        else:
            # Other types of edges - no constraints other than max/min flow
            return pyo.Constraint.Skip

    def _ruleEdgeFlowMaxMin(self, model, t):
        edge = self.id
        pmax = self.edge_data.flow_max
        if self.edge_data.bidirectional:
            # electricity (can flow either way) (-max <= flow <= max)
            expr = pyo.inequality(-pmax, model.varEdgeFlow[edge, t], pmax)
        else:
            # flow only in specified direction (0 <= flow <= max)
            expr = pyo.inequality(0, model.varEdgeFlow[edge, t], pmax)
        return expr

    def _ruleEdgeFlowAndLoss(self, model, t, i):
        """Split edge flow into positive and negative part, for loss calculations"""
        edge = self.id
        if i == 1:
            expr = (
                model.varEdgeFlow[edge, t]
                == model.varEdgeFlow12[edge, t] - model.varEdgeFlow21[edge, t]
            )
        elif i == 2:
            expr = (
                model.varEdgeLoss[edge, t]
                == model.varEdgeLoss12[edge, t] + model.varEdgeLoss21[edge, t]
            )
        return expr

    def _loss_function_constraint(self, i):
        # Piecewise constraints require independent variable to be bounded:
        self.pyomo_model.varEdgeFlow12[self.id, :].setub(self.edge_data.flow_max)
        self.pyomo_model.varEdgeFlow21[self.id, :].setub(self.edge_data.flow_max)
        # Losses on cables are: P_loss = R/V^2 * P^2, i.e. quadratic function of power flow
        # Losses in transformers are: P_loss = ...
        lookup_table = self.edge_data.power_loss_function
        pw_x = lookup_table[0]
        pw_y = lookup_table[1]
        # # If we use loss function giving loss fraction instead of absolute loss in MW:
        # pw_y_fraction = lookup_table[1]
        # # Lookup table gives losses as a fraction, so to get absolute values we need
        # # to multiply by power transfer
        # # NOTE: This changes a linear curve to a second order curve, so need more than
        # # two x-points to represent it properly.
        # pw_y = [pw_y_fraction[i] * pw_x[i] for i in len(pw_x)]
        if i == 1:
            var_x = self.pyomo_model.varEdgeFlow12
            var_y = self.pyomo_model.varEdgeLoss12
        elif i == 2:
            var_x = self.pyomo_model.varEdgeFlow21
            var_y = self.pyomo_model.varEdgeLoss21
        pw_repn = self.optimiser.optimisation_parameters.piecewise_repn
        constr_penalty = pyo.Piecewise(
            [self.id],
            self.pyomo_model.setHorizon,
            var_y,
            var_x,
            pw_repn=pw_repn,
            pw_constr_type="EQ",
            pw_pts=pw_x,
            f_rule=pw_y,
        )
        return constr_penalty

    def defineConstraints(self):
        """Returns the set of constraints for the node."""

        if self.edge_data.flow_max is not None:
            constr_edge_bounds = pyo.Constraint(
                self.pyomo_model.setHorizon, rule=self._ruleEdgeFlowMaxMin
            )
            setattr(
                self.pyomo_model,
                "constrE_{}_{}".format(self.id, "bounds"),
                constr_edge_bounds,
            )

        constr_flow = pyo.Constraint(
            self.pyomo_model.setHorizon, rule=self._ruleEdgeFlowEquations
        )
        setattr(self.pyomo_model, "constrE_{}_{}".format(self.id, "flow"), constr_flow)

        # Power loss on electrical connections:
        if (self.edge_data.carrier == "el") and (
            self.edge_data.power_loss_function is not None
        ):
            # First, connecting variables (flow in different directions and loss variables)
            constr_loss = pyo.Constraint(
                self.pyomo_model.setHorizon,
                pyo.RangeSet(1, 2),
                rule=self._ruleEdgeFlowAndLoss,
            )
            setattr(
                self.pyomo_model, "constrE_{}_{}".format(self.id, "loss"), constr_loss
            )
            # Then, add equations for losses vs power flow (piecewise linear equations):
            for i in pyo.RangeSet(1, 2):
                constr_loss_function = self._loss_function_constraint(i)
                setattr(
                    self.pyomo_model,
                    "constrE_{}_{}_{}".format(self.id, "lossfunction", i),
                    constr_loss_function,
                )

    def _compute_exps_and_k(self, carrier_data):
        """Derive exp_s and k parameters for Weymouth equation"""
        edge_data: EdgeFluidData = self.edge_data
        # gas pipeline parameters - derive k and exp(s) parameters:
        ga = carrier_data

        # if 'temperature_K' in model.paramEdge[edge]:
        temp = edge_data.temperature_K
        height_difference = edge_data.height_m
        length = edge_data.length_km
        diameter = edge_data.diameter_mm
        s = 0.0684 * (ga.G_gravity * height_difference / (temp * ga.Z_compressibility))
        if s > 0:
            # height difference - use equivalent length
            sfactor = (np.exp(s) - 1) / s
            length = length * sfactor

        k = (
            4.3328e-8
            * ga.Tb_basetemp_K
            / ga.Pb_basepressure_MPa
            * (ga.G_gravity * temp * length * ga.Z_compressibility) ** (-1 / 2)
            * diameter ** (8 / 3)
        )
        exp_s = np.exp(s)
        return exp_s, k

    def compute_edge_pressuredrop(
        self, p1, Q, method=None, linear=False, print_log=True
    ):
        """Compute pressure drop in pipe

        parameters
        ----------
         p1 : float
            pipe inlet pressure (MPa)
        Q : float
            flow rate (Sm3/s)
        method : string
            None, weymouth, darcy-weissbach
        linear : boolean
            whether to use linear model or not

        Returns
        -------
        p2 : float
            pipe outlet pressure (MPa)"""

        edge = self.id
        edge_data = self.edge_data
        height_difference = edge_data.height_m
        method = None
        carrier = edge_data.carrier
        carrier_data = self.optimiser.all_carriers[carrier]
        if carrier_data.pressure_method is not None:
            method = carrier_data.pressure_method

        n_from = edge_data.node_from
        n_to = edge_data.node_to
        n_from_obj: NetworkNode = self.optimiser.all_nodes[n_from]
        n_to_obj: NetworkNode = self.optimiser.all_nodes[n_to]
        p0_from = n_from_obj.get_nominal_pressure(carrier, "out")
        p0_to = n_to_obj.get_nominal_pressure(carrier, "in")
        if (p0_from is not None) and (p0_to is not None):
            if linear & (p0_from == p0_to):
                method = None
                if print_log:
                    logging.debug(
                        ("{}-{}: Pipe without pressure drop" " ({} / {} MPa)").format(
                            n_from, n_to, p0_from, p0_to
                        )
                    )
        elif linear:
            # linear equations, but nominal values not given - assume no drop
            # logging.debug("{}-{}: Aassuming no  pressure drop".format(n_from, n_to))
            method = None
        else:
            # use non-linear equations, no nominal pressure required
            pass

        if method is None:
            # no pressure drop
            p2 = p1
            return p2

        elif method == "weymouth":
            """
            Q = k * sqrt( Pin^2 - e^s Pout^2 ) [Weymouth equation, nonlinear]
                => Pout = sqrt(Pin^2-Q^2/k^2)/e^2
            Q = c * (Pin_0 Pin - e^s Pout0 Pout) [linearised version]
                => Pout = (Pin0 Pin - Q/c)/(e^s Pout0)
            c = k/sqrt(Pin0^2 - e^s Pout0^2)

            REFERENCES:
            1) E Sashi Menon, Gas Pipeline Hydraulics, Taylor & Francis (2005),
            https://doi.org/10.1201/9781420038224
            2) A Tomasgard et al., Optimization  models  for  the  natural  gas
            value  chain, in: Geometric Modelling, Numerical Simulation and
            Optimization. Springer Verlag, New York (2007),
            https://doi.org/10.1007/978-3-540-68783-2_16
            """
            exp_s, k = self._compute_exps_and_k(carrier_data=carrier_data)
            if print_log:
                logging.debug("pipe {}: exp_s={}, k={}".format(edge, exp_s, k))
            if linear:
                p_from = p1
                #                p_from = model.varPressure[(n_from,carrier,'out',t)]
                #                p_to = model.varPressure[(n_to,carrier,'in',t)]
                X0 = p0_from ** 2 - exp_s * p0_to ** 2
                #                logging.info("edge {}-{}: X0={}, p1={},Q={}"
                #                    .format(n_from,n_to,X0,p1,Q))
                coeff = k * (X0) ** (-1 / 2)
                #                Q_computed = coeff*(p0_from*p_from - exp_s*p0_to*p_to)
                p2 = (p0_from * p_from - Q / coeff) / (exp_s * p0_to)
            else:
                # weymouth eqn (non-linear)
                p2 = 1 / exp_s * (p1 ** 2 - Q ** 2 / k ** 2) ** (1 / 2)

        elif method == "darcy-weissbach":
            grav = 9.98  # m/s^2
            rho = carrier_data.rho_density
            D = edge_data.diameter_mm / 1000
            L = edge_data.length_km * 1000

            if (carrier_data.viscosity is not None) & (not linear):
                # compute darcy friction factor from flow rate and viscosity
                mu = carrier_data.viscosity
                Re = 2 * rho * Q / (np.pi * mu * D)
                f = 1 / (0.838 * scipy.special.lambertw(0.629 * Re)) ** 2
                f = f.real
            elif carrier_data.darcy_friction is not None:
                f = carrier_data.darcy_friction
                Re = None
            else:
                raise Exception(
                    "Must provide viscosity or darcy_friction for {}".format(carrier)
                )
            if linear:
                p_from = p1
                k = np.sqrt(np.pi ** 2 * D ** 5 / (8 * f * rho * L))
                sqrtX = np.sqrt(
                    p0_from * 1e6 - p0_to * 1e6 - rho * grav * height_difference
                )
                Q0 = k * sqrtX
                if print_log:
                    logging.debug(
                        (
                            "derived pipe ({}) flow rate:"
                            " Q={}, linearQ0={:5.3g},"
                            " friction={:5.3g}"
                        ).format(edge, Q, Q0, f)
                    )
                p2 = p_from - 1e-6 * (Q - Q0) * 2 * sqrtX / k - (p0_from - p0_to)
                # linearised darcy-weissbach:
                # Q = Q0 + k/(2*sqrtX)*(p_from-p_to - (p0_from-p0_to))
            else:
                # darcy-weissbach eqn (non-linear)
                p2 = 1e-6 * (
                    p1 * 1e6
                    - rho * grav * height_difference
                    - 8 * f * rho * L * Q ** 2 / (np.pi ** 2 * D ** 5)
                )
        else:
            raise Exception(
                "Unknown pressure drop calculation method ({})".format(method)
            )
        return p2


def darcy_weissbach_Q(p1, p2, f, rho, diameter_mm, length_km, height_difference_m=0):
    """compute flow rate from darcy-weissbach eqn

    parameters
    ----------
    p1 : float
        pressure at pipe input (Pa)
    p2 : float
        pressure at pipe output (Pa)
    f : float
        friction factor
    rho : float
        fluid density (kg/m3)
    diameter_mm : float
        pipe inner diameter (mm)
    length_km : float
        pipe length (km)
    height_difference_m : float
        height difference output vs input (m)

    """

    grav = 9.98
    L = length_km * 1000
    D = diameter_mm / 1000
    k = 8 * f * rho * L / (np.pi ** 2 * D ** 5)
    Q = np.sqrt(((p1 - p2) * 1e6 - rho * grav * height_difference_m) / k)
    return Q
