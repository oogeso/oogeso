import logging
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pyomo.environ as pyo
import scipy

from oogeso import dto
from oogeso.core.networks.edge import FluidEdge
from oogeso.core.networks.network import Network

logger = logging.getLogger(__name__)

GRAVITY_ACCELERATION_CONSTANT = 9.8  # m/s^2


class FluidNetwork(Network):
    carrier_data: dto.CarrierFluidData
    edges: Dict[str, FluidEdge]

    def define_constraints(self, pyomo_model: pyo.Model):
        super().define_constraints(pyomo_model)

        # additional
        if self.carrier_data.pressure_method in ["weymouth", "darcy-weissbach"]:
            edgelist = [e for e in self.edges.keys()]
            if edgelist:
                # non-empty list of edges
                constr_flow = pyo.Constraint(
                    edgelist,
                    pyomo_model.setHorizon,
                    rule=self._rule_pipeline_flow,
                )
                setattr(
                    pyomo_model,
                    "constrE_{}_{}".format(self.carrier_id, "flow"),
                    constr_flow,
                )

    def _rule_pipeline_flow(self, model: pyo.Model, edge, t: int) -> Union[pyo.Expression, pyo.Constraint.Skip]:
        """Pipeline flow vs pressure drop"""
        # edge = self.id
        edge_obj = self.edges[edge]
        edge_data = edge_obj.edge_data
        carrier = edge_data.carrier
        n_from = edge_data.node_from
        n_to = edge_data.node_to
        print_log = True if t == 0 else False

        p1 = model.varPressure[(n_from, carrier, "out", t)]
        p2 = model.varPressure[(n_to, carrier, "in", t)]
        Q = model.varEdgeFlow[edge, t]
        if hasattr(edge_data, "num_pipes"):
            if edge_data.num_pipes is not None:
                num_pipes = edge_data.num_pipes
                if print_log:
                    logger.debug("{},{}: {} parallel pipes".format(edge, t, num_pipes))
                Q = Q / num_pipes
        else:
            raise ValueError("FluidNetworkEdge is expected to have the attribute num_pipes.")
        p2_computed = self.compute_edge_pressuredrop(edge_obj, p1=p1, Q=Q, linear=True, print_log=print_log)
        return p2 == p2_computed

    @staticmethod
    def _compute_exps_and_k(edge_data: dto.EdgeFluidData, carrier_data: dto.CarrierFluidData):
        """Derive exp_s and k parameters for Weymouth equation"""
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
        self, edge: FluidEdge, p1, Q, method: Optional[str] = None, linear: bool = False, print_log: bool = True
    ):
        """Compute pressure drop in pipe

        parameters
        ----------
        edge : edge object
        p1 : float
            pipe inlet pressure (MPa)
        Q : float
            flow rate (Sm3/s)
        method : string
            None, weymouth, darcy-weissbach
        linear : boolean
            whether to use linear model or not
        print_log: bool
            To print or not to print the log

        Returns
        -------
        p2 : float
            pipe outlet pressure (MPa)"""

        # edge = self.id
        edge_data = edge.edge_data
        edge_id = edge_data.id
        carrier = edge_data.carrier
        carrier_data = self.carrier_data
        if carrier_data.pressure_method is not None:
            method = carrier_data.pressure_method

        n_from = edge_data.node_from
        n_to = edge_data.node_to
        p0_from = edge.node_from.get_pressure_nominal(carrier, "out")
        p0_to = edge.node_to.get_pressure_nominal(carrier, "in")
        if (p0_from is not None) and (p0_to is not None):
            if linear & (p0_from == p0_to):
                method = None
                if print_log:
                    logger.debug(f"{n_from}-{n_to}: Pipe without pressure drop" " ({p0_from} / {p0_to} MPa)")
        elif linear:
            # linear equations, but nominal values not given - assume no drop
            # logger.debug("{}-{}: Aassuming no  pressure drop".format(n_from, n_to))
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
            exp_s, k = self._compute_exps_and_k(edge_data, carrier_data=carrier_data)
            if print_log:
                logger.debug("pipe {}: exp_s={}, k={}".format(edge_id, exp_s, k))
            if linear:
                p_from = p1
                x0 = p0_from ** 2 - exp_s * p0_to ** 2
                coeff = k * x0 ** (-1 / 2)
                p2 = (p0_from * p_from - Q / coeff) / (exp_s * p0_to)
            else:
                # weymouth eqn (non-linear)
                p2 = 1 / exp_s * (p1 ** 2 - Q ** 2 / k ** 2) ** (1 / 2)
            return p2

        elif method == "darcy-weissbach":
            rho = carrier_data.rho_density
            D = edge_data.diameter_mm / 1000
            L = edge_data.length_km * 1000
            height_difference = edge_data.height_m

            if (carrier_data.viscosity is not None) & (not linear):
                # compute darcy friction factor from flow rate and viscosity (nonlinear)
                mu = carrier_data.viscosity
                Re = 2 * rho * Q / (np.pi * mu * D)
                f = 1 / (0.838 * scipy.special.lambertw(0.629 * Re)) ** 2
                f = f.real
            elif hasattr(carrier_data, "darcy_friction") and getattr(carrier_data, "darcy_friction") is not None:
                f = carrier_data.darcy_friction
            else:
                raise Exception("Must provide viscosity or darcy_friction for {}".format(carrier))
            (p2, Q0) = darcy_weissbach_p2(
                Q,
                p1 * 1e6,
                f=f,
                rho=rho,
                diameter=D,
                length=L,
                height_difference=height_difference,
                linear=linear,
                p0_from=p0_from * 1e6,
                p0_to=p0_to * 1e6,
            )
            p2 = p2 * 1e-6  # Convert to MPa
            if print_log:
                logger.debug(f"derived pipe ({edge_id}) flow rate:' ' Q={Q}, linearQ0={Q0:5.3g}, ' ' friction={f:5.3g}")

            return p2
        else:
            raise Exception("Unknown pressure drop calculation method ({})".format(method))


class GasNetwork(FluidNetwork):
    pass


class OilNetwork(FluidNetwork):
    pass


class WaterNetwork(FluidNetwork):
    pass


class WellStreamNetwork(FluidNetwork):
    pass


def darcy_weissbach_p2(
    Q,
    p1,
    f,
    rho,
    diameter,
    length,
    height_difference=0,
    linear=True,
    p0_from=None,  # Pa
    p0_to=None,  # Pa
) -> Tuple[float, ...]:
    """compute outlet pressure from darcy-weissbach equation

    parameters
    ----------
    p1 : float
        pressure at pipe inlet (Pa)
    Q : float
        flow rate (Sm3/s)
    f : float
        friction factor
    rho : float
        fluid density (kg/m3)
    diameter : float
        pipe inner diameter (m)
    length : float
        pipe length (m)
    height_difference : float
        height difference outlet vs inlet (m)
    linear : True or False
    p0_from : Optional pressure from
    p0_to : Optional pressure to

    """
    D = diameter  # m
    L = length  # m
    grav = GRAVITY_ACCELERATION_CONSTANT
    q0: float = np.nan
    if linear:
        k2 = (8 * f * rho * L) / (np.pi ** 2 * D ** 5)
        q0_squared = 1 / k2 * ((p0_from - p0_to) - rho * grav * height_difference)
        if q0_squared < 0:
            logging.error(
                "Negative q0^2 (flow direction error) q0^2=%s p1=%s p2=%s rho=%s grav=%s z=%s",
                q0_squared,
                p0_from,
                p0_to,
                rho,
                grav,
                height_difference,
            )
        q0 = np.sqrt(q0_squared)
        p2: float = p1 - rho * grav * height_difference - k2 * (2 * q0 * Q - q0 ** 2)
    else:
        # Quadratic in Q
        p2: float = p1 - rho * grav * height_difference - 8 * f * rho * L * Q ** 2 / (np.pi ** 2 * D ** 5)
    return tuple([p2, q0])


def darcy_weissbach_Q(p1, p2, f, rho, diameter_mm, length_km, height_difference_m=0):
    """compute flow rate from non-linear darcy-weissbach eqn

    parameters
    ----------
    p1 : float
        pressure at pipe input (MPa)
    p2 : float
        pressure at pipe output (MPa)
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
