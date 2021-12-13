import typing

import pyomo.environ as pyo

from oogeso import dto
from oogeso.core.networks.edge import Edge


class Network:
    def __init__(
        self,
        carrier_data: dto.CarrierData,
        edges: typing.Dict[str, Edge],
    ):
        """Energy carrier."""
        self.carrier_id = carrier_data.id
        self.carrier_data = carrier_data
        self.edges = edges

    def define_constraints(self, pyomo_model: pyo.Model):
        piecewise_repn = pyo.value(pyomo_model.paramPiecewiseRepn)
        for edge in self.edges.values():
            edge.define_constraints(pyomo_model=pyomo_model, piecewise_repn=piecewise_repn)


class Heat(Network):
    pass


class Hydrogen(Network):
    pass
