import typing
import pyomo.environ as pyo
from ...dto.oogeso_input_data_objects import (
    CarrierData,
    OptimisationParametersData,
    NodeData,
    EdgeData,
    DeviceData,
)


class Network:
    def __init__(
        self,
        carrier_data: CarrierData,
        edges: typing.Dict[str, EdgeData],
    ):
        "Energy carrier"
        self.carrier_id = carrier_data.id
        self.carrier_data = carrier_data
        self.edges = edges

    def defineConstraints(self, pyomo_model):
        piecewise_repn = pyo.value(pyomo_model.paramPiecewiseRepn)
        for edge in self.edges.values():
            edge.defineConstraints(pyomo_model, piecewise_repn)


class Heat(Network):
    pass


class Hydrogen(Network):
    pass
