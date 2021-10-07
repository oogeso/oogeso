import typing
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
        all_nodes: typing.Dict[str, NodeData],
        all_devices: typing.Dict[str, DeviceData],
        edges: typing.Dict[str, EdgeData],
        optimisation_parameters: OptimisationParametersData,
    ):
        "Energy carrier"
        self.carrier_id = carrier_data.id
        self.carrier_data = carrier_data
        self.edges = edges
        self.all_nodes = all_nodes
        self.all_devices = all_devices
        self.optimisation_parameters = optimisation_parameters

    def defineConstraints(self, pyomo_model):
        for edge_id, edge in self.edges.items():
            edge.defineConstraints(pyomo_model)


class Heat(Network):
    pass


class Hydrogen(Network):
    pass
