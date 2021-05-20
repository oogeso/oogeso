import json
from copy import deepcopy
from dataclasses import dataclass, is_dataclass, asdict
from typing import List, Optional, Tuple, Any, Union


@dataclass
class PowerSource:
    power_to_penalty_data: Tuple[
        List[float], List[float]
    ]  # Penalty may be fuel, emissions, cost and combinations of these
    spinning_reserve: Optional[
        float
    ] = 0  # question: should spinning reserve be a Node attribute instead? Spinning reserve needed for all generators or only on one?


@dataclass
class Node:
    id: str
    power_sources: List[PowerSource]
    minimum_power_generation: float
    power_need: Optional[List[float]] = None
    maximum_power_delivery: Optional[Union[float, List[float]]] = None
    # For Wind park, this may be a list of power per input time series in popwer need
    # For Power from shore, this may be a constant maximum power delivery


# Examples for illustration
platform = Node(
    id="platform",
    power_need=[20, 33.0, 45.0],
    power_sources=[
        PowerSource(
            power_to_penalty_data=([0, 0.1, 20.0], [0, 50.000, 10000.0]),
            spinning_reserve=0.0,
        )
    ],
    minimum_power_generation=0.0,
)
offshore_wind_park = Node(
    id="windpark",
    power_sources=[
        PowerSource(
            power_to_penalty_data=(
                [0, 300.0],
                [0, 0],
            ),  # This power does not here have any penalty. May be cost?
        ),
    ],
    minimum_power_generation=0.0,
    maximum_power_delivery=[
        70.0,
        75,
        85.0,
    ],  # Actual delivered power from wind park for our evaluation points
)
power_from_shore = Node(
    id="PfS",
    power_sources=[
        PowerSource(
            power_to_penalty_data=(
                [0, 300.0],
                [0, 0],
            ),  # This power does not here have any penalty. May be cost?
        ),
    ],
    minimum_power_generation=0.0,
    maximum_power_delivery=50.0,
)


@dataclass
class PowerLink:
    id_from_node: str
    id_to_node: str
    maximum_load_capacity: Optional[float]
    power_loss_function: dict
    directional: Optional[bool] = False


@dataclass
class PowerNetwork:
    nodes: List[Node]
    edges: List[PowerLink]
    constraints: dict


generator_a1 = PowerSource(
    power_to_penalty_data=(
        [0.0, 0.1, 10.0, 20.0],
        [0.0, 75000.0, 80000.0, 130000.0],
    ),
    spinning_reserve=2,
)
generator_a2 = deepcopy(generator_a1)
generator_a3 = PowerSource(
    power_to_penalty_data=(
        [0.0, 0.1, 10.0, 15.0],
        [0.0, 60000.0, 70000.0, 100000.0],
    ),
    spinning_reserve=2,
)
generator_a4 = deepcopy(generator_a3)
power_need_a = [25.0, 22.0, 35.0, 23.0, 27.0, 55.0, 42.0]
skjold_a = Node(
    id="skjold_a",
    power_need=power_need_a,
    power_sources=[generator_a1, generator_a2, generator_a3, generator_a4],
    minimum_power_generation=5,
)

power_need_b = [8.0, 7.0, 10.0, 17.5, 5.0, 5.0, 11.0]
skjold_b = Node(
    id="skjold_b",
    power_need=power_need_b,
    power_sources=[],
    minimum_power_generation=0,
)

generator_c1 = PowerSource(
    power_to_penalty_data=(
        [0.00, 0.10, 5.00, 15.00, 20.00, 25.00, 30.00],
        [0.0, 84000.0, 84000.0, 120000.0, 150000.0, 160000.0, 180000.0],
    ),
    spinning_reserve=1.5,
)


power_need_c = [35.0, 52.0, 50.0, 43.0, 37.0, 35.0, 47.0]
skjold_c = Node(
    id="skjold_c",
    power_need=power_need_c,
    power_sources=[
        generator_c1,
        deepcopy(generator_c1),
        deepcopy(generator_c1),
    ],
    minimum_power_generation=10.0,
)

link_a_b = PowerLink(
    id_from_node="skjold_a",
    id_to_node="skjold_b",
    power_loss_function={},
    maximum_load_capacity=None,
    directional=True,  # Means it only goes from skjold_a to skjold_b, not the other way around
)
link_a_c = PowerLink(
    id_from_node="skjold_a",
    id_to_node="skjold_c",
    power_loss_function={},
    maximum_load_capacity=None,
)

power_network = PowerNetwork(
    nodes=[
        skjold_a,
        skjold_b,
        skjold_c,
    ],  # Potentially skip this as all the nodes are defined in the edges anyway?
    edges=[link_a_b, link_a_c],
    constraints={},
)


# Serialize and dump to print/file or whatever
class DataclassJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any):
        if is_dataclass(obj=obj):
            return asdict(obj=obj)
        return super().default(obj)


serialized = json.dumps(power_network, cls=DataclassJSONEncoder, indent=2)

print(serialized)
