from oogeso import dto
from oogeso.dto.serialisation import deserialize_oogeso_data, serialize_oogeso_data


def test_create_energy_system_data():
    energy_system = dto.EnergySystemData(
        carriers=[
            dto.CarrierElData(
                id="el",
                reference_node="node1",
                el_reserve_margin=-1,
                reserve_storage_minutes=30,
            ),
            dto.CarrierHeatData(id="heat"),
            dto.CarrierGasData(
                id="gas",
                co2_content=0.4,
                Pb_basepressure_MPa=100,
                R_individual_gas_constant=9,
                Tb_basetemp_K=300,
                Z_compressibility=0.9,
                energy_value=40,
                k_heat_capacity_ratio=0.7,
                rho_density=0.6,
            ),
        ],
        nodes=[dto.NodeData(id="node1"), dto.NodeData(id="node2")],
        edges=[
            dto.EdgeElData(
                id="edge1",
                node_from="node1",
                node_to="node2",
                flow_max=500,
                reactance=1,
                resistance=1,
                voltage=33,
                length_km=10,
            ),
            dto.EdgeElData(
                id="edge2",
                node_from="node2",
                node_to="node1",
                flow_max=50,
                voltage=33,
                length_km=10,
                # reactance=1,
                # resistance=1,
                power_loss_function=([0, 1], [0, 0.02]),
            ),
        ],
        devices=[
            dto.DeviceSourceElData(id="elsource", node_id="node1", flow_max=12),
            dto.DevicePowerSourceData(id="gt1", node_id="node2", flow_max=30, profile="profile1"),
            dto.DeviceSinkElData(id="demand", node_id="node2", flow_min=4, profile="profile1"),
        ],
        parameters=dto.OptimisationParametersData(
            time_delta_minutes=30,
            planning_horizon=12,
            optimisation_timesteps=6,
            forecast_timesteps=6,
            objective="exportRevenue",
        ),
        profiles=[dto.TimeSeriesData(id="profile1", data=[12, 10, 21])],
    )

    # Only check if the calls raise exceptions:

    energy_system_str = energy_system.__str__()
    assert isinstance(energy_system_str, str)

    json_str = energy_system.json()
    assert isinstance(json_str, str)


def test_serialisation_deserialisation(testcase2_data, testcase2_expected_result):
    data_str = serialize_oogeso_data(testcase2_data)
    assert isinstance(data_str, str)

    data_obj = deserialize_oogeso_data(data_str)
    assert isinstance(data_obj, dto.EnergySystemData)

    assert data_obj == testcase2_data
