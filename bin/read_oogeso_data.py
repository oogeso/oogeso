import json

import oogeso.io.file_io
from oogeso.dto.serialisation import DataclassJSONDecoder, serialize_oogeso_data

# Read in data, validate date et.c. with methods from io
test_data_file = "examples/test case2.yaml"
json_data = oogeso.io.file_io.read_data_from_yaml(test_data_file)

json_formatted_str = json.dumps(json_data, indent=2)
print("Type json formatted str=", type(json_formatted_str))

# deserialize json data to objects
# encoder = oogeso.dto.oogeso_input_data_objects.DataclassJSONEncoder
decoder = DataclassJSONDecoder
# decoder = json.JSONDecoder()
with open("examples/energysystem.json", "r") as jsonfile:
    energy_system = json.load(jsonfile, cls=decoder)

es_str = serialize_oogeso_data(energy_system)
print("Type seriealised=", type(es_str))
mydecoder = decoder()
energy_system = mydecoder.decode(json_formatted_str)
print("Type energysystem=", type(energy_system))
# energy_system = json.loads(
#    json_formatted_str, cls=encoder
# )

# energy_system: oogeso.dto.oogeso_input_data_objects.EnergySystem = (
#    oogeso.dto.oogeso_input_data_objects.deserialize_oogeso_data(json_data)
# )

print("========================")
print("Energy system:")
# print("Energy system type=", type(energy_system))
# print("Nodes: ", energy_system.nodes)
# print("Node1: ", energy_system.nodes["node1"])
# print("Parameters: ", energy_system.parameters)
# print("Parameters type=", type(energy_system.parameters))
# print("planning horizon: ", energy_system.parameters.planning_horizon)
# print("Carriers: ", energy_system.carriers)
print(energy_system)
