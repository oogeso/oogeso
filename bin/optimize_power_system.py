import core
import dto
import io
import plotting_utils


# Read in data, validate date et.c. with methods from io
some_test_data_file = "test_data.json"
json_data = io.file_io.read_data_from_json(some_test_data_file)

# deserialize json data to objects
power_network: dto.PowerNetwork = dto.deserialize_power_network_data(json_data)

# call optimization method, input are data objects
optimized_result = core.optimize_power_system(
    nodes=power_network.nodes,
    edges=power_network.edges,
    constraints=power_network.constraints,
)

# Serialize results and dump to file
optimized_result_serialized = dto.serialize_power_network_optimization_result(
    optimized_result
)
io.file_io.dump_to_file(
    json_data=optimized_result_serialized, filename="my_output_filename.json"
)

# Alternatively some plotting
plotting_utils.plot_something(optimized_result.part_of_result)
