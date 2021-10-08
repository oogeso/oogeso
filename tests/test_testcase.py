import oogeso
import oogeso.core.util
import oogeso.io.file_io
import oogeso.dto.oogeso_input_data_objects as dto

def test_file_input():
    # Test oogeso object creation from yaml file

    profiles_dfs = oogeso.io.file_io.read_profiles_from_csv(
        filename_forecasts="examples/testcase2_profiles_forecasts.csv",
        filename_nowcasts="examples/testcase2_profiles_nowcasts.csv",
        timestamp_col="timestamp",exclude_cols=["timestep"])
    profiles_json = oogeso.core.util.create_timeseriesdata(
        profiles_dfs["forecast"],profiles_dfs["nowcast"],
        time_start=None,time_end=None,timestep_minutes=15)
    data0 = oogeso.io.file_io.read_data_from_yaml('examples/test case2.yaml')
    data0.profiles = profiles_json

    # If not failed above, it's OK
    assert True

def test_DevicePowersource():
    startstop_data = dto.StartStopData(
        is_on_init=False,penalty_start=1,penalty_stop=0,
        delay_start_minutes=30,minimum_time_on_minutes==0,minimum_time_off_minutes=0)
    dev_data = dto.DevicePowersourceData(
        id="gen",node_id="node",name="generator 1",include=True,
        profile=None,flow_min=None,flow_max=50,max_ramp_up=50,
        max_ramp_down=50,start_stop=startstop_data,reserve_factor=1,op_cost=0,
        penalty_function=[[0,50],[1,20]])
    pyomo_model=None
    optimiser=None
    obj = oogeso.devices.Powersource(pyomo_model,optimiser,dev_data)
    assert isinstance(obj,oogeso.devices.Powersource)
