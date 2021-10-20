import oogeso
import oogeso.dto.oogeso_input_data_objects as dto


def test_DevicePowersource():
    startstop_data = dto.StartStopData(
        is_on_init=False,
        penalty_start=1,
        penalty_stop=0,
        delay_start_minutes=30,
        minimum_time_on_minutes=0,
        minimum_time_off_minutes=0,
    )
    dev_data = dto.DevicePowersourceData(
        id="gen",
        node_id="node",
        name="generator 1",
        include=True,
        profile=None,
        flow_min=None,
        flow_max=50,
        max_ramp_up=50,
        max_ramp_down=50,
        start_stop=startstop_data,
        reserve_factor=1,
        op_cost=0,
        penalty_function=[[0, 50], [1, 20]],
    )
    pyomo_model = None
    # optimiser = oogeso.Optimiser(data=None)

    obj = oogeso.devices.Powersource(pyomo_model, optimiser, dev_data)
    assert isinstance(obj, oogeso.devices.Powersource)
