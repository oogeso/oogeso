import numpy as np
from oogeso.core.networks import pipelines


def test_darcyweissbach():
    p10 = 25.063e6  # Pa (nominal value for linearisation)
    p20 = 26.000e6  # Pa (nominal value for linearisation)
    D = 0.135  # m
    f = 0.01
    L = 1100  # m
    z = -100  # m
    rho = 1000  # kg/m3
    p1 = 25.081e6  # Pa (actual inlet pressure)
    Q = 0.277  # m3/s (actual flow)
    num_pipes = 15

    # Offline computed (with linear model) outlet pressure, given above assumptions:
    p2_correct_linear = 25.9960035e6

    # k2 = (8 * f * rho * L) / (np.pi ** 2 * D ** 5)
    # q0 = np.sqrt(1 / k2 * ((p10 - p20) - rho * grav * z))
    # p22 = p1 - rho * grav * z - k2 * (2 * q0 * Q / num_pipes - q0 ** 2)

    (p2_computed, q0) = pipelines.darcy_weissbach_p2(
        Q / num_pipes,
        p1=p1,
        f=f,
        rho=rho,
        diameter=D,
        length=L,
        height_difference=z,
        linear=True,
        p0_from=p10,
        p0_to=p20,
    )
    pressure_threshold = 1  # Pa
    assert (
        np.abs(p2_computed - p2_correct_linear) < pressure_threshold
    ), "Pressure drop calculation error"
