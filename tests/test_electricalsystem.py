import pandas as pd
import numpy as np
from oogeso.core.networks import electricalsystem as el


def test_powerflow_matrices():
    # IEEE 3-bus system:
    #  1--0.03--2
    #   \      /
    #  0.05   0.08
    #     \ /
    #      3
    nodes = ["n1", "n2", "n3"]
    branches = {
        "b1": {"node_from": "n1", "node_to": "n2", "reactance": 0.03},
        "b2": {"node_from": "n1", "node_to": "n3", "reactance": 0.05},
        "b3": {"node_from": "n2", "node_to": "n3", "reactance": 0.08},
    }

    baseZ = 1  # pu
    M = el.computePowerFlowMatrices(nodes, branches, baseZ=baseZ)
    matrix_B = M[0]
    matrix_DA = M[1]

    # offline calculation

    # diagnal matrix
    m_D = np.diag([-1 / 0.03, -1 / 0.05, -1 / 0.08])
    # node-branch incidence matrix: (row=branch, col=node) (from=+,to=-)
    m_A = np.array([[1, -1, 0], [1, 0, -1], [0, 1, -1]])
    m_DA = np.dot(m_D, m_A)

    # susceptance matrix (admittance matrix in DC power flow approx):
    # Bkk = b_k + sum_j!=k b_kj; b_k=0 (shunt)
    m_B = -np.array(
        [
            [-1 / 0.03 - 1 / 0.05, 1 / 0.03, +1 / 0.05],
            [+1 / 0.03, -1 / 0.03 - 1 / 0.08, +1 / 0.08],
            [+1 / 0.05, +1 / 0.08, -1 / 0.05 - 1 / 0.08],
        ]
    )

    assert isinstance(matrix_B, dict)
    assert isinstance(matrix_DA, dict)

    assert m_DA[0, 0] == matrix_DA[("b1", "n1")]
    assert m_DA[0, 1] == matrix_DA[("b1", "n2")]
    assert m_DA[0, 2] == 0
    assert ("b1", "n3") not in matrix_DA
    assert m_DA[1, 0] == matrix_DA[("b2", "n1")]
    assert m_DA[1, 1] == 0
    assert ("b2", "n2") not in matrix_DA
    assert m_DA[1, 2] == matrix_DA[("b2", "n3")]
    assert m_DA[2, 0] == 0
    assert ("b3", "n1") not in matrix_DA
    assert m_DA[2, 1] == matrix_DA[("b3", "n2")]
    assert m_DA[2, 2] == matrix_DA[("b3", "n3")]

    assert m_B[0, 0] == matrix_B[("n1", "n1")]
    assert m_B[0, 1] == matrix_B[("n1", "n2")]
    assert m_B[0, 2] == matrix_B[("n1", "n3")]
    assert m_B[1, 0] == matrix_B[("n2", "n1")]
    assert m_B[1, 1] == matrix_B[("n2", "n2")]
    assert m_B[1, 2] == matrix_B[("n2", "n3")]
    assert m_B[2, 0] == matrix_B[("n3", "n1")]
    assert m_B[2, 1] == matrix_B[("n3", "n2")]
    assert m_B[2, 2] == matrix_B[("n3", "n3")]
