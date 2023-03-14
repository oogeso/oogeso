"""This module is for electrical computations
"""
import networkx as nx
import pandas as pd
import scipy

elbase = {"baseMVA": 100, "baseAngle": 1}


def compute_power_flow_matrices(nodes, branches, base_Z=1):
    """
    Compute and return dc power flow matrices B' and DA

    Parameters
    ==========
    nodes : list of nodes (their ids)
    branches : dictionary {key:edge}
            edge is itself a dictionay with fields:
            (node_from, node_to, index, reactance_pu)
    base_Z : float (impedance should already be in pu.)
            base value for impedance

    Returns
    =======
    (coeff_B, coeff_DA) : dictionary of matrix values

    """

    df_branch = pd.DataFrame.from_dict(branches, orient="index")
    susceptance = 1 / df_branch["reactance"] * base_Z
    nodes = list(nodes)
    node_ids = nodes
    edge_ids = []
    edges = []
    for br_id, branch in branches.items():
        b = susceptance[br_id]
        edges.append((branch["node_from"], branch["node_to"], br_id, {"i": br_id, "b": b}))
        edge_ids.append(br_id)

    # MultiDiGraph to allow parallel lines
    G = nx.MultiDiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    A_incidence_matrix = -nx.incidence_matrix(G, oriented=True, nodelist=nodes, edgelist=edges).T
    
    # NOTE:
    # networkx returns ndarray instead of matrix (after recent change)
    # this affectts the * operator, which meanselement-wise multiplication, 
    # for numpy.matrix but not for numpy.ndarray
    # To get elementwise multiplication below, we therefore convert to matrix.
    # Example
    # This gives correct result:
    #     (scipy.sparse.csc_matrix(A_incidence_matrix).T*scipy.sparse.csc_matrix(Bf)).todense() 
    # This fails (when the matrices are (sparse) ndarrays):
    #     (A_incidence_matrix.T*Bf)
    A_incidence_matrix = scipy.sparse.csc_matrix(A_incidence_matrix)

    # Diagonal matrix
    D = scipy.sparse.diags(-susceptance, offsets=0)
    # Element-wise multiplication:
    DA = D * A_incidence_matrix

    # Bf constructed from incidence matrix with branch susceptance
    # used as weight (this is quite fast)
    Bf = -nx.incidence_matrix(G, oriented=True, nodelist=nodes, edgelist=edges, weight="b").T
    # See note above for this conversion
    Bf = scipy.sparse.csc_matrix(Bf)
    # Element-wise multiplication:
    Bbus = A_incidence_matrix.T * Bf

    coeff_B = dict()
    coeff_DA = dict()
    cx = scipy.sparse.coo_matrix(Bbus)
    for i, j, v in zip(cx.row, cx.col, cx.data):
        coeff_B[(node_ids[i], node_ids[j])] = v

    cx = scipy.sparse.coo_matrix(DA)
    for i, j, v in zip(cx.row, cx.col, cx.data):
        coeff_DA[(edge_ids[i], node_ids[j])] = v

    return coeff_B, coeff_DA
