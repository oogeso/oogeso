"""This module is for electrical computations
"""
import pandas as pd
import networkx as nx
import scipy
import logging

elbase = {"baseMVA": 100, "baseAngle": 1}


def computePowerFlowMatrices(nodes, branches, baseZ=1):
    """
    Compute and return dc power flow matrices B' and DA

    Parameters
    ==========
    nodes : list of nodes (their ids)
    branches : dictionary {key:edge}
            edge is itself a dictionay with fields:
            (node_from, node_to, index, reactance_pu)
    baseZ : float (impedance should already be in pu.)
            base value for impedance

    Returns
    =======
    (coeff_B, coeff_DA) : dictionary of matrix values

    """

    df_branch = pd.DataFrame.from_dict(branches, orient="index")
    susceptance = 1 / df_branch["reactance"] * baseZ
    nodes = list(nodes)
    node_ids = nodes
    edge_ids = []
    edges = []
    for br_id, branch in branches.items():
        b = susceptance[br_id]
        edges.append(
            (branch["node_from"], branch["node_to"], br_id, {"i": br_id, "b": b})
        )
        edge_ids.append(br_id)

    # MultiDiGraph to allow parallel lines
    G = nx.MultiDiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    A_incidence_matrix = -nx.incidence_matrix(
        G, oriented=True, nodelist=nodes, edgelist=edges
    ).T
    # Diagonal matrix
    D = scipy.sparse.diags(-susceptance, offsets=0)
    DA = D * A_incidence_matrix

    # Bf constructed from incidence matrix with branch susceptance
    # used as weight (this is quite fast)
    Bf = -nx.incidence_matrix(
        G, oriented=True, nodelist=nodes, edgelist=edges, weight="b"
    ).T
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


# def _susceptancePu(df_edge,baseOhm=1):
#    '''If impedance is already given in pu, baseOhm should be 1
#    If not, well... baseOhm depends on the voltage level, so need to know
#    the nominal voltage at the bus to convert from ohm to pu.
#    '''
#    #return [-1/self.branch['reactance'][i]*baseOhm
#    #        for i in self.branch.index.tolist()]
#    return 1/df_edge['reactance']*baseOhm
