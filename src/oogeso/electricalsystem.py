"""This module is for electrical computations
"""
import pandas as pd
import networkx as nx
import scipy

elbase = {'baseMVA':100, 'baseAngle':1}

def computePowerFlowMatrices(df_node,df_edge,baseZ=1):
    """
    Compute and return dc power flow matrices B' and DA

    Parameters
    ==========
    baseZ : float (impedance should already be in pu.)
            base value for impedance

    Returns
    =======
    (coeff_B, coeff_DA) : dictionary of matrix values

    """

    df_branch = df_edge[df_edge['type']=='el']
    #el_edges = df_edge[df_edge['type']=='el'].index

    b = (1/df_branch['reactance']*baseZ)
    #b0 = np.asarray(_susceptancePu(baseZ))

    # MultiDiGraph to allow parallel lines
    G = nx.MultiDiGraph()
    edges = [(df_branch['nodeFrom'][i],
              df_branch['nodeTo'][i],
              i,{'i':i,'b':b[i]})
              for i in df_branch.index]
    G.add_nodes_from(df_node.index)
    G.add_edges_from(edges)
    A_incidence_matrix = -nx.incidence_matrix(G,oriented=True,
                                             nodelist=df_node.index,
                                             edgelist=edges).T
    # Diagonal matrix
    D = scipy.sparse.diags(-b,offsets=0)
    DA = D*A_incidence_matrix

    # Bf constructed from incidence matrix with branch susceptance
    # used as weight (this is quite fast)
    Bf = -nx.incidence_matrix(G,oriented=True,
                             nodelist=df_node.index,
                             edgelist=edges,
                             weight='b').T
    Bbus = A_incidence_matrix.T * Bf

    n_i = df_node.index
    b_i = df_edge[df_edge['type']=='el'].index
    coeff_B = dict()
    coeff_DA = dict()

    #logging.info("Creating B and DA coefficients...")
    cx = scipy.sparse.coo_matrix(Bbus)
    for i,j,v in zip(cx.row, cx.col, cx.data):
        coeff_B[(n_i[i],n_i[j])] = v

    cx = scipy.sparse.coo_matrix(DA)
    for i,j,v in zip(cx.row, cx.col, cx.data):
        coeff_DA[(b_i[i],n_i[j])] = v

    return coeff_B,coeff_DA


#def _susceptancePu(df_edge,baseOhm=1):
#    '''If impedance is already given in pu, baseOhm should be 1
#    If not, well... baseOhm depends on the voltage level, so need to know
#    the nominal voltage at the bus to convert from ohm to pu.
#    '''
#    #return [-1/self.branch['reactance'][i]*baseOhm
#    #        for i in self.branch.index.tolist()]
#    return 1/df_edge['reactance']*baseOhm
