import numpy as np


def ic_el_matrix_fs(basic_matrices, comp_struct, el_phys_props, edge_props, edge_nodes_pos):
    """
    Compute element matrices for fluid-solid interface.
    [T.Zharnikov, SMR v0.3_08.2014]
    Converted to Python 2025
    """

    nfl_edge_nodes = comp_struct['Advanced']['NEdge_nodes']
    ns_edge_nodes = 3 * comp_struct['Advanced']['NEdge_nodes']

    # Allocate memory
    el_matrices = {
        'NfltRhonNMatrix': np.zeros((nfl_edge_nodes, ns_edge_nodes))
    }

    # Prepare properties
    rho = el_phys_props['DfEl']['RhoVec'][edge_nodes_pos['Df']]  # fluid
    ones_matrix = np.ones(nfl_edge_nodes)
    id_matrix = np.eye(3)

    # Create enlarged convolution matrix
    nenene_conv_large = np.zeros((nfl_edge_nodes, nfl_edge_nodes, ns_edge_nodes))
    node_numbers = np.arange(1, nfl_edge_nodes + 1)

    for ii in range(3):
        indices = ii + 3 * (node_numbers - 1)
        indices = indices.astype(int) - 1  # 0-index
        nenene_conv_large[:, :, indices] = basic_matrices['NENENEConvMatrixInt']

    # Compute Rho_nE3_large
    rho_nE3_large = np.zeros((nfl_edge_nodes, nfl_edge_nodes, ns_edge_nodes))
    for kk in range(nfl_edge_nodes):
        rho_nE3_kk = rho[kk] * edge_props['Normal'][:3].T @ id_matrix
        rho_nE3_large[:, kk, :] = np.kron(ones_matrix, rho_nE3_kk).reshape(nfl_edge_nodes, 1, ns_edge_nodes)

    # Compute final matrix
    nflt_rhonN_unsummed = rho_nE3_large * nenene_conv_large
    el_matrices['NfltRhonNMatrix'] = np.sum(nflt_rhonN_unsummed, axis=1)

    return el_matrices