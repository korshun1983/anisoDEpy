# File 9: convolve_edge_matrices.py
import numpy as np
from scipy.signal import convolve
import os


def convolve_edge_matrices(comp_struct, basic_matrices):
    """
    Compute double convolutions of edge shape functions

    Inputs:
        comp_struct : dict
            Dictionary containing model parameters
        basic_matrices : dict
            Dictionary containing basic matrices

    Outputs:
        basic_matrices : dict
            Updated with edge convolution matrices
    """
    fname = os.path.join(comp_struct['Config']['root_path'], 'output', 'ConvEdgeMatrices.mat')

    if os.path.exists(fname):
        # Load cached results
        data = np.load(fname, allow_pickle=True).item()
        basic_matrices['NENENEConvMatrixInt'] = data['NENENEConvMatrixInt']
    else:
        n_edge_nodes = comp_struct['Advanced']['NEdge_nodes']

        # Allocate memory: (node_a, node_b, node_c, L1_pow, L2_pow)
        nenene_conv = np.zeros((n_edge_nodes, n_edge_nodes, n_edge_nodes, 4, 4))

        # Integration matrix (4×4 submatrix)
        lint_matrix = basic_matrices['LEdgeIntMatrix9'][:4, :4]
        lint_matrix_nenene = np.zeros_like(nenene_conv)

        print('conv')

        # Compute triple convolutions
        for bb in range(n_edge_nodes):
            for cc in range(n_edge_nodes):
                ne_b = basic_matrices['NLEdgeMatrix'][bb, :, :]
                ne_c = basic_matrices['NLEdgeMatrix'][cc, :, :]
                conv_bc = convolve(ne_b, ne_c, mode='full')

                for aa in range(n_edge_nodes):
                    ne_a = basic_matrices['NLEdgeMatrix'][aa, :, :]
                    nenene_conv[aa, bb, cc, :, :] = convolve(ne_a, conv_bc, mode='full')
                    lint_matrix_nenene[aa, bb, cc, :, :] = lint_matrix

        # Integrate
        print('conv_int')
        nenene_conv_unsummed = nenene_conv * lint_matrix_nenene

        # Sum over L dimensions (last two axes)
        basic_matrices['NENENEConvMatrixInt'] = np.sum(nenene_conv_unsummed, axis=4)
        basic_matrices['NENENEConvMatrixInt'] = np.sum(basic_matrices['NENENEConvMatrixInt'], axis=3)

        # Cache results
        np.save(fname, {'NENENEConvMatrixInt': basic_matrices['NENENEConvMatrixInt']})

    return basic_matrices