# File 8: NL_edge_matrix.py
import numpy as np


def NL_edge_matrix():
    """
    Prepare shape function expansion matrix for edge of cubic element

    Outputs:
        nl_edge_matrix : ndarray
            3D array (4 × 4 × 4) of expansion coefficients
    """
    # 4 nodes, expansion up to 3rd power
    nl_edge_matrix = np.zeros((4, 4, 4))

    # Corner nodes: N_i = 9/2*L_i^3 - 9/2*L_i^2 + L_i
    nl_edge_matrix[0, 1, 0] = 1.0;
    nl_edge_matrix[0, 2, 0] = -9 / 2;
    nl_edge_matrix[0, 3, 0] = 9 / 2
    nl_edge_matrix[1, 0, 1] = 1.0;
    nl_edge_matrix[1, 0, 2] = -9 / 2;
    nl_edge_matrix[1, 0, 3] = 9 / 2

    # Edge nodes: N3, N4
    nl_edge_matrix[2, 1, 1] = -9 / 2;
    nl_edge_matrix[2, 2, 1] = 27 / 2
    nl_edge_matrix[3, 1, 1] = -9 / 2;
    nl_edge_matrix[3, 1, 2] = 27 / 2

    return nl_edge_matrix