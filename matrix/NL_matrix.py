# File 7: NL_matrix.py
import numpy as np


def NL_matrix():
    """
    Prepare shape function expansion matrix for cubic triangular elements

    Outputs:
        nl_matrix : ndarray
            4D array (10 × 4 × 4 × 4) of expansion coefficients
        node_l_coord : ndarray
            Node coordinates in L-space (10 × 3)
    """
    # 10 nodes, expansion up to 3rd power
    nl_matrix = np.zeros((10, 4, 4, 4))

    # Node coordinates in L1, L2, L3 space
    node_l_coord = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [2 / 3, 1 / 3, 0.0],
        [1 / 3, 2 / 3, 0.0],
        [0.0, 2 / 3, 1 / 3],
        [0.0, 1 / 3, 2 / 3],
        [2 / 3, 0.0, 1 / 3],
        [1 / 3, 0.0, 2 / 3],
        [1 / 3, 1 / 3, 1 / 3]
    ])

    # Corner nodes: N_i = 9/2*L_i^3 - 9/2*L_i^2 + L_i
    for i in range(3):
        nl_matrix[i, 1, 0, 0] = 1.0
        nl_matrix[i, 2, 0, 0] = -9 / 2
        nl_matrix[i, 3, 0, 0] = 9 / 2

    # Edge nodes 4-9
    nl_matrix[3, 1, 1, 0] = -9 / 2;
    nl_matrix[3, 2, 1, 0] = 27 / 2  # N4
    nl_matrix[4, 1, 1, 0] = -9 / 2;
    nl_matrix[4, 1, 2, 0] = 27 / 2  # N5
    nl_matrix[5, 0, 1, 1] = -9 / 2;
    nl_matrix[5, 0, 2, 1] = 27 / 2  # N6
    nl_matrix[6, 0, 1, 1] = -9 / 2;
    nl_matrix[6, 0, 1, 2] = 27 / 2  # N7
    nl_matrix[7, 1, 0, 1] = -9 / 2;
    nl_matrix[7, 1, 0, 2] = 27 / 2  # N8
    nl_matrix[8, 1, 0, 1] = -9 / 2;
    nl_matrix[8, 2, 0, 1] = 27 / 2  # N9

    # Internal node: N10 = 27*L1*L2*L3
    nl_matrix[9, 1, 1, 1] = 27.0

    return nl_matrix, node_l_coord