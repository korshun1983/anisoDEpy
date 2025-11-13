# File 1: assemble_basic_matrices_sp_safe.py
import numpy as np


def assemble_basic_matrices_sp_safe(comp_struct):
    """
    Part of the toolbox for solving problems of wave propagation
    in arbitrary anisotropic inhomogeneous waveguides.

    Assemble basic matrices for SAFE (Spectral Analysis of Finite Elements)

    Inputs:
        comp_struct : dict
            Dictionary containing the parameters of the model

    Outputs:
        basic_matrices : dict
            Dictionary containing basic matrices (Lx, Ly, Lz, identity, etc.)
    """
    basic_matrices = {}

    # ===============================================================================
    # Preparing the matrices (Auld book / Bartoli et al. 2006)
    # ===============================================================================

    # Lx matrix
    basic_matrices['Lx'] = np.array([
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0]
    ], dtype=float)

    # Ly matrix
    basic_matrices['Ly'] = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
        [0, 0, 1],
        [0, 0, 0],
        [1, 0, 0]
    ], dtype=float)

    # Lz matrix
    basic_matrices['Lz'] = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 0]
    ], dtype=float)

    # Fluid versions
    basic_matrices['Lx_fluid'] = np.array([[1], [0], [0]], dtype=float)
    basic_matrices['Ly_fluid'] = np.array([[0], [1], [0]], dtype=float)
    basic_matrices['Lz_fluid'] = np.array([[0], [0], [1]], dtype=float)

    basic_matrices['E3'] = np.eye(3)

    # Compute integral matrices for L_i functions
    n_nodes = comp_struct['Advanced']['N_nodes']
    basic_matrices['LEdgeIntMatrix9'] = comp_struct['Methods']['L1L2_int_matrix'](n_nodes)
    basic_matrices['LIntMatrix9'] = comp_struct['Methods']['L1L2L3_int_matrix'](n_nodes)

    # Compute shape function expansions and derivatives
    basic_matrices['NLMatrix'], basic_matrices['NodeLCoord'] = \
        comp_struct['Methods']['NL_matrix']()
    basic_matrices['dNLMatrix'], basic_matrices['dNLMatrix_val'] = \
        comp_struct['Methods']['dNL_matrices'](basic_matrices)

    # Compute double convolutions
    basic_matrices = comp_struct['Methods']['ConvolveMatrices'](comp_struct, basic_matrices)

    # Compute edge shape function expansions
    basic_matrices['NLEdgeMatrix'] = comp_struct['Methods']['NLEdge_matrix']()

    # Compute edge convolutions
    basic_matrices = comp_struct['Methods']['ConvolveEdgeMatrices'](comp_struct, basic_matrices)

    return basic_matrices