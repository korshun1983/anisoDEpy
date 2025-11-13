import numpy as np


def km_el_matrix_fluid(basic_matrices, comp_struct, fem_matrices, ii_d, ii_el, el_phys_props, tri_props):
    """
    Part of the toolbox for solving problems of wave propagation
    in arbitrary anisotropic inhomogeneous waveguides.

    KM_el_matrix_fluid computes element-level matrices for fluid.
    [T.Zharnikov, SMR v0.3_08.2014]
    Converted to Python 2025
    """

    n_nodes = comp_struct['Advanced']['N_nodes']
    dvar_num = comp_struct['Data']['DVarNum'][ii_d]
    matrix_size = dvar_num * n_nodes

    # Initialize matrices
    el_matrices = {
        'B1tCB1Matrix': np.zeros((matrix_size, matrix_size)),
        'B1tCB2Matrix': np.zeros((matrix_size, matrix_size)),
        'B2tCB1Matrix': np.zeros((matrix_size, matrix_size)),
        'B2tCB2Matrix': np.zeros((matrix_size, matrix_size)),
        'NtRho2LambdaNMatrix': np.zeros((matrix_size, matrix_size))
    }

    # Prepare properties
    rho = el_phys_props['RhoVec']
    rho2_lambda = el_phys_props['Rho2LambdaVec']
    dxL = tri_props['dxL']
    dyL = tri_props['dyL']

    # Create large arrays for vectorized operations
    # These replace MATLAB's automatic array expansion
    rho2_lambda_large = np.zeros_like(basic_matrices['NNNConvMatrixInt'])
    rho_large = np.zeros_like(basic_matrices['NNNConvMatrixInt'])
    rho_dxL_dxL_large = np.zeros_like(basic_matrices['dNNdNConvMatrixInt'])

    for kk in range(n_nodes):
        rho2_lambda_large[:, kk, :] = rho2_lambda[kk]
        rho_large[:, kk, :] = rho[kk]
        for ii in range(3):
            for jj in range(3):
                rho_dxL_dxL_large[ii, jj, :, kk, :] = rho[kk] * (dxL[ii] * dxL[jj] + dyL[ii] * dyL[jj])

    # Compute matrices using element-wise multiplication and summation
    # These operations replace MATLAB's convn and automatic broadcasting
    nt_rho2lambda_unsummed = rho2_lambda_large * basic_matrices['NNNConvMatrixInt']
    el_matrices['NtRho2LambdaNMatrix'] = np.sum(nt_rho2lambda_unsummed, axis=1)

    b2tcb2_unsummed = rho_large * basic_matrices['NNNConvMatrixInt']
    el_matrices['B2tCB2Matrix'] = np.sum(b2tcb2_unsummed, axis=1)

    # K2 matrices are zero for fluid (no coupling)
    el_matrices['B1tCB2Matrix'] = np.zeros((matrix_size, matrix_size))
    el_matrices['B2tCB1Matrix'] = np.zeros((matrix_size, matrix_size))

    # Compute B1tCB1Matrix (most complex)
    b1tcb1_unsummed = rho_dxL_dxL_large * basic_matrices['dNNdNConvMatrixInt']
    el_matrices['B1tCB1Matrix'] = np.sum(b1tcb1_unsummed, axis=(0, 1, 3))

    return el_matrices
