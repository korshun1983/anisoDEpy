import numpy as np


def km_el_matrix_HTTI(basic_matrices, comp_struct, fem_matrices, ii_d, ii_el, el_phys_props, tri_props):
    """
    Part of the toolbox for solving problems of wave propagation
    in arbitrary anisotropic inhomogeneous waveguides.

    KM_el_matrix_HTTI computes element-level matrices for HTTI solid.
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
        'NtRhoNMatrix': np.zeros((matrix_size, matrix_size))
    }

    # Prepare properties
    rho = el_phys_props['RhoVec']
    cij_matrix = el_phys_props['CijMatrix']
    dxL = tri_props['dxL']
    dyL = tri_props['dyL']

    Lx = basic_matrices['Lx']
    Ly = basic_matrices['Ly']
    Lz = basic_matrices['Lz']

    ones_matrix = np.ones(n_nodes)
    id_matrix = np.eye(dvar_num)

    # Create large arrays
    rho_large = np.zeros([matrix_size, n_nodes, matrix_size])
    lx_cij_lx_large = np.zeros([3, 3, matrix_size, n_nodes, matrix_size])
    lx_cij_lz_large = np.zeros([3, matrix_size, n_nodes, matrix_size])
    lz_cij_lx_large = np.zeros([3, matrix_size, n_nodes, matrix_size])
    lz_cij_lz_large = np.zeros([matrix_size, n_nodes, matrix_size])

    for kk in range(n_nodes):
        # Compute Cij products
        lx_cij_lx = np.conj(Lx.T) @ cij_matrix[kk, :, :] @ Lx
        lx_cij_ly = np.conj(Lx.T) @ cij_matrix[kk, :, :] @ Ly
        ly_cij_lx = np.conj(Ly.T) @ cij_matrix[kk, :, :] @ Lx
        ly_cij_ly = np.conj(Ly.T) @ cij_matrix[kk, :, :] @ Ly
        lx_cij_lz = np.conj(Lx.T) @ cij_matrix[kk, :, :] @ Lz
        lz_cij_lx = np.conj(Lz.T) @ cij_matrix[kk, :, :] @ Lx
        ly_cij_lz = np.conj(Ly.T) @ cij_matrix[kk, :, :] @ Lz
        lz_cij_ly = np.conj(Lz.T) @ cij_matrix[kk, :, :] @ Ly
        lz_cij_lz = np.conj(Lz.T) @ cij_matrix[kk, :, :] @ Lz

        # Fill large arrays using Kronecker products
        rho_large[:, kk, :] = np.kron(ones_matrix, rho[kk] * id_matrix).reshape(matrix_size, 1, matrix_size)

        lz_cij_lz_large[:, kk, :] = np.kron(ones_matrix, lz_cij_lz)

        for ii in range(3):
            lx_cij_lz_dx = lx_cij_lz * dxL[ii] + ly_cij_lz * dyL[ii]
            lz_cij_lx_dx = lz_cij_lx * dxL[ii] + lz_cij_ly * dyL[ii]

            lx_cij_lz_large[ii, :, kk, :] = np.kron(ones_matrix, lx_cij_lz_dx)
            lz_cij_lx_large[ii, :, kk, :] = np.kron(ones_matrix, lz_cij_lx_dx)

            for jj in range(3):
                lx_cij_lx_dxdx = (lx_cij_lx * dxL[ii] * dxL[jj] +
                                  lx_cij_ly * dxL[ii] * dyL[jj] +
                                  ly_cij_lx * dyL[ii] * dxL[jj] +
                                  ly_cij_ly * dyL[ii] * dyL[jj])
                lx_cij_lx_large[ii, jj, :, kk, :] = np.kron(ones_matrix, lx_cij_lx_dxdx)

    # Use precomputed large matrices from basic_matrices
    nnn_conv_large = basic_matrices['NNNConvMatrixIntLarge'][ii_d]
    dnn_dn_conv_large = basic_matrices['dNNdNConvMatrixIntLarge'][ii_d]
    dnnn_conv_large = basic_matrices['dNNNConvMatrixIntLarge'][ii_d]
    nndn_conv_large = basic_matrices['NNdNConvMatrixIntLarge'][ii_d]

    # Compute final matrices
    nt_rho_n_unsummed = rho_large * nnn_conv_large
    el_matrices['NtRhoNMatrix'] = np.sum(nt_rho_n_unsummed, axis=1)

    b1tcb1_unsummed = lx_cij_lx_large * dnn_dn_conv_large
    el_matrices['B1tCB1Matrix'] = np.sum(b1tcb1_unsummed, axis=(0, 1, 3))

    b1tcb2_unsummed = lx_cij_lz_large * dnnn_conv_large
    el_matrices['B1tCB2Matrix'] = np.sum(b1tcb2_unsummed, axis=(0, 2))

    b2tcb1_unsummed = lz_cij_lx_large * nndn_conv_large
    el_matrices['B2tCB1Matrix'] = np.sum(b2tcb1_unsummed, axis=(0, 2))

    b2tcb2_unsummed = lz_cij_lz_large * nnn_conv_large
    el_matrices['B2tCB2Matrix'] = np.sum(b2tcb2_unsummed, axis=1)

    return el_matrices
