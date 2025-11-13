import numpy as np


def km_el_matrix_HTTI_PML(basic_matrices, comp_struct, fem_matrices, ii_d, ii_el, el_phys_props, tri_props):
    """
    KM_el_matrix_HTTI with Perfectly Matched Layer (PML)
    [T.Zharnikov, SMR v0.3_08.2014]
    Converted to Python 2025

    PML: Complex coordinate stretching for absorbing boundaries
    """

    n_nodes = comp_struct['Advanced']['N_nodes']
    dvar_num = comp_struct['Data']['DVarNum'][ii_d]
    matrix_size = dvar_num * n_nodes

    # Initialize matrices (complex)
    el_matrices = {
        'B1tCB1Matrix': np.zeros((matrix_size, matrix_size), dtype=complex),
        'B1tCB2Matrix': np.zeros((matrix_size, matrix_size), dtype=complex),
        'B2tCB1Matrix': np.zeros((matrix_size, matrix_size), dtype=complex),
        'B2tCB2Matrix': np.zeros((matrix_size, matrix_size), dtype=complex),
        'NtRhoNMatrix': np.zeros((matrix_size, matrix_size), dtype=complex)
    }

    # Prepare properties
    rho = el_phys_props['RhoVec']
    cij_matrix = el_phys_props['CijMatrix']
    dxL = tri_props['dxL']
    dyL = tri_props['dyL']

    Lx_base = basic_matrices['Lx']
    Ly_base = basic_matrices['Ly']
    Lz = basic_matrices['Lz']

    ones_matrix = np.ones(n_nodes)
    id_matrix = np.eye(dvar_num)

    # PML parameters
    tri_nodes = fem_matrices['DElements'][ii_d][:10, ii_el]
    f_peak = comp_struct['f_grid'][comp_struct['if_grid']] * comp_struct['Misc']['F_conv']
    w_peak = 2 * np.pi * f_peak
    cone = 1j

    r_zv = comp_struct['Model']['DomainRx'][-2]
    L_ext_layer = comp_struct['Model']['DomainRx'][-1] - r_zv

    # Create large arrays
    rho_large = np.zeros([matrix_size, n_nodes, matrix_size], dtype=complex)
    lx_cij_lx_large = np.zeros([3, 3, matrix_size, n_nodes, matrix_size], dtype=complex)
    lx_cij_lz_large = np.zeros([3, matrix_size, n_nodes, matrix_size], dtype=complex)
    lz_cij_lx_large = np.zeros([3, matrix_size, n_nodes, matrix_size], dtype=complex)
    lz_cij_lz_large = np.zeros([matrix_size, n_nodes, matrix_size], dtype=complex)

    for kk in range(n_nodes):
        # PML coordinate stretching
        Lx = Lx_base.copy()
        Ly = Ly_base.copy()
        var_df_xy = 1.0

        if ii_d == comp_struct['Data']['N_domain'] - 1:
            xy_node = fem_matrices['MeshNodes'][:2, tri_nodes[kk]]

            if comp_struct['Model']['PML_method'] == 1:  # Circular PML
                r_node = np.sqrt(xy_node[0] ** 2 + xy_node[1] ** 2)

                if r_node > r_zv:
                    ivar = comp_struct['Model']['PML_degree']
                    sigma_r = ((r_node - r_zv) / L_ext_layer) ** ivar
                    gamma_r = cone * comp_struct['Model']['PML_factor'] * sigma_r
                    r_tilde = r_node - cone * (comp_struct['Model']['PML_factor'] / (ivar + 1)) * (
                                (r_node - r_zv) / L_ext_layer) ** (ivar + 1)

                    var_xy = xy_node[0] ** 2 / (gamma_r * r_node ** 2) + xy_node[1] ** 2 / (r_tilde * r_node)
                    var_yx = xy_node[1] ** 2 / (gamma_r * r_node ** 2) + xy_node[0] ** 2 / (r_tilde * r_node)
                    var_all = (1.0 / (gamma_r * r_node ** 2) - 1.0 / (r_tilde * r_node)) * xy_node[0] * xy_node[1]

                    Lx = var_xy * Lx_base + var_all * Ly_base
                    Ly = var_all * Lx_base + var_yx * Ly_base

                    var_df_xy = gamma_r * r_tilde / r_node

            elif comp_struct['Model']['PML_method'] == 2:  # Rectangular PML
                ivar = comp_struct['Model']['PML_degree']
                x_zv = comp_struct['Model']['DomainRx'][-2]
                Lx_ext_layer = comp_struct['Model']['DomainRx'][-1] - x_zv
                y_zv = comp_struct['Model']['DomainRy'][-2]
                Ly_ext_layer = comp_struct['Model']['DomainRy'][-1] - y_zv

                gamma_x = 1.0
                if abs(xy_node[0]) > x_zv:
                    sigma_x = ((abs(xy_node[0]) - x_zv) / Lx_ext_layer) ** ivar
                    gamma_x = 1.0 - cone * comp_struct['Model']['PML_factor'] * sigma_x
                    Lx = (1.0 / gamma_x) * Lx_base

                gamma_y = 1.0
                if abs(xy_node[1]) > y_zv:
                    sigma_y = ((abs(xy_node[1]) - y_zv) / Ly_ext_layer) ** ivar
                    gamma_y = 1.0 - cone * comp_struct['Model']['PML_factor'] * sigma_y
                    Ly = (1.0 / gamma_y) * Ly_base

                var_df_xy = gamma_x * gamma_y

        # Compute Cij products (complex)
        lx_cij_lx = np.conj(Lx.T) @ cij_matrix[kk, :, :] @ Lx
        lx_cij_ly = np.conj(Lx.T) @ cij_matrix[kk, :, :] @ Ly
        ly_cij_lx = np.conj(Ly.T) @ cij_matrix[kk, :, :] @ Lx
        ly_cij_ly = np.conj(Ly.T) @ cij_matrix[kk, :, :] @ Ly
        lx_cij_lz = np.conj(Lx.T) @ cij_matrix[kk, :, :] @ Lz
        lz_cij_lx = np.conj(Lz.T) @ cij_matrix[kk, :, :] @ Lx
        ly_cij_lz = np.conj(Ly.T) @ cij_matrix[kk, :, :] @ Lz
        lz_cij_ly = np.conj(Lz.T) @ cij_matrix[kk, :, :] @ Ly
        lz_cij_lz = np.conj(Lz.T) @ cij_matrix[kk, :, :] @ Lz

        # Apply PML scaling
        lx_cij_lx *= var_df_xy
        lx_cij_ly *= var_df_xy
        ly_cij_lx *= var_df_xy
        ly_cij_ly *= var_df_xy
        lx_cij_lz *= var_df_xy
        lz_cij_lx *= var_df_xy
        ly_cij_lz *= var_df_xy
        lz_cij_ly *= var_df_xy
        lz_cij_lz *= var_df_xy

        # Fill large arrays
        rho_large[:, kk, :] = np.kron(ones_matrix, rho[kk] * id_matrix).reshape(matrix_size, 1, matrix_size)
        rho_large[:, kk, :] *= var_df_xy

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

    # Use precomputed convolution matrices
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