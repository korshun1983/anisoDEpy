import numpy as np


def remove_redundant_variables_SAFE(comp_struct, basic_matrices, fem_matrices, full_matrices):
    """
    Part of the toolbox for solving problems of wave propagation
    in arbitrary anisotropic inhomogeneous waveguides.

    Remove redundant variables from full matrices after assembly.
    [T.Zharnikov SMR v0.3_09.2014]
    Converted to Python 2025
    """

    full_var_arr = np.arange(basic_matrices['Pos'][comp_struct['Data']['N_domain'] - 1][1])
    full_remove_var_arr = np.array([], dtype=int)

    # Assemble array of variables to remove
    for ii_d in range(comp_struct['Data']['N_domain']):
        if len(fem_matrices['DNodesRem'][ii_d]) > 0:
            # Build variable removal positions
            var_arr = np.arange(comp_struct['Data']['DVarNum'][ii_d]) + 1
            size_var_arr = len(fem_matrices['DNodesRem'][ii_d])
            large_var_arr = np.tile(var_arr, (size_var_arr, 1)).T

            remove_nodes_pos = np.where(np.isin(fem_matrices['DNodes'][ii_d], fem_matrices['DNodesRem'][ii_d]))[0]
            dremove_var_pos = comp_struct['Data']['DVarNum'][ii_d] * remove_nodes_pos
            dremove_var_pos = np.tile(dremove_var_pos, (comp_struct['Data']['DVarNum'][ii_d], 1)) + large_var_arr
            dremove_var_pos = basic_matrices['Pos'][ii_d][0] - 1 + dremove_var_pos.reshape(-1, order='F')

            full_remove_var_arr = np.concatenate([full_remove_var_arr, dremove_var_pos])

    # Find variables to keep
    full_keep_var_arr = np.where(~np.isin(full_var_arr, full_remove_var_arr))[0]

    # Reduce matrices by removing rows/columns
    full_matrices['K1Matrix'] = full_matrices['K1Matrix'][full_keep_var_arr, :][:, full_keep_var_arr]
    full_matrices['K2Matrix'] = full_matrices['K2Matrix'][full_keep_var_arr, :][:, full_keep_var_arr]
    full_matrices['K3Matrix'] = full_matrices['K3Matrix'][full_keep_var_arr, :][:, full_keep_var_arr]
    full_matrices['MMatrix'] = full_matrices['MMatrix'][full_keep_var_arr, :][:, full_keep_var_arr]
    full_matrices['PMatrix'] = full_matrices['PMatrix'][full_keep_var_arr, :][:, full_keep_var_arr]

    return fem_matrices, full_matrices
