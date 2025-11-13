import numpy as np
from scipy.sparse import csr_matrix, coo_matrix


def matrices_parts_HTTI_sp_SAFE_cubic(comp_struct, basic_matrices, fem_matrices, ii_d):
    """
    Part of the toolbox for solving problems of wave propagation
    in arbitrary anisotropic inhomogeneous waveguides.

    MatricesParts_HTTI_sp_SAFE_cubic prepares matrix blocks for HTTI solid domain.
    [T.Zharnikov SMR v0.3_09.2014]
    Converted to Python 2025
    """

    # Get domain information
    n_domain_elements = fem_matrices['DElements'][ii_d].shape[1]
    d_nodes_num = fem_matrices['DNodes'][ii_d].size
    dvar_num = comp_struct['Data']['DVarNum'][ii_d]
    dfe_var_num = dvar_num * d_nodes_num

    # Create zero sparse matrices
    fem_matrices['K1Matrix_d'][ii_d] = csr_matrix((dfe_var_num, dfe_var_num))
    fem_matrices['K2Matrix_d'][ii_d] = csr_matrix((dfe_var_num, dfe_var_num))
    fem_matrices['B1tCB2Matrix_d'][ii_d] = csr_matrix((dfe_var_num, dfe_var_num))
    fem_matrices['B2tCB1Matrix_d'][ii_d] = csr_matrix((dfe_var_num, dfe_var_num))
    fem_matrices['K3Matrix_d'][ii_d] = csr_matrix((dfe_var_num, dfe_var_num))
    fem_matrices['MMatrix_d'][ii_d] = csr_matrix((dfe_var_num, dfe_var_num))
    fem_matrices['PMatrix_d'][ii_d] = csr_matrix((dfe_var_num, dfe_var_num))

    var_vec_arr = np.zeros(dvar_num * comp_struct['Advanced']['N_nodes'], dtype=int)

    # Process all elements in the domain
    for ii_el in range(n_domain_elements):
        # Read node numbers for this element
        tri_nodes = fem_matrices['DElements'][ii_d][:10, ii_el]

        # Identify positions of element nodes and variables
        for ii_n in range(10):
            dnode_pos = np.where(fem_matrices['DNodes'][ii_d] == tri_nodes[ii_n])[0]
            if len(dnode_pos) > 0:
                start_idx = dvar_num * ii_n
                var_range = np.arange(dvar_num * dnode_pos[0], dvar_num * (dnode_pos[0] + 1))
                var_vec_arr[start_idx:start_idx + dvar_num] = var_range

        # Read physical properties
        el_phys_props = comp_struct['Methods']['getPhysProps'][ii_d](
            comp_struct, basic_matrices, fem_matrices, ii_d, ii_el
        )

        # Read element properties
        tri_props = {
            'a': fem_matrices['DEMeshProps'][ii_d]['a'][:, ii_el],
            'b': fem_matrices['DEMeshProps'][ii_d]['b'][:, ii_el],
            'c': fem_matrices['DEMeshProps'][ii_d]['c'][:, ii_el],
            'delta': fem_matrices['DEMeshProps'][ii_d]['delta'][:, ii_el]
        }

        # Compute derivative vectors
        tri_props['dxL'] = 0.5 * tri_props['b']
        tri_props['dyL'] = 0.5 * tri_props['c']

        # Compute element matrices
        el_matrices = comp_struct['Methods']['KM_matrix'][ii_d](
            basic_matrices, comp_struct, fem_matrices, ii_d, ii_el, el_phys_props, tri_props
        )

        # Prepare sparse matrices for insertion
        # For HTTI, we have more variables per node (3 displacement components)
        var_rows = np.tile(var_vec_arr, 30)  # 30 = 3 var * 10 nodes
        var_cols = np.repeat(var_vec_arr, 30)

        # Create vectors of elements to insert
        str_k1 = el_matrices['K1Matrix'].reshape(-1)
        str_k2 = el_matrices['K2Matrix'].reshape(-1)
        str_b1tcb2 = el_matrices['B1tCB2Matrix'].reshape(-1)
        str_b2tcb1 = el_matrices['B2tCB1Matrix'].reshape(-1)
        str_k3 = el_matrices['K3Matrix'].reshape(-1)
        str_m = el_matrices['MMatrix'].reshape(-1)

        # Create and add sparse matrices
        sp_k1 = coo_matrix((str_k1, (var_rows, var_cols)), shape=(dfe_var_num, dfe_var_num)).tocsr()
        sp_k2 = coo_matrix((str_k2, (var_rows, var_cols)), shape=(dfe_var_num, dfe_var_num)).tocsr()
        sp_b1tcb2 = coo_matrix((str_b1tcb2, (var_rows, var_cols)), shape=(dfe_var_num, dfe_var_num)).tocsr()
        sp_b2tcb1 = coo_matrix((str_b2tcb1, (var_rows, var_cols)), shape=(dfe_var_num, dfe_var_num)).tocsr()
        sp_k3 = coo_matrix((str_k3, (var_rows, var_cols)), shape=(dfe_var_num, dfe_var_num)).tocsr()
        sp_m = coo_matrix((str_m, (var_rows, var_cols)), shape=(dfe_var_num, dfe_var_num)).tocsr()

        fem_matrices['K1Matrix_d'][ii_d] += sp_k1
        fem_matrices['K2Matrix_d'][ii_d] += sp_k2
        fem_matrices['B1tCB2Matrix_d'][ii_d] += sp_b1tcb2
        fem_matrices['B2tCB1Matrix_d'][ii_d] += sp_b2tcb1
        fem_matrices['K3Matrix_d'][ii_d] += sp_k3
        fem_matrices['MMatrix_d'][ii_d] += sp_m

    return fem_matrices