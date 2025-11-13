import numpy as np
from scipy.sparse import csr_matrix, coo_matrix


def matrices_parts_fluid_sp_SAFE_cubic(comp_struct, basic_matrices, fem_matrices, ii_d):
    """
    Part of the toolbox for solving problems of wave propagation
    in arbitrary anisotropic inhomogeneous waveguides.

    MatricesParts_fluid_sp_SAFE_cubic prepares matrix blocks for fluid domain.
    [T.Zharnikov SMR v0.3_09.2014]
    Converted to Python 2025
    """

    # Get domain information
    n_domain_elements = fem_matrices['DElements'][ii_d].shape[1]
    d_nodes_num = fem_matrices['DNodes'][ii_d].size
    dfe_var_num = d_nodes_num  # Fluid has 1 variable per node

    # Create zero sparse matrices
    fem_matrices['K1Matrix_d'][ii_d] = csr_matrix((dfe_var_num, dfe_var_num))
    fem_matrices['K2Matrix_d'][ii_d] = csr_matrix((dfe_var_num, dfe_var_num))
    fem_matrices['K3Matrix_d'][ii_d] = csr_matrix((dfe_var_num, dfe_var_num))
    fem_matrices['MMatrix_d'][ii_d] = csr_matrix((dfe_var_num, dfe_var_num))
    fem_matrices['PMatrix_d'][ii_d] = csr_matrix((dfe_var_num, dfe_var_num))

    var_vec_arr = np.zeros(10, dtype=int)

    # Process all elements in the domain
    for ii_el in range(n_domain_elements):
        # Read node numbers for this element
        tri_nodes = fem_matrices['DElements'][ii_d][:10, ii_el]

        # Identify positions of element nodes and variables
        for ii_n in range(10):
            dnode_pos = np.where(fem_matrices['DNodes'][ii_d] == tri_nodes[ii_n])[0]
            if len(dnode_pos) > 0:
                var_vec_arr[ii_n] = dnode_pos[0]

        # Read physical properties
        el_phys_props = comp_struct['Methods']['getPhysProps'][ii_d](
            comp_struct, basic_matrices, fem_matrices, ii_d, ii_el
        )

        # Read element properties (a, b, c, delta from Zienkiewicz)
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
        # Compute rows and columns for insertion
        var_rows = np.tile(var_vec_arr, 10)
        var_cols = np.repeat(var_vec_arr, 10)

        # Create vectors of elements to insert
        str_k1 = el_matrices['K1Matrix'].reshape(-1)
        str_k2 = el_matrices['K2Matrix'].reshape(-1)
        str_k3 = el_matrices['K3Matrix'].reshape(-1)
        str_m = el_matrices['MMatrix'].reshape(-1)

        # Create sparse matrices and add to domain matrices
        sp_k1 = coo_matrix((str_k1, (var_rows, var_cols)), shape=(dfe_var_num, dfe_var_num)).tocsr()
        sp_k2 = coo_matrix((str_k2, (var_rows, var_cols)), shape=(dfe_var_num, dfe_var_num)).tocsr()
        sp_k3 = coo_matrix((str_k3, (var_rows, var_cols)), shape=(dfe_var_num, dfe_var_num)).tocsr()
        sp_m = coo_matrix((str_m, (var_rows, var_cols)), shape=(dfe_var_num, dfe_var_num)).tocsr()

        fem_matrices['K1Matrix_d'][ii_d] += sp_k1
        fem_matrices['K2Matrix_d'][ii_d] += sp_k2
        fem_matrices['K3Matrix_d'][ii_d] += sp_k3
        fem_matrices['MMatrix_d'][ii_d] += sp_m

    return fem_matrices