import numpy as np
from scipy.sparse import csr_matrix, coo_matrix


def ic_matrices_fluid_HTTI_safe_cubic(comp_struct, basic_matrices, fem_matrices, ii_int, ii_d1, ii_d2):
    """
    Main function for fluid-solid interface matrices.
    [T.Zharnikov SMR v0.3_09.2014]
    Converted to Python 2025
    """

    # Get domain sizes
    d1_nodes_num = fem_matrices['DNodes'][ii_d1].size
    d1_fe_var_num = comp_struct['Data']['DVarNum'][ii_d1] * d1_nodes_num
    d2_nodes_num = fem_matrices['DNodes'][ii_d2].size
    d2_fe_var_num = comp_struct['Data']['DVarNum'][ii_d2] * d2_nodes_num

    # Create zero sparse matrices for fluid-solid boundary condition
    fem_matrices['ZeroD12'] = csr_matrix((d1_fe_var_num, d2_fe_var_num))
    fem_matrices['ZeroD21'] = csr_matrix((d2_fe_var_num, d1_fe_var_num))

    # Identify domain types and put fluid domain first
    if comp_struct['Model']['DomainType'][ii_d1] == 'fluid':
        ii_df = ii_d1
        ii_ds = ii_d2
    else:  # HTTI
        ii_df = ii_d2
        ii_ds = ii_d1

    # Get boundary edges
    boundary_edges_mask = np.isin(fem_matrices['BoundaryEdges'][2, :], ii_int)
    belements = fem_matrices['BoundaryEdges'][:2, boundary_edges_mask]
    nb_elements = belements.shape[1]

    b_nodes_full = np.array([], dtype=int)

    # Process boundary edges
    for ii_ed in range(nb_elements):
        edge_nodes = np.zeros(4, dtype=int)

        # Adjust edge orientation based on domain order
        if comp_struct['Model']['DomainType'][ii_d1] == 'fluid':
            edge_nodes[0] = belements[0, ii_ed]
            edge_nodes[1] = belements[1, ii_ed]
        else:  # HTTI - reverse orientation
            edge_nodes[0] = belements[1, ii_ed]
            edge_nodes[1] = belements[0, ii_ed]

        # Find adjacent elements in both domains
        fnodes_f = np.isin(fem_matrices['DElements'][ii_df][:10, :], edge_nodes[:2])
        nnodes_f = np.sum(fnodes_f, axis=0)
        df_edge_el = np.argmax(nnodes_f)

        fnodes_s = np.isin(fem_matrices['DElements'][ii_ds][:10, :], edge_nodes[:2])
        nnodes_s = np.sum(fnodes_s, axis=0)
        ds_edge_el = np.argmax(nnodes_s)

        # Read triangle nodes
        tri_nodes_f = fem_matrices['DElements'][ii_df][:10, df_edge_el]
        tri_nodes_s = fem_matrices['DElements'][ii_ds][:10, ds_edge_el]

        # Find mid-side nodes for fluid domain
        edge_nodes_pos_df = np.zeros(4, dtype=int)
        edge_nodes_pos_df[0] = np.where(np.isin(tri_nodes_f, edge_nodes[0]))[0][0]
        edge_nodes_pos_df[1] = np.where(np.isin(tri_nodes_f, edge_nodes[1]))[0][0]

        if edge_nodes_pos_df[1] > (edge_nodes_pos_df[0] % 3):
            edge_nodes_pos_df[2] = (edge_nodes_pos_df[0] + 1) * 2
            edge_nodes_pos_df[3] = (edge_nodes_pos_df[0] + 1) * 2 + 1
        else:
            edge_nodes_pos_df[2] = (edge_nodes_pos_df[1] + 1) * 2 + 1
            edge_nodes_pos_df[3] = (edge_nodes_pos_df[1] + 1) * 2

        edge_nodes[2] = tri_nodes_f[edge_nodes_pos_df[2]]
        edge_nodes[3] = tri_nodes_f[edge_nodes_pos_df[3]]

        # Find corresponding nodes in solid domain
        edge_nodes_pos_ds = np.zeros(4, dtype=int)
        edge_nodes_pos_ds[0] = np.where(np.isin(tri_nodes_s, edge_nodes[0]))[0][0]
        edge_nodes_pos_ds[1] = np.where(np.isin(tri_nodes_s, edge_nodes[1]))[0][0]
        edge_nodes_pos_ds[2] = np.where(np.isin(tri_nodes_s, edge_nodes[2]))[0][0]
        edge_nodes_pos_ds[3] = np.where(np.isin(tri_nodes_s, edge_nodes[3]))[0][0]

        b_nodes_full = np.concatenate([b_nodes_full, edge_nodes])

        # Get edge coordinates
        edge_nodes_x = fem_matrices['MeshNodes'][0, edge_nodes[:2]]
        edge_nodes_y = fem_matrices['MeshNodes'][1, edge_nodes[:2]]

        # Read physical properties
        el_phys_props = {
            'DfEl': comp_struct['Methods']['getPhysProps'][ii_df](
                comp_struct, basic_matrices, fem_matrices, ii_df, df_edge_el
            ),
            'DsEl': comp_struct['Methods']['getPhysProps'][ii_ds](
                comp_struct, basic_matrices, fem_matrices, ii_ds, ds_edge_el
            )
        }

        # Compute edge properties
        edge_props = {
            'DxEdge': edge_nodes_x[1] - edge_nodes_x[0],
            'DyEdge': edge_nodes_y[1] - edge_nodes_y[0]
        }
        edge_props['Dl'] = np.sqrt(edge_props['DxEdge'] ** 2 + edge_props['DyEdge'] ** 2)

        # Compute normal vector
        normal_xy = np.cross(
            [edge_props['DxEdge'] / edge_props['Dl'],
             edge_props['DyEdge'] / edge_props['Dl'], 0],
            [0, 0, -1]
        )
        edge_props['Normal'] = normal_xy[:2]  # Only x,y components

        # Compute interface matrices
        bmatrix_dfs_edge, bmatrix_dsf_edge = comp_struct['Methods']['IC_matrix'][ii_int](
            basic_matrices, comp_struct, ii_int, ii_df, ii_ds,
            el_phys_props, edge_props, edge_nodes_pos
        )

        # Get variable positions
        dvar_num_df = comp_struct['Data']['DVarNum'][ii_df]
        dvar_num_ds = comp_struct['Data']['DVarNum'][ii_ds]

        var_row_arr = np.zeros(dvar_num_df * 4, dtype=int)
        var_col_arr = np.zeros(dvar_num_ds * 4, dtype=int)

        for ii_n in range(4):
            df_node_pos = np.where(fem_matrices['DNodes'][ii_df] == edge_nodes[ii_n])[0][0]
            ds_node_pos = np.where(fem_matrices['DNodes'][ii_ds] == edge_nodes[ii_n])[0][0]

            row_start = dvar_num_df * ii_n
            col_start = dvar_num_ds * ii_n

            var_row_arr[row_start:row_start + dvar_num_df] = \
                np.arange(dvar_num_df * df_node_pos, dvar_num_df * (df_node_pos + 1))
            var_col_arr[col_start:col_start + dvar_num_ds] = \
                np.arange(dvar_num_ds * ds_node_pos, dvar_num_ds * (ds_node_pos + 1))

        # Assemble sparse matrix
        bfe_var_num_fluid = dvar_num_df * len(fem_matrices['DNodes'][ii_df])
        bfe_var_num_solid = dvar_num_ds * len(fem_matrices['DNodes'][ii_ds])

        var_rows = np.tile(var_row_arr, dvar_num_ds * 4)
        var_cols = np.repeat(var_col_arr, dvar_num_df * 4)
        bmatrix_dfs_flat = bmatrix_dfs_edge.reshape(-1)

        sp_bmatrix_dfs = coo_matrix(
            (bmatrix_dfs_flat, (var_rows, var_cols)),
            shape=(bfe_var_num_fluid, bfe_var_num_solid)
        ).tocsr()

        if 'BMatrixDfs' not in locals():
            bmatrix_dfs = sp_bmatrix_dfs
        else:
            bmatrix_dfs += sp_bmatrix_dfs

    # Assemble full node list
    fem_matrices['BNodesFull'][ii_int] = np.unique(b_nodes_full)

    # Compute solid-fluid matrix (transpose)
    bmatrix_dsf = bmatrix_dfs.T

    # Assign blocks based on domain order
    if comp_struct['Model']['DomainType'][ii_d1] == 'fluid':
        fem_matrices['PMatrixD12'][ii_int] = bmatrix_dfs
        fem_matrices['PMatrixD21'][ii_int] = bmatrix_dsf
    else:  # HTTI
        fem_matrices['PMatrixD12'][ii_int] = -bmatrix_dsf
        fem_matrices['PMatrixD21'][ii_int] = -bmatrix_dfs

    return fem_matrices
