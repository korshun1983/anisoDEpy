# File: assemble_full_matrices_ff_ss_safe_cubic.py
import numpy as np
from scipy.sparse import csr_matrix, block_diag, find as sparse_find


def assemble_full_matrices_ff_ss_safe_cubic(comp_struct, basic_matrices, fem_matrices, full_matrices, ii_int, ii_d1,
                                            ii_d2):
    """
    Assemble full matrices for fluid-solid boundary conditions (cubic elements).

    Part of the toolbox for solving problems of wave propagation
    in arbitrary anisotropic inhomogeneous waveguides.
    For details see User manual and comments to the main script gen_aniso.m

    Last Modified by Timur Zharnikov SMR v0.3_09.2014
    Converted to Python 2025

    Args:
        comp_struct: Structure containing model parameters (dict)
        basic_matrices: Structure containing basic matrices (dict)
        fem_matrices: FEM matrices structure (dict)
        full_matrices: Full matrices structure (dict)
        ii_int: Interface index
        ii_d1: First domain index
        ii_d2: Second domain index

    Returns:
        fem_matrices: Updated FEM matrices structure (dict)
        full_matrices: Updated full matrices structure (dict)
    """

    # Assembling full matrices by adding the mass and stiffness matrices of the
    # next block and by inserting the matrices responsible for boundary conditions

    # Add matrices for the next block using block diagonal assembly
    full_matrices['K1Matrix'] = block_diag(full_matrices['K1Matrix'], fem_matrices['K1Matrix_d'][ii_d2])
    full_matrices['K2Matrix'] = block_diag(full_matrices['K2Matrix'], fem_matrices['K2Matrix_d'][ii_d2])
    full_matrices['K3Matrix'] = block_diag(full_matrices['K3Matrix'], fem_matrices['K3Matrix_d'][ii_d2])
    full_matrices['MMatrix'] = block_diag(full_matrices['MMatrix'], fem_matrices['MMatrix_d'][ii_d2])
    full_matrices['PMatrix'] = block_diag(full_matrices['PMatrix'], fem_matrices['PMatrix_d'][ii_d2])

    # Adjust the list of variables (remove duplicates, if any, etc.)
    # and the positions of variables in the full vector of variables

    # Combine and get unique nodes to remove
    fem_matrices['DNodesRem'][ii_d2] = np.unique(np.concatenate([
        fem_matrices['DNodesRem'][ii_d2],
        fem_matrices['BNodesFull'][ii_int]
    ]))

    # Find nodes to keep (complement of nodes to remove)
    remove_nodes_pos = np.isin(fem_matrices['DNodes'][ii_d2], fem_matrices['DNodesRem'][ii_d2])
    fem_matrices['DNodesComp'][ii_d2] = fem_matrices['DNodes'][ii_d2][~remove_nodes_pos]

    # Find positions of rows corresponding to nodes that will be removed
    # NB! Nodes in DNodes and BNodesFull are ordered!
    # This is an important condition for this script to work.

    b_nodes_full = fem_matrices['BNodesFull'][ii_int]
    b_nodes_full_size = len(b_nodes_full)

    add_var_pos = []
    remove_var_pos = []

    for ii_n in range(b_nodes_full_size):
        # Find position of node in each domain
        d1_node_pos = np.where(fem_matrices['DNodes'][ii_d1] == b_nodes_full[ii_n])[0]
        d2_node_pos = np.where(fem_matrices['DNodes'][ii_d2] == b_nodes_full[ii_n])[0]

        # Handle potential multiple matches (should be one)
        if len(d1_node_pos) > 0 and len(d2_node_pos) > 0:
            d1_node_pos = d1_node_pos[0]
            d2_node_pos = d2_node_pos[0]

            # Calculate variable positions (convert to 0-indexed)
            dvar_num_1 = comp_struct['Data']['DVarNum'][ii_d1]
            dvar_num_2 = comp_struct['Data']['DVarNum'][ii_d2]

            # Add positions (source)
            start_idx = basic_matrices['Pos'][ii_d1][0] - 1  # Convert to 0-indexed
            var_range = np.arange(dvar_num_1 * d1_node_pos, dvar_num_1 * (d1_node_pos + 1))
            add_var_pos.extend((start_idx + var_range).tolist())

            # Remove positions (target)
            start_idx = basic_matrices['Pos'][ii_d2][0] - 1  # Convert to 0-indexed
            var_range = np.arange(dvar_num_2 * d2_node_pos, dvar_num_2 * (d2_node_pos + 1))
            remove_var_pos.extend((start_idx + var_range).tolist())

    # Identify nodes and variables that will be removed from the full matrices
    # and which values should be taken from other nodes (duplicate nodes) and variables
    # after computing the eigenvectors of the reduced matrices

    fem_matrices['DTakeFromVarPos'][ii_d2] = np.unique(np.concatenate([
        fem_matrices['DTakeFromVarPos'][ii_d2],
        np.array(add_var_pos)
    ]))
    fem_matrices['DPutToVarPos'][ii_d2] = np.unique(np.concatenate([
        fem_matrices['DPutToVarPos'][ii_d2],
        np.array(remove_var_pos)
    ]))

    # Convert to numpy arrays for indexing
    add_var_pos = np.array(add_var_pos, dtype=int)
    remove_var_pos = np.array(remove_var_pos, dtype=int)

    # Sum rows and columns corresponding to coincident nodes in both domains
    for matrix_name in ['K1Matrix', 'K2Matrix', 'K3Matrix', 'MMatrix', 'PMatrix']:
        # Get the matrix
        mat = full_matrices[matrix_name]

        # Sum rows: add rows from remove_var_pos to add_var_pos
        for add_idx, remove_idx in zip(add_var_pos, remove_var_pos):
            mat[add_idx, :] += mat[remove_idx, :]
            mat[:, add_idx] += mat[:, remove_idx]

        # Zero out rows and columns that will be removed
        mat[remove_var_pos, :] = 0
        mat[:, remove_var_pos] = 0

        # Update the matrix in the dictionary
        full_matrices[matrix_name] = mat

    return fem_matrices, full_matrices