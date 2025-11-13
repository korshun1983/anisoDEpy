# File: assemble_full_matrices_fs_safe_cubic.py
import numpy as np
from scipy.sparse import csr_matrix, block_diag, vstack, hstack


def assemble_full_matrices_fs_safe_cubic(comp_struct, basic_matrices, fem_matrices, full_matrices, ii_int, ii_d1,
                                         ii_d2):
    """
    Assemble full matrices for fluid-solid interface conditions (cubic elements).

    Part of the toolbox for solving problems of wave propagation
    in arbitrary anisotropic inhomogeneous waveguides.

    Last Modified by Timur Zharnikov SMR v0.3_09.2014
    Converted to Python 2025

    Args:
        comp_struct: Structure containing model parameters (dict)
        basic_matrices: Structure containing basic matrices (dict)
        fem_matrices: FEM matrices structure (dict)
        full_matrices: Full matrices structure (dict)
        ii_int: Interface index
        ii_d1: First domain index (fluid)
        ii_d2: Second domain index (solid)

    Returns:
        fem_matrices: Updated FEM matrices structure (dict)
        full_matrices: Updated full matrices structure (dict)
    """

    # Assemble full matrices by adding mass and stiffness matrices
    # and inserting boundary condition matrices

    # Add matrices for the next block using block diagonal assembly
    full_matrices['K1Matrix'] = block_diag(full_matrices['K1Matrix'], fem_matrices['K1Matrix_d'][ii_d2])
    full_matrices['K2Matrix'] = block_diag(full_matrices['K2Matrix'], fem_matrices['K2Matrix_d'][ii_d2])
    full_matrices['K3Matrix'] = block_diag(full_matrices['K3Matrix'], fem_matrices['K3Matrix_d'][ii_d2])
    full_matrices['MMatrix'] = block_diag(full_matrices['MMatrix'], fem_matrices['MMatrix_d'][ii_d2])

    # Assemble PMatrix with off-diagonal blocks for fluid-solid coupling
    cur_full_mat_size = full_matrices['PMatrix'].shape
    cur_mat_d12_size = fem_matrices['PMatrixD12'][ii_int].shape
    cur_zero_mat12_size = (cur_full_mat_size[0] - cur_mat_d12_size[0], cur_mat_d12_size[1])
    zero_matrix12 = csr_matrix(cur_zero_mat12_size)
    insert_pmatrix_d12 = vstack([zero_matrix12, fem_matrices['PMatrixD12'][ii_int]])

    cur_mat_d21_size = fem_matrices['PMatrixD21'][ii_int].shape
    cur_zero_mat21_size = (cur_mat_d21_size[0], cur_full_mat_size[1] - cur_mat_d21_size[1])
    zero_matrix21 = csr_matrix(cur_zero_mat21_size)
    insert_pmatrix_d21 = hstack([zero_matrix21, fem_matrices['PMatrixD21'][ii_int]])

    # Stack horizontally and vertically to create full matrix
    top_block = hstack([full_matrices['PMatrix'], insert_pmatrix_d12])
    bottom_block = hstack([insert_pmatrix_d21, fem_matrices['PMatrix_d'][ii_d2]])
    full_matrices['PMatrix'] = vstack([top_block, bottom_block])

    # For fluid-solid interface, typically no nodes are removed
    # The interface conditions are handled through the coupling matrices

    # Update node lists
    if 'DNodesRem' not in fem_matrices:
        fem_matrices['DNodesRem'] = [np.array([]) for _ in range(10)]

    # Ensure DNodesRem[ii_d2] exists
    if ii_d2 >= len(fem_matrices['DNodesRem']):
        fem_matrices['DNodesRem'].extend([np.array([]) for _ in range(ii_d2 + 1 - len(fem_matrices['DNodesRem']))])

    fem_matrices['DNodesRem'][ii_d2] = np.unique(fem_matrices['DNodesRem'][ii_d2])
    remove_nodes_pos = np.isin(fem_matrices['DNodes'][ii_d2], fem_matrices['DNodesRem'][ii_d2])
    fem_matrices['DNodesComp'][ii_d2] = fem_matrices['DNodes'][ii_d2][~remove_nodes_pos]

    return fem_matrices, full_matrices