# File: assemble_full_fe_matrices_sp_safe.py
import numpy as np
from scipy.sparse import csr_matrix, find as sparse_find


def assemble_full_fe_matrices_sp_safe(comp_struct, basic_matrices, fem_matrices):
    """
    Assemble full finite element matrices for SAFE computations.

    Currently this procedure is not used in the main workflow.
    It contains several possible leads to modify other procedures.

    Part of the toolbox for solving problems of wave propagation
    in arbitrary anisotropic inhomogeneous waveguides.

    Last Modified by Timur Zharnikov SMR v0.3_08.2014
    Converted to Python 2025

    Args:
        comp_struct: Structure containing model parameters (dict)
        basic_matrices: Structure containing basic matrices (dict)
        fem_matrices: FEM matrices structure (dict)

    Returns:
        full_matrices: Full matrices structure (dict)
    """

    # Get number of domains
    n_domains = comp_struct['Data']['N_domains']

    # Determine total number of variables
    # Assuming DVarNum is a list of [start, end] pairs for each domain
    total_vars = comp_struct['Data']['DVarNum'][n_domains - 1][1]

    # Initialize full matrices as sparse zero matrices
    full_matrices = {
        'K1Matrix': csr_matrix((total_vars, total_vars)),
        'K2Matrix': csr_matrix((total_vars, total_vars)),
        'K3Matrix': csr_matrix((total_vars, total_vars)),
        'MMatrix': csr_matrix((total_vars, total_vars)),
        'PMatrix': csr_matrix((total_vars, total_vars)),
        'ICBCMatrix': csr_matrix((total_vars, total_vars))
    }

    # Insert matrix blocks into full matrices
    for ii_d in range(n_domains):
        var_start = comp_struct['Data']['DVarNum'][ii_d][0] - 1  # 0-indexed

        # K1Matrix
        mat_d = fem_matrices['K1Matrix_d'][ii_d]
        rows, cols, vals = sparse_find(mat_d)
        rows += var_start
        cols += var_start
        full_matrices['K1Matrix'] = full_matrices['K1Matrix'] + csr_matrix((vals, (rows, cols)),
                                                                           shape=(total_vars, total_vars))

        # K2Matrix
        mat_d = fem_matrices['K2Matrix_d'][ii_d]
        rows, cols, vals = sparse_find(mat_d)
        rows += var_start
        cols += var_start
        full_matrices['K2Matrix'] = full_matrices['K2Matrix'] + csr_matrix((vals, (rows, cols)),
                                                                           shape=(total_vars, total_vars))

        # K3Matrix
        mat_d = fem_matrices['K3Matrix_d'][ii_d]
        rows, cols, vals = sparse_find(mat_d)
        rows += var_start
        cols += var_start
        full_matrices['K3Matrix'] = full_matrices['K3Matrix'] + csr_matrix((vals, (rows, cols)),
                                                                           shape=(total_vars, total_vars))

        # MMatrix
        mat_d = fem_matrices['MMatrix_d'][ii_d]
        rows, cols, vals = sparse_find(mat_d)
        rows += var_start
        cols += var_start
        full_matrices['MMatrix'] = full_matrices['MMatrix'] + csr_matrix((vals, (rows, cols)),
                                                                         shape=(total_vars, total_vars))

    # Insert interface and boundary condition matrix blocks
    full_matrices = comp_struct['Methods']['InsertBC'](full_matrices, basic_matrices, comp_struct)

    # Process interfaces
    for ii_int in range(n_domains - 1):
        ii_d1 = ii_int
        ii_d2 = ii_d1 + 1
        full_matrices = comp_struct['Methods']['InsertIC'](
            full_matrices, basic_matrices, comp_struct, ii_d1, ii_d2, ii_int
        )

    return full_matrices