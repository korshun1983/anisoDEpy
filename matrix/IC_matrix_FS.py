import numpy as np
from scipy.sparse import csr_matrix


def ic_matrix_fs(basic_matrices, comp_struct, ii_int, ii_df, ii_ds,
                 el_phys_props, edge_props, edge_nodes_pos):
    """
    Part of the toolbox for solving problems of wave propagation
    in arbitrary anisotropic inhomogeneous waveguides.

    IC_matrix_FS function computes the interface conditions' matrices,
    which enter into the FE formulation.
    They are obtained by multiplying the matrix of the coefficients of the expansion
    of the integrands for Nfl_t_rho_(nN) matrix elements into the interpolating functions
    L_j (L1, L2) defined on the edge onto the values of the integrals L*∫(1 - ξ)^a ξ^b.
    For HTTI solid - fluid contact.
    The K_i and M matrices are defined according to Bartoli_Marzani_DiScalea_Viola_JSoundVibration_v295_p685_2006.

    Current implementation is for cubic elements.
    NB!!! This implementation is specific for FE computations

    [T.Zharnikov, SMR v0.3_08.2014]
    Converted to Python 2025

    Args:
        basic_matrices: Structure containing basic matrices
        comp_struct: Structure containing model parameters
        ii_int: Interface index
        ii_df: Fluid domain index
        ii_ds: Solid domain index
        el_phys_props: Element physical properties
        edge_props: Edge properties
        edge_nodes_pos: Edge node positions

    Returns:
        bmatrix_dfs_edge: Fluid-to-solid boundary matrix
        bmatrix_dsf_edge: Solid-to-fluid boundary matrix
    """

    # Identify number of nodes
    n_edge_nodes = comp_struct['Advanced']['NEdge_nodes']

    # Get the length of the edge (the element of the boundary)
    dl = edge_props['Dl']

    # Allocate memory for the matrices
    # The implementation is for cubic elements, hence expansion is up to the 3rd power in L_i

    dvar_num_df = comp_struct['Data']['DVarNum'][ii_df]
    dvar_num_ds = comp_struct['Data']['DVarNum'][ii_ds]

    bmatrix_dfs_edge = np.zeros((dvar_num_df * n_edge_nodes, dvar_num_ds * n_edge_nodes))
    bmatrix_dsf_edge = np.zeros((dvar_num_ds * n_edge_nodes, dvar_num_df * n_edge_nodes))

    # Compute the elements of K_i and M matrices according to the above formulas
    el_matrices = comp_struct['Methods']['IC_el_matrix'][ii_int](
        basic_matrices, comp_struct, el_phys_props, edge_props, edge_nodes_pos
    )

    # Process all elements of the 2d matrices
    bmatrix_dfs_edge = dl * el_matrices['NfltRhonNMatrix']
    bmatrix_dsf_edge = dl * el_matrices['NfltRhonNMatrix'].T

    return bmatrix_dfs_edge, bmatrix_dsf_edge