import numpy as np
from scipy.sparse import csr_matrix


def km_matrix_fluid(basic_matrices, comp_struct, fem_matrices, ii_d, ii_el, el_phys_props, tri_props):
    """
    Part of the toolbox for solving problems of wave propagation
    in arbitrary anisotropic inhomogeneous waveguides.

    KM_el_matrix_fluid function computes the matrices, which indicates the expansion
    and the coefficients of the expansion of the integrands for K_i and M matrices' elements
    into the interpolating functions L_j (L1, L2, L3).
    For ideal inviscid fluid.

    Current implementation is for cubic elements.
    NB!!! This implementation is specific for FE computations

    [T.Zharnikov, SMR v0.3_08.2014]
    Converted to Python 2025

    Args:
        basic_matrices: Structure containing basic matrices
        comp_struct: Structure containing model parameters
        fem_matrices: FEM matrices structure
        ii_d: Domain index
        ii_el: Element index
        el_phys_props: Element physical properties
        tri_props: Triangle properties

    Returns:
        el_matrices: Element matrices dictionary
    """

    # Identify number of nodes
    n_nodes = comp_struct['Advanced']['N_nodes']

    # Get the triangle area
    delta = tri_props['delta'][0]

    # Allocate memory for the matrices
    dvar_num = comp_struct['Data']['DVarNum'][ii_d]
    matrix_size = dvar_num * n_nodes

    # Compute element matrices using the method specific to the domain
    el_matrices = comp_struct['Methods']['KM_el_matrix'][ii_d](
        basic_matrices, comp_struct, fem_matrices, ii_d, ii_el, el_phys_props, tri_props
    )

    # Mass matrix term - the kinetic energy
    el_matrices['MMatrix'] = delta * el_matrices['NtRho2LambdaNMatrix']

    # Stiffness matrix terms - the potential energy
    el_matrices['K1Matrix'] = (1.0 / delta) * el_matrices['B1tCB1Matrix']
    el_matrices['K2Matrix'] = (el_matrices['B1tCB2Matrix'] - el_matrices['B2tCB1Matrix'])
    el_matrices['K3Matrix'] = delta * el_matrices['B2tCB2Matrix']

    return el_matrices
