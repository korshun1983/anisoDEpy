import numpy as np
from scipy.sparse import csr_matrix


def km_matrix_HTTI(basic_matrices, comp_struct, fem_matrices, ii_d, ii_el, el_phys_props, tri_props):
    """
    Part of the toolbox for solving problems of wave propagation
    in arbitrary anisotropic inhomogeneous waveguides.

    KM_el_matrix_HTTI function computes the matrices for HTTI solid.
    [T.Zharnikov SMR v0.3_08.2014]
    Converted to Python 2025
    """

    # Identify number of nodes
    n_nodes = comp_struct['Advanced']['N_nodes']

    # Get the triangle area
    delta = tri_props['delta'][0]

    # Allocate memory for the matrices
    dvar_num = comp_struct['Data']['DVarNum'][ii_d]
    matrix_size = dvar_num * n_nodes

    # Compute element matrices
    el_matrices = comp_struct['Methods']['KM_el_matrix'][ii_d](
        basic_matrices, comp_struct, fem_matrices, ii_d, ii_el, el_phys_props, tri_props
    )

    # Mass matrix term - the kinetic energy (negative sign convention)
    el_matrices['MMatrix'] = -delta * el_matrices['NtRhoNMatrix']

    # Stiffness matrix terms - the potential energy (negative sign convention)
    el_matrices['K1Matrix'] = -(1.0 / delta) * el_matrices['B1tCB1Matrix']
    el_matrices['K2Matrix'] = -(el_matrices['B1tCB2Matrix'] - el_matrices['B2tCB1Matrix'])
    el_matrices['K3Matrix'] = -delta * el_matrices['B2tCB2Matrix']

    return el_matrices
