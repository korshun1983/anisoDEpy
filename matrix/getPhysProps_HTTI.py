import numpy as np


def get_phys_props_HTTI(comp_struct, basic_matrices, fem_matrices, ii_d, ii_el):
    """
    Construct interpolation for elastic moduli and density for HTTI solid.
    [T.Zharnikov, SMR v0.3_08.2014]
    Converted to Python 2025

    Args:
        comp_struct: Model parameters
        basic_matrices: Basic matrices
        fem_matrices: FEM matrices
        ii_d: Domain index
        ii_el: Element index

    Returns:
        el_phys_props: Element physical properties
    """

    n_nodes = comp_struct['Advanced']['N_nodes']

    # Allocate memory
    el_phys_props = {
        'CijMatrix': np.zeros((n_nodes, 6, 6)),
        'RhoVec': np.zeros(n_nodes)
    }

    # Retrieve properties (convert units)
    cij_6x6 = fem_matrices['PhysProp'][ii_d]['c_ij'] * 1e9
    rho_value = fem_matrices['PhysProp'][ii_d]['rho'] * 1e3

    # Fill vectors
    el_phys_props['RhoVec'].fill(rho_value)
    el_phys_props['CijMatrix'][:, :, :] = cij_6x6

    return el_phys_props
