import numpy as np


def get_phys_props_fluid(comp_struct, basic_matrices, fem_matrices, ii_d, ii_el):
    """
    Construct interpolation for elastic moduli and density for fluid.
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
        'LambdaVec': np.zeros(n_nodes),
        'RhoVec': np.zeros(n_nodes),
        'Rho2LambdaVec': np.zeros(n_nodes)
    }

    # Retrieve properties (convert units: GPa→Pa, g/cm³→kg/m³)
    lambda_value = fem_matrices['PhysProp'][ii_d]['lambda'] * 1e9
    rho_value = fem_matrices['PhysProp'][ii_d]['rho'] * 1e3
    rho2lambda_value = rho_value ** 2 / lambda_value

    # Fill vectors (constant for all nodes in element)
    el_phys_props['LambdaVec'].fill(lambda_value)
    el_phys_props['RhoVec'].fill(rho_value)
    el_phys_props['Rho2LambdaVec'].fill(rho2lambda_value)

    return el_phys_props