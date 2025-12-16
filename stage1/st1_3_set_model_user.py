"""
St1_3_SetModelUser.py
=====================
PLACEHOLDER for user model - exact MATLAB equivalent structure.
"""

from utils import debug_print  # Import from root


def st1_3_set_model_user_sp_safe(InputParam):
    """
    Set user model parameters - PLACEHOLDER.

    Parameters
    ----------
    InputParam : dict
        Input parameters structure

    Returns
    -------
    InputParam : dict
        Updated with Model parameters
    """
    debug_print("  St1_3: Setting user model parameters...", level=3)
    debug_print("    WARNING: Using placeholder model parameters!", level=1)

    Model = {}

    # Minimal placeholder model for testing framework
    Model['DomainRx'] = [0.1, 2.0]
    Model['DomainRy'] = [0.1, 2.0]
    Model['DomainTheta'] = [0, 0]
    Model['DomainEcc'] = [0, 0]
    Model['DomainEccAngle'] = [0, 0]
    Model['LDomain_in_LSH'] = 'yes'

    # Layer types
    Model['DomainType'] = ['fluid', 'HTTI']
    debug_print(f"    Layer types: {Model['DomainType']}", level=4)

    # PML/ABC parameters
    Model['AddDomainLoc'] = 'ext'
    Model['AddDomainType'] = 'abc'
    Model['AddDomainL'] = 1.0
    Model['PML_factor'] = 10
    Model['PML_degree'] = 2.0
    Model['PML_method'] = 2.0
    Model['ABC_factor'] = 0.1
    Model['ABC_degree'] = 1.0
    Model['ABC_account_r'] = 'yes'

    # Physical properties
    Model['DomainParam'] = [
        [1.0, 2.25e9],  # Fluid
        [2.23, 40.9e9, 8.5e9, 26.9e9, 10.5e9, 15.3e9, 0, 0]  # HTTI
    ]

    # Reference
    Model['RefDomainType'] = ['HTTI']
    Model['RefDomainParam'] = [Model['DomainParam'][1]]
    Model['RefDomainParam'][0][6] = 0

    # BCs
    Model['BCType'] = ['FS', 'rigid']

    # Frequency array
    Model['f_array'] = [1.0, 2.0, 3.0]  # Small array for testing

    # Discretization
    Model['DomainNth'] = [12, 12]
    Model['mud_domain'] = 1

    # Assign to InputParam
    InputParam['Model'] = Model

    # Advanced parameters (will be overridden by St1_4)
    InputParam['Advanced'] = {
        'num_eig_max': 10,
        'EigSearchStart': 1.0
    }

    # Mesh parameters
    InputParam['Mesh'] = {
        'output': 'no',
        'hmax': 0.16,
        'dhmax': 0.25,
        'ext_boundary_shape': 'cir'
    }

    debug_print("  St1_3: User model parameters set", level=3)

    return InputParam