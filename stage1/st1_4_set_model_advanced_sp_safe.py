"""
St1_4_SetModelAdvanced.py
=========================
EXACT MATLAB equivalent of St1_4_SetModelAdvanced_sp_SAFE.m
"""

from utils import debug_print  # Import from root


def st1_4_set_model_advanced_sp_safe(InputParam):
    """Set advanced parameters for SAFE computations."""
    debug_print("  St1_4: Setting advanced parameters...", level=3)

    # Initialize Advanced structure if not exists
    if 'Advanced' not in InputParam:
        InputParam['Advanced'] = {}

    Advanced = InputParam['Advanced']

    # Advanced parameters initialization
    Advanced['VisualizeMesh'] = True
    Advanced['N_nodes'] = 10
    Advanced['NEdge_nodes'] = 4

    # Ccheck Model.Advanced
    model_advanced = InputParam.get('Model', {}).get('Advanced', {})

    Advanced['num_eig_max'] = model_advanced.get('num_eig_max', 10)
    Advanced['EigSearchStart'] = model_advanced.get('EigSearchStart', 1.0)

    # eigs options
    Advanced['EigsOptions'] = {'disp': 0, 'tol': 1e-8}

    # Source parameters
    Advanced['Source'] = {
        'xc': 0, 'yc': 0, 'r0x': 0.06, 'r0y': 0.06,
        'theta_r': 0, 'theta0': 0, 'sigma': 0.02,
        'Plim': 5e-4, 'symmetry': 0
    }

    debug_print("  St1_4: Advanced parameters complete", level=3)
    return InputParam