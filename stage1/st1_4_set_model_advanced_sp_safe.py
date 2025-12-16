"""
St1_4_SetModelAdvanced.py
=========================
EXACT MATLAB equivalent of St1_4_SetModelAdvanced_sp_SAFE.m
"""

from utils import debug_print  # Import from root


def st1_4_set_model_advanced_sp_safe(InputParam):
    """
    Set advanced parameters for SAFE computations.

    Parameters
    ----------
    InputParam : dict
        Input parameters structure

    Returns
    -------
    InputParam : dict
        Updated with Advanced parameters
    """
    debug_print("  St1_4: Setting advanced parameters...", level=3)

    # Initialize Advanced structure if not exists
    if 'Advanced' not in InputParam:
        InputParam['Advanced'] = {}

    Advanced = InputParam['Advanced']

    # Mesh visualization
    Advanced['VisualizeMesh'] = True
    debug_print(f"    VisualizeMesh: {Advanced['VisualizeMesh']}", level=4)

    # Number of nodes per element (3=linear, 6=quadratic, 10=cubic)
    Advanced['N_nodes'] = 10
    Advanced['NEdge_nodes'] = 4
    debug_print(f"    Element order: {Advanced['N_nodes']} nodes", level=4)

    # Number of eigenvalues (from Model.Advanced, set by user)
    if 'num_eig_max' not in Advanced:
        Advanced['num_eig_max'] = InputParam['Model'].get('num_eig_max', 50)
    debug_print(f"    Number of eigenvalues: {Advanced['num_eig_max']}", level=4)

    # Starting velocity for eigs
    if 'EigSearchStart' not in Advanced:
        Advanced['EigSearchStart'] = InputParam['Model'].get('EigSearchStart', 1.0)
    debug_print(f"    Eigenvalue search start: {Advanced['EigSearchStart']} km/s", level=4)

    # eigs options
    Advanced['EigsOptions'] = {
        'disp': 0,
        'tol': 1e-8
    }
    debug_print(f"    eigs tolerance: {Advanced['EigsOptions']['tol']}", level=4)

    # Source parameters
    Advanced['Source'] = {
        'xc': 0, 'yc': 0, 'r0x': 0.06, 'r0y': 0.06,
        'theta_r': 0, 'theta0': 0, 'sigma': 0.02,
        'Plim': 5e-4, 'symmetry': 0
    }

    debug_print("  St1_4: Advanced parameters complete", level=3)

    return InputParam