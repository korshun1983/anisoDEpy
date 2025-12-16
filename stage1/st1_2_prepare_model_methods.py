"""
St1_2_PrepareModelMethods.py
============================
EXACT MATLAB equivalent of St1_2_PrepareModelMethods.m
"""

import os
from utils import debug_print  # Import from root


def st1_2_prepare_model_methods(InputParam):
    """
    Assign methods based on configuration.

    Parameters
    ----------
    InputParam : dict
        Input parameters structure

    Returns
    -------
    InputParam : dict
        Updated with Methods structure
    """
    debug_print("  St1_2: Preparing model methods...", level=3)

    InputParam['Methods'] = {}
    Methods = InputParam['Methods']

    # Get root path (MATLAB: cd('..'); root_path = cd('routines\'))
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.abspath(os.path.join(current_dir, '..'))  # Go up from stage1/

    # Configure root path with routines subdir
    InputParam['Config']['root_path'] = os.path.join(root_path, 'routines')
    debug_print(f"    Root path: {InputParam['Config']['root_path']}", level=4)

    # Assign user model function (placeholder)
    Methods['St1_3_SetModelUser'] = None
    debug_print("    St1_3_SetModelUser: [WILL BE SET BY USER MODEL]", level=4)

    # Helper routines (placeholders)
    Methods['em_tensor_VTI'] = None
    Methods['rot_c_ij'] = None
    Methods['rot_matrix'] = None
    Methods['ComputeAsymptotes'] = None
    debug_print("    Helper routines: [PLACEHOLDERS]", level=4)

    # Determine solver path based on configuration
    solver_path = ''

    if InputParam['Config']['ProblemType'] == 'spectrum':
        if InputParam['Config']['NumMethod'] == 'SAFE':
            solver_path = os.path.join('spectrum', '')
            debug_print(f"    Solver path: {solver_path}", level=4)

            # Assign Stage 2-4 function placeholders
            Methods['St1_4_SetModelAdvanced'] = None
            Methods['St2_PrepareModel'] = None
            Methods['St2_1_PrepareModelParams'] = None
            Methods['St2_2_PrepareModelMethods'] = None
            Methods['St3_PrepareBasicMatrices'] = None
            Methods['St4_ComputeSolution'] = None

    InputParam['Config']['solver_path'] = solver_path

    debug_print("  St1_2: Methods preparation complete", level=3)

    return InputParam