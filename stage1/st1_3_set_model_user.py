"""
St1_3_SetModelUser.py
=====================
User model setup - JSON-based and placeholder versions.
"""

import numpy as np
from utils import debug_print


def st1_3_set_model_user_from_json(InputParam: dict, model_data: dict) -> dict:
    """
    Populate model parameters from loaded JSON data.

    Parameters
    ----------
    InputParam : dict
        Input parameters structure to populate
    model_data : dict
        Loaded JSON model data

    Returns
    -------
    InputParam : dict
        Updated with Model parameters from JSON
    """
    debug_print("  St1_3: Setting model parameters from JSON...", level=3)

    # Extract JSON sections
    model_json = model_data.get('Model', {})
    advanced_json = model_data.get('Advanced', {})
    mesh_json = model_data.get('Mesh', {})

    Model = {}

    # Domain geometry
    Model['DomainRx'] = model_json.get('DomainRx', [0.1, 2.0])
    Model['DomainRy'] = model_json.get('DomainRy', [0.1, 2.0])
    Model['DomainTheta'] = model_json.get('DomainTheta', [0, 0])
    Model['DomainEcc'] = model_json.get('DomainEcc', [0, 0])
    Model['DomainEccAngle'] = model_json.get('DomainEccAngle', [0, 0])
    Model['LDomain_in_LSH'] = model_json.get('LDomain_in_LSH', 'yes')

    # Layer types
    Model['DomainType'] = model_json.get('DomainType', ['fluid', 'HTTI'])
    debug_print(f"    Layer types: {Model['DomainType']}", level=4)

    # PML/ABC parameters
    Model['AddDomainLoc'] = model_json.get('AddDomainLoc', 'ext')
    Model['AddDomainType'] = model_json.get('AddDomainType', 'abc')
    Model['AddDomainL'] = model_json.get('AddDomainL', 1.0)
    Model['PML_factor'] = model_json.get('PML_factor', 10)
    Model['PML_degree'] = model_json.get('PML_degree', 2.0)
    Model['PML_method'] = model_json.get('PML_method', 2.0)
    Model['ABC_factor'] = model_json.get('ABC_factor', 0.1)
    Model['ABC_degree'] = model_json.get('ABC_degree', 1.0)
    Model['ABC_account_r'] = model_json.get('ABC_account_r', 'yes')

    # Physical properties
    Model['DomainParam'] = model_json.get('DomainParam', [
        [1000.0, 2.25e9],  # Fluid
        [2600.0, 40.9e9, 8.5e9, 26.9e9, 10.5e9, 15.3e9, 0, 0]  # HTTI
    ])

    # Reference domain
    Model['RefDomainType'] = model_json.get('RefDomainType', ['HTTI'])
    Model['RefDomainParam'] = model_json.get('RefDomainParam', [Model['DomainParam'][1]])
    if len(Model['RefDomainParam']) > 0 and len(Model['RefDomainParam'][0]) > 6:
        Model['RefDomainParam'][0][6] = 0  # Set theta to 0 for reference

    # Boundary conditions
    Model['BCType'] = model_json.get('BCType', ['FS', 'rigid'])

    # CRITICAL: Preserve f_array_range for downstream use
    f_array_range = model_json.get('f_array_range', {})
    if f_array_range:
        start = f_array_range.get('start', 1.0)
        step = f_array_range.get('step', 1.0)
        end = f_array_range.get('end', 15.0)
        Model['f_array'] = np.arange(start, end + step/2, step).tolist()
        Model['f_array_range'] = f_array_range  # ⭐ Сохраняем для Gmsh builder
        debug_print(f"    Frequency range: {start}:{step}:{end} kHz", level=4)
    else:
        # Fallback if no range specified
        Model['f_array'] = model_json.get('f_array', [1.0, 2.0, 3.0])
        debug_print(f"    Frequency array: {Model['f_array']}", level=4)

    # Discretization
    Model['DomainNth'] = model_json.get('DomainNth', [12, 12])
    Model['mud_domain'] = model_json.get('mud_domain', 1)

    # Store in InputParam
    InputParam['Model'] = Model
    InputParam['Advanced'] = advanced_json
    InputParam['Mesh'] = mesh_json

    # Store number of frequency points for convenience
    InputParam['Model']['N_disp'] = len(InputParam['Model']['f_array'])

    debug_print("  St1_3: Model parameters from JSON complete", level=3)

    return InputParam


def st1_3_set_model_user_sp_safe(InputParam: dict) -> dict:
    """
    PLACEHOLDER user model - for testing when no JSON is provided.
    Do NOT modify for production - always use JSON.
    """
    debug_print("  St1_3: Using PLACEHOLDER model parameters!", level=1)

    Model = {}

    # Simplified placeholder (matches Bakken-B structure)
    Model['DomainRx'] = [0.1, 2.0]
    Model['DomainRy'] = [0.1, 2.0]
    Model['DomainTheta'] = [0, 0]
    Model['DomainEcc'] = [0, 0]
    Model['DomainEccAngle'] = [0, 0]
    Model['LDomain_in_LSH'] = 'yes'

    Model['DomainType'] = ['fluid', 'HTTI']
    debug_print(f"    Layer types: {Model['DomainType']}", level=4)

    Model['AddDomainLoc'] = 'ext'
    Model['AddDomainType'] = 'abc'
    Model['AddDomainL'] = 1.0
    Model['PML_factor'] = 10
    Model['PML_degree'] = 2.0
    Model['PML_method'] = 2.0
    Model['ABC_factor'] = 0.1
    Model['ABC_degree'] = 1.0
    Model['ABC_account_r'] = 'yes'

    Model['DomainParam'] = [
        [1000.0, 2.25e9],  # Fluid
        [2600.0, 40.9e9, 8.5e9, 26.9e9, 10.5e9, 15.3e9, 0, 0]  # HTTI
    ]

    Model['RefDomainType'] = ['HTTI']
    Model['RefDomainParam'] = [Model['DomainParam'][1]]
    Model['RefDomainParam'][0][6] = 0

    Model['BCType'] = ['FS', 'rigid']

    # Placeholder frequency
    Model['f_array'] = [1.0, 2.0, 3.0]
    Model['f_array_range'] = {'start': 1.0, 'step': 1.0, 'end': 3.0}  # Для совместимости

    Model['DomainNth'] = [12, 12]
    Model['mud_domain'] = 1

    # Store in InputParam
    InputParam['Model'] = Model
    InputParam['Advanced'] = {
        'num_eig_max': 10,
        'EigSearchStart': 1.0
    }
    InputParam['Mesh'] = {
        'output': 'no',
        'hmax': 0.16,
        'dhmax': 0.25,
        'ext_boundary_shape': 'cir'
    }

    InputParam['Model']['N_disp'] = len(InputParam['Model']['f_array'])

    debug_print("  St1_3: Placeholder model parameters set", level=3)

    return InputParam