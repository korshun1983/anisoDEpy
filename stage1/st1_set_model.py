"""
St1_SetModel.py
===============
EXACT MATLAB equivalent of St1_SetModel.m - modified to accept JSON model data.
"""

import numpy as np  # <-- ADD THIS IMPORT
# Import all substages
from .st1_1_set_model_config import st1_1_set_model_config
from .st1_2_prepare_model_methods import st1_2_prepare_model_methods
from .st1_3_set_model_user import st1_3_set_model_user_sp_safe
from .st1_4_set_model_advanced_sp_safe import st1_4_set_model_advanced_sp_safe
from utils import debug_print


def st1_set_model(model_data: dict = None):
    """
    Main entry point for Stage 1.
    Calls all substages in exact MATLAB sequence.

    Parameters
    ----------
    model_data : dict, optional
        Loaded JSON model data. If None, uses placeholder model.

    Returns
    -------
    InputParam : dict
        Complete input parameter structure
    """
    debug_print("=" * 70, level=1)
    debug_print("STAGE 1: Setting Up Model Configuration", level=1)
    debug_print("=" * 70, level=1)

    # Initialize empty structure (like MATLAB's InputParam = struct)
    InputParam = {}

    # Stage 1.1: Configuration
    debug_print("Stage 1.1: Setting configuration...", level=2)
    InputParam = st1_1_set_model_config()

    # Stage 1.2: Prepare methods
    debug_print("Stage 1.2: Preparing model methods...", level=2)
    InputParam = st1_2_prepare_model_methods(InputParam)

    # Stage 1.3: User model parameters
    debug_print("Stage 1.3: Setting user model parameters...", level=2)
    if model_data is not None:
        # Use JSON data
        InputParam = st1_3_set_model_user_from_json(InputParam, model_data)
    else:
        # Use placeholder
        InputParam = st1_3_set_model_user_sp_safe(InputParam)

    # Set number of frequencies (MATLAB line: InputParam.Model.N_disp=length(InputParam.Model.f_array))
    InputParam['Model']['N_disp'] = len(InputParam['Model']['f_array'])
    debug_print(f"Number of frequency points: {InputParam['Model']['N_disp']}", level=3)

    # Stage 1.4: Advanced parameters
    debug_print("Stage 1.4: Setting advanced parameters...", level=2)
    InputParam = st1_4_set_model_advanced_sp_safe(InputParam)

    debug_print("Stage 1 complete: Model structure ready", level=1)
    debug_print("=" * 70, level=1)

    return InputParam


def st1_3_set_model_user_from_json(InputParam: dict, model_data: dict) -> dict:
    """
    Populate model parameters from loaded JSON data.
    Replaces the placeholder model with actual configuration.
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

    # Frequency array
    f_array_range = model_json.get('f_array_range', {})
    if f_array_range:
        start = f_array_range.get('start', 1.0)
        step = f_array_range.get('step', 1.0)
        end = f_array_range.get('end', 15.0)
        Model['f_array'] = np.arange(start, end + step/2, step).tolist()
    else:
        Model['f_array'] = model_json.get('f_array', [1.0, 2.0, 3.0])

    # Discretization
    Model['DomainNth'] = model_json.get('DomainNth', [12, 12])
    Model['mud_domain'] = model_json.get('mud_domain', 1)

    # Store in InputParam
    InputParam['Model'] = Model
    InputParam['Advanced'] = advanced_json
    InputParam['Mesh'] = mesh_json

    debug_print("  St1_3: Model parameters from JSON complete", level=3)

    return InputParam