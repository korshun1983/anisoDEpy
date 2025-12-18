#!/usr/bin/env python3
"""
Step 1 functions for ANISO_SAFE project.
Analogous to MATLAB St1_*.m functions.
"""

import json
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

from config.structures import (
    Config, ModelConfig, AdvancedConfig, Methods, InputParam
)


# ============================================================================
# STEP 1: SUB-FUNCTIONS
# ============================================================================

def St1_1_SetModelConfig() -> Config:
    """
    Set the configuration parameters
    From St1_1_SetModelConfig.m
    """
    config = Config()

    config.ProblemType = 'spectrum'
    config.NumMethod = 'SAFE'
    config.SpeedUp = 'no'
    config.SaveData = 'yes'
    config.OuterBC = 'fixed'
    config.PML = 'r2'
    config.Eccentricity = 'no'
    config.Symmetry = 'none'
    config.EigenVar = 'k'
    config.SloUnits = 'us/ft'
    config.FreqUnits = 'kHz'
    config.PressureUnits = 'GPa'
    config.CheckAsymptote = 'yes'
    config.DisplayAttenuation = 'yes'

    return config


def St1_2_PrepareModelMethods(config: Config) -> Methods:
    """
    Assign the methods which will be used for further steps
    From St1_2_PrepareModelMethods.m
    """
    methods = Methods()

    methods.St1_3_SetModelUser = St1_3_SetModelUser_sp_SAFE
    methods.St1_4_SetModelAdvanced = St1_4_SetModelAdvanced_sp_SAFE

    root_path = Path(__file__).parent.parent.resolve()
    config.root_path = str(root_path) + '/'

    # Set solver path
    solver_path = config.root_path
    if config.ProblemType == 'spectrum':
        solver_path = str(Path(solver_path) / 'spectrum')
        if config.NumMethod == 'SAFE':
            methods.ComputeAsymptotes = None

    config.solver_path = solver_path + '/'

    # Placeholders for Steps 2-4
    methods.St2_PrepareModel = None
    methods.St2_1_PrepareModelParams = None
    methods.St2_2_PrepareModelMethods = None
    methods.St3_PrepareBasicMatrices = None
    methods.St4_ComputeSolution = None

    return methods


# ============================================================================
# DOMAIN EXTENSION LOGIC
# ============================================================================

def add_external_domain(model: ModelConfig) -> ModelConfig:
    """
    Add external PML/ABC domain to the model
    Mirrors MATLAB logic from gen_aniso.m
    """
    print(f"    Adding external {model.AddDomainType} domain...")

    # Append domain parameters
    model.DomainTheta.append(model.DomainTheta[-1])
    model.DomainEcc.append(model.DomainEcc[-1])
    model.DomainEccAngle.append(model.DomainEccAngle[-1])
    model.DomainParam.append(model.DomainParam[-1])
    model.DomainType.append(model.DomainType[-1])
    model.DomainNth.append(model.DomainNth[-1])

    # Update radii
    if model.LDomain_in_LSH == 'yes':
        ref_domain = model.DomainParam[-2]
        if len(ref_domain) >= 7:
            rho = ref_domain[0] * 1000  # g/cm³ → kg/m³
            c66 = ref_domain[5] * 1e9  # GPa → Pa
            v_sh = np.sqrt(c66 / rho)
            wavelength = v_sh / (model.f_min * 1000)
            outer_radius = model.DomainRx[-2] + model.AddDomainL * wavelength
        else:
            outer_radius = model.DomainRx[-2] * (1 + model.AddDomainL)
    else:
        outer_radius = model.DomainRx[-2] + model.AddDomainL

    model.DomainRx.append(outer_radius)
    model.DomainRy.append(outer_radius)

    # Update boundary conditions
    if len(model.BCType) == len(model.DomainType) - 1:
        model.BCType.append('rigid')
    else:
        model.BCType[-1] = 'SSstiff'
        model.BCType.append('rigid')

    return model


# ============================================================================
# STEP 1: MAIN FUNCTIONS
# ============================================================================

def St1_3_SetModelUser_sp_SAFE(input_param: InputParam, json_path: Path) -> InputParam:
    """
    Set user model parameters from JSON file
    Adapted from St1_3_SetModelUser_sp_SAFE.m
    """
    print(f"    Loading model from JSON: {json_path.name}")

    with open(json_path, 'r') as f:
        json_data = json.load(f)

    model = input_param.Model

    # Geometry
    model.DomainRx = json_data['Model']['DomainRx']
    model.DomainRy = json_data['Model']['DomainRy']
    model.DomainTheta = json_data['Model']['DomainTheta']
    model.DomainEcc = json_data['Model']['DomainEcc']
    model.DomainEccAngle = json_data['Model']['DomainEccAngle']
    model.LDomain_in_LSH = json_data['Model']['LDomain_in_LSH']

    # Domains and BCs
    model.DomainType = json_data['Model']['DomainType']
    model.BCType = json_data['Model']['BCType']

    # Additional domain
    model.AddDomainLoc = json_data['Model']['AddDomainLoc']
    model.AddDomainType = json_data['Model']['AddDomainType']
    model.AddDomainL = json_data['Model']['AddDomainL']

    # PML/ABC parameters
    model.PML_factor = json_data['Model']['PML_factor']
    model.PML_degree = json_data['Model']['PML_degree']
    model.PML_method = json_data['Model']['PML_method']
    model.ABC_factor = json_data['Model']['ABC_factor']
    model.ABC_degree = json_data['Model']['ABC_degree']
    model.ABC_account_r = json_data['Model']['ABC_account_r']

    # Physical properties
    model.DomainParam = json_data['Model']['DomainParam']
    model.RefDomainType = json_data['Model']['RefDomainType']
    model.RefDomainParam = json_data['Model']['RefDomainParam']

    # Discretization
    model.DomainNth = json_data['Model']['DomainNth']
    model.mud_domain = json_data['Model']['mud_domain']

    # Frequency array
    f_range = json_data['Model']['f_array_range']
    model.f_array = np.arange(f_range['start'], f_range['end'] + f_range['step'] / 2, f_range['step'])
    model.N_disp = len(model.f_array)
    model.f_min = model.f_array[0]
    model.f_max = model.f_array[-1]

    # Override advanced parameters from JSON
    if 'Advanced' in json_data:
        advanced_data = json_data['Advanced']
        input_param.Advanced.num_eig_max = advanced_data.get('num_eig_max',
                                                             input_param.Advanced.num_eig_max)
        input_param.Advanced.EigSearchStart = advanced_data.get('EigSearchStart',
                                                                input_param.Advanced.EigSearchStart)

        # Update model fields for reference (optional, for backward compatibility)
        model.num_eig_max = input_param.Advanced.num_eig_max
        model.EigSearchStart = input_param.Advanced.EigSearchStart

    # Process AddDomain_Exist flag
    domain_type_lower = model.AddDomainType.lower()
    if domain_type_lower == 'abc+pml':
        domain_type_lower = 'pml+abc'

    model.AddDomain_Exist = 'none'
    if domain_type_lower != 'none':
        if domain_type_lower in ['pml', 'abc', 'pml+abc', 'same']:
            model.AddDomain_Exist = 'yes'
        else:
            raise ValueError(f"Error: AddDomainType '{model.AddDomainType}' is not set correctly!")

    # Add external domain if needed
    if model.AddDomain_Exist == 'yes' and model.AddDomainLoc == 'ext':
        model = add_external_domain(model)

    return input_param


def St1_4_SetModelAdvanced_sp_SAFE(input_param: InputParam) -> InputParam:
    """
    Set advanced parameters for SAFE method
    From St1_4_SetModelAdvanced_sp_SAFE.m
    """
    advanced = input_param.Advanced

    advanced.VisualizeMesh = True
    advanced.N_nodes = 10
    advanced.NEdge_nodes = 4
    advanced.EigsOptions = {'disp': 0, 'tol': 1e-8}
    advanced.Source = {
        'xc': 0.0, 'yc': 0.0, 'r0x': 0.06, 'r0y': 0.06,
        'theta_r': 0.0, 'theta0': 0.0, 'sigma': 0.02,
        'Plim': 5e-4, 'symmetry': 0
    }

    return input_param


def St1_SetModel(json_path: Path) -> InputParam:
    """
    Main Step 1 function - sets up the complete model
    From St1_SetModel.m
    """
    print('\n=== Step 1: Initialization ===\n')

    input_param = InputParam()

    print('  Setting configuration parameters...')
    input_param.Config = St1_1_SetModelConfig()

    print('  Preparing model methods...')
    input_param.Methods = St1_2_PrepareModelMethods(input_param.Config)

    print('  Loading user model parameters...')
    input_param = St1_3_SetModelUser_sp_SAFE(input_param, json_path)

    input_param.Model.N_disp = len(input_param.Model.f_array)

    print('  Setting advanced parameters...')
    input_param = St1_4_SetModelAdvanced_sp_SAFE(input_param)

    print('\n=== Step 1 Completed ===\n')
    return input_param