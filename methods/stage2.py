# methods/stage2.py
"""
===============================================================================
Stage 2: Model Preparation Pipeline
Replicates St2_PrepareModel, St2_1_PrepareModelParams, St2_2_PrepareModelMethods
===============================================================================
"""

import numpy as np
from typing import Dict, List, Any, Callable
from pathlib import Path
from core.config import InputParam
import logging

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Stage 2.1: Parameter Preprocessing
# -----------------------------------------------------------------------------

def prepare_model_params(InputParam: InputParam) -> InputParam:
    """
    Replicates St2_1_PrepareModelParams_sp_SAFE.m
    Preprocesses model parameters and computes derived quantities.
    """
    logger.info("    Stage 2.1: Preparing model parameters...")

    # Unit conversion factors
    # Frequency units
    if InputParam.Config.FreqUnits == 'Hz':
        InputParam.Misc['F_conv'] = 1.0
    elif InputParam.Config.FreqUnits == 'kHz':
        InputParam.Misc['F_conv'] = 1e3
    else:
        raise ValueError(f"Unsupported frequency units: {InputParam.Config.FreqUnits}")

    # Slowness units
    if InputParam.Config.SloUnits == 'us/m':
        InputParam.Misc['S_conv'] = 1e3
    elif InputParam.Config.SloUnits == 'us/ft':
        InputParam.Misc['S_conv'] = 0.3048 * 1e3
    else:
        raise ValueError(f"Unsupported slowness units: {InputParam.Config.SloUnits}")

    # Number of computational domains
    InputParam.Data['N_domain'] = len(InputParam.Model['DomainType'])
    logger.info(f"      Detected {InputParam.Data['N_domain']} domains")

    # Variables per domain (1 for fluid, 3 for HTTI)
    InputParam.Data['DVarNum'] = np.zeros(InputParam.Data['N_domain'], dtype=int)
    for i, domain_type in enumerate(InputParam.Model['DomainType']):
        if domain_type.lower() == 'fluid':
            InputParam.Data['DVarNum'][i] = 1
        elif domain_type.lower() == 'htti':
            InputParam.Data['DVarNum'][i] = 3
        else:
            raise ValueError(f"Unknown domain type: {domain_type}")

    logger.info(f"      Variables per domain: {InputParam.Data['DVarNum']}")

    # Asymptote computation (if enabled)
    if InputParam.Config.CheckAsymptote == 'yes':
        logger.info("      Computing asymptotes...")
        # This will be implemented when we get to asymptotes.py
        # For now, create placeholder
        InputParam.Asymp = {}
        # InputParam.Asymp = InputParam.Methods['ComputeAsymptotes'](InputParam)

        # Adjust velocity search range if asymptotes available
        if 'V_SH' in InputParam.Asymp:
            InputParam.Advanced.V_min = InputParam.Asymp['V_SH']
        if 'V_qP' in InputParam.Asymp:
            InputParam.Advanced.V_max = InputParam.Asymp['V_qP']

    return InputParam


# -----------------------------------------------------------------------------
# Stage 2.2: Method Assignment
# -----------------------------------------------------------------------------

def prepare_model_methods(InputParam: InputParam) -> InputParam:
    """
    Replicates St2_2_PrepareModelMethods_sp_SAFE.m
    Assigns function handles based on domain types and configuration.
    """
    logger.info("    Stage 2.2: Assigning model methods...")

    # Initialize methods dictionary
    InputParam.Methods = {}

    # Utility methods (always available)
    InputParam.Methods['chebdif'] = None  # Will implement when needed
    InputParam.Methods['em_tensor_VTI'] = None
    InputParam.Methods['rot_c_ij'] = None
    InputParam.Methods['rot_matrix'] = None
    InputParam.Methods['V_phase_VTI_exact_RPH'] = None
    InputParam.Methods['MeshFaces'] = None

    # Mesh generation methods
    InputParam.Methods['PrepareMesh'] = None
    InputParam.Methods['PrepareMeshBH'] = None
    InputParam.Methods['FindBEdges'] = None
    InputParam.Methods['MakeContBEdges'] = None
    InputParam.Methods['FindEdgeOrient'] = None
    InputParam.Methods['AddNodesCubic'] = None

    # Basic matrix assembly methods
    InputParam.Methods['FindPos'] = None
    InputParam.Methods['L1L2_int_matrix'] = None
    InputParam.Methods['L1L2L3_int_matrix'] = None
    InputParam.Methods['NL_matrix'] = None
    InputParam.Methods['dNL_matrices'] = None
    InputParam.Methods['ConvolveMatrices'] = None
    InputParam.Methods['ConvolveEdgeMatrices'] = None
    InputParam.Methods['NLEdge_matrix'] = None
    InputParam.Methods['AssembleBasicMatrices'] = None

    # Per-domain methods (cell arrays indexed by domain)
    InputParam.Methods['PreparePhysProp'] = [None] * InputParam.Data['N_domain']
    InputParam.Methods['MatricesParts_sp_SAFE'] = [None] * InputParam.Data['N_domain']
    InputParam.Methods['getPhysProps'] = [None] * InputParam.Data['N_domain']
    InputParam.Methods['KM_matrix'] = [None] * InputParam.Data['N_domain']
    InputParam.Methods['KM_el_matrix'] = [None] * InputParam.Data['N_domain']

    # Assign domain-specific methods
    for i, domain_type in enumerate(InputParam.Model['DomainType']):
        if domain_type.lower() == 'fluid':
            InputParam.Methods['PreparePhysProp'][i] = 'prepare_physprop_fluid'
            InputParam.Methods['MatricesParts_sp_SAFE'][i] = 'matrices_parts_fluid'
            InputParam.Methods['getPhysProps'][i] = 'get_physprops_fluid'
            InputParam.Methods['KM_matrix'][i] = 'km_matrix_fluid'
            InputParam.Methods['KM_el_matrix'][i] = 'km_el_matrix_fluid'

        elif domain_type.lower() == 'htti':
            InputParam.Methods['PreparePhysProp'][i] = 'prepare_physprop_htti'
            InputParam.Methods['MatricesParts_sp_SAFE'][i] = 'matrices_parts_htti'
            InputParam.Methods['getPhysProps'][i] = 'get_physprops_htti'
            InputParam.Methods['KM_matrix'][i] = 'km_matrix_htti'
            InputParam.Methods['KM_el_matrix'][i] = 'km_el_matrix_htti'

            # PML/ABC variants for outer domain if present
            if i == InputParam.Data['N_domain'] - 1 and InputParam.Model['AddDomain_Exist'] == 'yes':
                if InputParam.Model['AddDomainType'].lower() == 'pml':
                    InputParam.Methods['KM_el_matrix'][i] = 'km_el_matrix_htti_pml'
                elif InputParam.Model['AddDomainType'].lower() == 'abc':
                    InputParam.Methods['KM_el_matrix'][i] = 'km_el_matrix_htti_abc'
                elif InputParam.Model['AddDomainType'].lower() == 'pml+abc':
                    InputParam.Methods['KM_el_matrix'][i] = 'km_el_matrix_htti_pml_abc'

    # Interface methods (between domains)
    n_interfaces = InputParam.Data['N_domain'] - 1
    InputParam.Methods['IC_Matrices_sp_SAFE'] = [None] * n_interfaces
    InputParam.Methods['AssembleFullMatrices'] = [None] * (n_interfaces + 1)

    for i in range(n_interfaces):
        type1 = InputParam.Model['DomainType'][i].lower()
        type2 = InputParam.Model['DomainType'][i + 1].lower()

        if {type1, type2} == {'fluid', 'htti'} or {type1, type2} == {'htti', 'fluid'}:
            InputParam.Methods['IC_Matrices_sp_SAFE'][i] = 'ic_matrices_fluid_htti'
            InputParam.Methods['AssembleFullMatrices'][i] = 'assemble_full_matrices_fs'
        elif type1 == type2:
            InputParam.Methods['IC_Matrices_sp_SAFE'][i] = 'ic_matrices_ff_ss'
            InputParam.Methods['AssembleFullMatrices'][i] = 'assemble_full_matrices_ff_ss'

    # Boundary condition methods
    outer_bc = InputParam.Model['BCType'][-1].lower()
    if outer_bc == 'rigid':
        InputParam.Methods['AssembleFullMatrices'][-1] = 'assemble_full_matrices_rigid'
    elif outer_bc == 'free':
        InputParam.Methods['AssembleFullMatrices'][-1] = 'assemble_full_matrices_free'
    else:
        raise ValueError(f"Unsupported outer BC: {outer_bc}")

    # Variable reduction method
    InputParam.Methods['RemoveRedundantVariables'] = 'remove_redundant_variables'

    logger.info("      Method assignment complete")
    return InputParam


# -----------------------------------------------------------------------------
# Main Stage 2 Orchestrator
# -----------------------------------------------------------------------------

def prepare_model(InputParam: InputParam) -> InputParam:
    """
    Replicates St2_PrepareModel_sp_SAFE.m
    Main orchestrator for Stage 2 model preparation.
    """
    logger.info("  Stage 2: Preparing model for computation...")

    # Copy InputParam to CompStruct (Python: just use same object)
    CompStruct = InputParam

    # Stage 2.1: Preprocess parameters
    CompStruct = prepare_model_params(CompStruct)

    # Stage 2.2: Assign methods
    CompStruct = prepare_model_methods(CompStruct)

    logger.info("  Stage 2 complete")
    return CompStruct