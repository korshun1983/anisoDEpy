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
        InputParam.Misc.F_conv = 1.0
    elif InputParam.Config.FreqUnits == 'kHz':
        InputParam.Misc.F_conv = 1e3
    else:
        raise ValueError(f"Unsupported frequency units: {InputParam.Config.FreqUnits}")

    # Slowness units
    if InputParam.Config.SloUnits == 'us/m':
        InputParam.Misc.S_conv = 1e3
    elif InputParam.Config.SloUnits == 'us/ft':
        InputParam.Misc.S_conv = 0.3048 * 1e3
    else:
        raise ValueError(f"Unsupported slowness units: {InputParam.Config.SloUnits}")

    # Number of computational domains
    InputParam.Data.N_domain = len(InputParam.Model['DomainType'])
    logger.info(f"      Detected {InputParam.Data.N_domain} domains")

    # Variables per domain (1 for fluid, 3 for HTTI)
    InputParam.Data.DVarNum = np.zeros(InputParam.Data.N_domain, dtype=int)
    for i, domain_type in enumerate(InputParam.Model['DomainType']):
        if domain_type.lower() == 'fluid':
            InputParam.Data.DVarNum[i] = 1
        elif domain_type.lower() == 'htti':
            InputParam.Data.DVarNum[i] = 3
        else:
            raise ValueError(f"Unknown domain type: {domain_type}")

    logger.info(f"      Variables per domain: {InputParam.Data.DVarNum}")

    # Asymptote computation (if enabled)
    if InputParam.Config.CheckAsymptote == 'yes':
        logger.info("      Computing asymptotes...")
        # This will be implemented when we get to asymptotes.py
        # For now, create placeholder
        InputParam.Asymp = {}

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

    # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: НЕ перезаписываем Methods словарь!
    # Сохраняем существующие методы из Stage 1 (MeshFaces, St2/3/4 методы и т.д.)
    # Вместо InputParam.Methods = {}, только добавляем новые методы

    # Utility methods (только если не были заданы ранее)
    if 'AssembleBasicMatrices' not in InputParam.Methods:
        InputParam.Methods['AssembleBasicMatrices'] = None

    # Per-domain methods (cell arrays indexed by domain)
    # Эти методы специфичны для Stage 2 и должны быть установлены
    InputParam.Methods['PreparePhysProp'] = [None] * InputParam.Data.N_domain
    InputParam.Methods['MatricesParts_sp_SAFE'] = [None] * InputParam.Data.N_domain
    InputParam.Methods['getPhysProps'] = [None] * InputParam.Data.N_domain
    InputParam.Methods['KM_matrix'] = [None] * InputParam.Data.N_domain
    InputParam.Methods['KM_el_matrix'] = [None] * InputParam.Data.N_domain

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
            if i == InputParam.Data.N_domain - 1 and InputParam.Model.get('AddDomain_Exist', 'no') == 'yes':
                if InputParam.Model['AddDomainType'].lower() == 'pml':
                    InputParam.Methods['KM_el_matrix'][i] = 'km_el_matrix_htti_pml'
                elif InputParam.Model['AddDomainType'].lower() == 'abc':
                    InputParam.Methods['KM_el_matrix'][i] = 'km_el_matrix_htti_abc'
                elif InputParam.Model['AddDomainType'].lower() == 'pml+abc':
                    InputParam.Methods['KM_el_matrix'][i] = 'km_el_matrix_htti_pml_abc'

    # Interface methods (between domains)
    n_interfaces = InputParam.Data.N_domain - 1
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

    # Variable reduction method (только если не был задан ранее)
    if 'RemoveRedundantVariables' not in InputParam.Methods:
        InputParam.Methods['RemoveRedundantVariables'] = 'remove_redundant_variables'

    logger.info("      Method assignment complete")
    return InputParam


# -----------------------------------------------------------------------------
# Main Stage 2 Orchestrator
# -----------------------------------------------------------------------------

def prepare_model(InputParam: InputParam) -> InputParam:
    logger.info("  Stage 2: Preparing model for computation...")

    # Stage 2.1: Preprocess parameters
    CompStruct = prepare_model_params(InputParam)

    # Stage 2.2: Assign methods
    CompStruct = prepare_model_methods(CompStruct)

    # Stage 2.3: Generate mesh
    logger.info("    Stage 2.3: Generating mesh...")

    # Check if mesh already exists
    if not hasattr(CompStruct, 'FEMatrices') or CompStruct.FEMatrices is None:
        # Initialize FEMatrices structure
        CompStruct.FEMatrices = {}

        # Generate mesh using the registered method (should be prepare_mesh)
        if CompStruct.Methods.get('MeshFaces') is not None:
            try:
                logger.info("      Running mesh generation...")
                mesh_data = CompStruct.Methods['MeshFaces'](CompStruct)
                CompStruct.FEMatrices.update(mesh_data)
                logger.info(f"      Mesh generated: {CompStruct.FEMatrices['MeshNodes'].shape[1]} nodes")
            except Exception as e:
                logger.error(f"Mesh generation failed: {e}")
                raise
        else:
            raise ValueError("MeshFaces method not registered in Stage 1")

    logger.info("  Stage 2 complete")
    return CompStruct