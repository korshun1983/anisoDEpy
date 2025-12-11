# methods/stage2.py
"""
===============================================================================
Stage 2: Model Preparation Pipeline
Replicates St2_PrepareModel, St2_1_PrepareModelParams, St2_2_PrepareModelMethods
===============================================================================
CRITICAL FIXES:
- Separated mesh generation from model preparation
- Added hmax calculation based on wavelength (LDomain_in_LSH)
- Fixed DVarNum handling to prevent append errors
- MATLAB-compatible layer structure: DomainRx = outer radii
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

    # Number of computational domains (layers)
    InputParam.Data.N_domain = len(InputParam.Model['DomainType'])
    logger.info(f"      Detected {InputParam.Data.N_domain} domains")

    # Variables per domain (1 for fluid, 3 for HTTI)
    # CRITICAL: Use list instead of numpy array for append compatibility
    InputParam.Data.DVarNum = []
    for i, domain_type in enumerate(InputParam.Model['DomainType']):
        if domain_type.lower() == 'fluid':
            InputParam.Data.DVarNum.append(1)
        elif domain_type.lower() == 'htti':
            InputParam.Data.DVarNum.append(3)
        else:
            raise ValueError(f"Unknown domain type: {domain_type}")

    logger.info(f"      Variables per domain: {InputParam.Data.DVarNum}")

    # === WAVELENGTH-BASED MESH SIZE CALCULATION ===
    if InputParam.Model.get('LDomain_in_LSH', 'no') == 'yes':
        logger.info("      LDomain_in_LSH='yes': Computing wavelength-based mesh size...")

        # Get maximum frequency (kHz -> Hz)
        f_max = max(InputParam.Model['f_array']) * 1e3

        # Compute minimum velocity in model (m/s)
        min_velocity = float('inf')

        for i, domain_type in enumerate(InputParam.Model['DomainType']):
            params = InputParam.Model['DomainParam'][i]

            if domain_type.lower() == 'fluid':
                rho, lam = params[0], params[1]
                v = np.sqrt(lam / rho)  # Sound speed in fluid
                logger.info(f"        Domain {i + 1} (fluid): v={v:.1f} m/s")

            elif domain_type.lower() == 'htti':
                rho, c44 = params[0], params[3]  # c44 = shear modulus
                v = np.sqrt(c44 / rho)  # S-wave speed
                logger.info(f"        Domain {i + 1} (HTTI): v_s={v:.1f} m/s")

            min_velocity = min(min_velocity, v)

        if min_velocity == float('inf'):
            raise ValueError("Could not compute minimum velocity")

        # Minimum wavelength at maximum frequency
        lambda_min = min_velocity / f_max

        # Recommended hmax = fraction of lambda_min
        hmax_fraction = InputParam.Mesh.hmax
        hmax_absolute = lambda_min * hmax_fraction

        logger.info(f"      Max frequency: {f_max / 1e3:.2f} kHz")
        logger.info(f"      Min velocity: {min_velocity:.1f} m/s")
        logger.info(f"      Min wavelength: {lambda_min:.4f} m")
        logger.info(f"      hmax fraction: {hmax_fraction} ({hmax_fraction * 100:.0f}%)")
        logger.info(f"      Computed hmax (absolute): {hmax_absolute:.4f} m")

        # Store absolute value for mesh generation
        InputParam.Mesh.hmax_absolute = hmax_absolute
    else:
        # If LDomain_in_LSH='no', use hmax as absolute value (meters)
        InputParam.Mesh.hmax_absolute = InputParam.Mesh.hmax
        logger.info(f"      LDomain_in_LSH='no': Using absolute hmax={InputParam.Mesh.hmax} m")
    # ==========================================================

    # Asymptote computation (if enabled)
    if InputParam.Config.CheckAsymptote == 'yes':
        logger.info("      Computing asymptotes...")
        # Placeholder for future implementation
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

    # CRITICAL: Do NOT overwrite Methods dictionary!
    # Preserve existing methods from Stage 1 (MeshFaces, St2/3/4 methods, etc.)
    # Instead of InputParam.Methods = {}, only add new methods

    # Utility methods (only if not already set)
    if 'AssembleBasicMatrices' not in InputParam.Methods:
        InputParam.Methods['AssembleBasicMatrices'] = None

    # Per-domain methods (cell arrays indexed by domain)
    # These methods are specific to Stage 2 and must be set
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

    # Variable reduction method (only if not previously set)
    if 'RemoveRedundantVariables' not in InputParam.Methods:
        InputParam.Methods['RemoveRedundantVariables'] = 'remove_redundant_variables'

    logger.info("      Method assignment complete")
    return InputParam


# -----------------------------------------------------------------------------
# Main Stage 2 Orchestrator
# -----------------------------------------------------------------------------

def prepare_model(InputParam: InputParam) -> InputParam:
    """
    Stage 2 main function: Prepare parameters and methods only.
    Mesh generation is separated and called later after PML setup.
    """
    logger.info("  Stage 2: Preparing model for computation...")

    # Stage 2.1: Preprocess parameters
    CompStruct = prepare_model_params(InputParam)

    # Stage 2.2: Assign methods
    CompStruct = prepare_model_methods(CompStruct)

    logger.info("  Stage 2 complete (mesh generation will be after PML setup)")
    return CompStruct


def generate_mesh(CompStruct: InputParam) -> InputParam:
    """
    Stage 2.3: Generate mesh after all domains including PML are configured.
    This is called AFTER setup_additional_domains() in main pipeline.
    """
    logger.info("    Stage 2.3: Generating mesh...")

    # Initialize FEMatrices if not exists
    if not hasattr(CompStruct, 'FEMatrices') or CompStruct.FEMatrices is None:
        CompStruct.FEMatrices = {}

        # Generate mesh using the registered method
        if CompStruct.Methods.get('MeshFaces') is not None:
            try:
                logger.info("      Running mesh generation...")
                mesh_data = CompStruct.Methods['MeshFaces'](CompStruct)
                CompStruct.FEMatrices.update(mesh_data)
                logger.info(f"      Mesh generated: {CompStruct.FEMatrices['MeshNodes'].shape[1]} nodes, "
                            f"{CompStruct.FEMatrices['MeshTri'].shape[1]} elements")
            except Exception as e:
                logger.error(f"Mesh generation failed: {e}")
                raise
        else:
            raise ValueError("MeshFaces method not registered in Stage 1")

    return CompStruct