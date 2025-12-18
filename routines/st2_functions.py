#!/usr/bin/env python3
"""
Step 2 functions for ANISO_SAFE project.
Preparing data and structure for computations.
"""

import numpy as np
from typing import Dict, Any
import warnings

from config.structures import CompStruct, InputParam
from routines.st1_functions import add_external_domain

# Import mesh and matrices modules
from mesh import (
    PrepareMesh_sp_SAFE, PrepareMeshBH
)
from matrices import (
    # Basis functions
    FindPos_sp_SAFE, L1L2_int_matrix, L1L2L3_int_matrix,
    NL_matrix, dNL_matrices, ConvolveMatrices,
    ConvolveEdgeMatrices, NLEdge_matrix,

    # Assembly
    AssembleBasicMatrices_sp_SAFE, ICMatrices_fluid_HTTI_SAFE_cubic,
    ICMatrices_ff_ss_SAFE_cubic, AssembleFullMatrices_fs_SAFE_cubic,
    AssembleFullMatrices_ff_ss_SAFE_cubic, AssembleFullMatrices_rigid_SAFE_cubic,
    AssembleFullMatrices_free_SAFE_cubic, RemoveRedundantVariables_SAFE,

    # Properties
    PreparePhysProp_fluid_sp_SAFE, PreparePhysProp_HTTI_sp_SAFE,
    getPhysProps_fluid, getPhysProps_HTTI,

    # Element matrices
    dxNL_matrix, dyNL_matrix, MatricesParts_fluid_sp_SAFE_cubic,
    MatricesParts_HTTI_sp_SAFE_cubic, MatricesParts_HTTI_PML_sp_SAFE,
    MatricesParts_HTTI_ABC_sp_SAFE, KM_el_matrix_fluid, KM_el_matrix_HTTI,
    KM_el_matrix_HTTI_PML, KM_el_matrix_HTTI_ABC, KM_el_matrix_HTTI_PML_ABC,
    KM_matrix_fluid, KM_matrix_HTTI, IC_matrix_FS, IC_el_matrix_FS
)


def St2_1_PrepareModelParams_sp_SAFE(comp_struct: CompStruct) -> CompStruct:
    """
    Prepare model parameters for SAFE computation.
    From St2_1_PrepareModelParams_sp_SAFE.m
    """
    print("    Preparing model parameters...")

    # Initialize Misc structure
    comp_struct.Misc = {}

    # Unit conversion factors for frequency
    freq_units = comp_struct.Config.FreqUnits
    if freq_units == 'Hz':
        comp_struct.Misc['F_conv'] = 1.0
    elif freq_units == 'kHz':
        comp_struct.Misc['F_conv'] = 1e3
    else:
        raise ValueError(f"Unsupported frequency units: {freq_units}")

    # Unit conversion factors for slowness
    slo_units = comp_struct.Config.SloUnits
    if slo_units == 'us/m':
        comp_struct.Misc['S_conv'] = 1e3
    elif slo_units == 'us/ft':
        comp_struct.Misc['S_conv'] = 0.3048 * 1e3
    else:
        raise ValueError(f"Unsupported slowness units: {slo_units}")

    # Compute number of computational domains
    comp_struct.Data = {}
    comp_struct.Data['N_domain'] = len(comp_struct.Model.DomainType)

    # Compute array indicating number of variables per layer
    comp_struct.Data['DVarNum'] = np.zeros(comp_struct.Data['N_domain'], dtype=int)
    for ii_d in range(comp_struct.Data['N_domain']):
        domain_type = comp_struct.Model.DomainType[ii_d].lower()
        if domain_type == 'fluid':
            comp_struct.Data['DVarNum'][ii_d] = 1
        elif domain_type in ['htti', 'htr', 'vti', 'vtr']:
            comp_struct.Data['DVarNum'][ii_d] = 3
        else:
            raise ValueError(f"Unsupported domain type: {domain_type}")

    # Compute asymptotes if requested
    check_asymptote = comp_struct.Config.CheckAsymptote
    if check_asymptote == 'yes':
        if hasattr(comp_struct.Methods, 'ComputeAsymptotes') and comp_struct.Methods.ComputeAsymptotes:
            comp_struct.Asymp = comp_struct.Methods.ComputeAsymptotes(comp_struct)
            # Adjust velocity range based on asymptotes
            if hasattr(comp_struct.Asymp, 'V_SH'):
                comp_struct.Advanced.V_min = comp_struct.Asymp.V_SH
            if hasattr(comp_struct.Asymp, 'V_qP'):
                comp_struct.Advanced.V_max = comp_struct.Asymp.V_qP
        else:
            warnings.warn("ComputeAsymptotes method not available, skipping asymptote computation")

    return comp_struct


def St2_2_PrepareModelMethods_sp_SAFE(comp_struct: CompStruct) -> CompStruct:
    """
    Assign methods for SAFE computation.
    From St2_2_PrepareModelMethods_sp_SAFE.m
    """
    print("    Preparing model methods...")

    # Assign utility methods
    comp_struct.Methods.chebdif = None  # Placeholder

    comp_struct.Methods.em_tensor_VTI = None
    comp_struct.Methods.rot_c_ij = None
    comp_struct.Methods.rot_matrix = None

    # Assign Step 2-4 method placeholders
    comp_struct.Methods.St2_2_PrepareModelParams = St2_2_PrepareModelMethods_sp_SAFE
    comp_struct.Methods.St3_ProblemFormulation = None
    comp_struct.Methods.St3_1_PrepareBasicMatrices = None
    comp_struct.Methods.St4_ComputeSolution = None

    # Assign mesh methods
    comp_struct.Methods.PrepareMesh = PrepareMesh_sp_SAFE
    comp_struct.Methods.PrepareMeshBH = PrepareMeshBH
    # comp_struct.Methods.FindBEdges = FindBEdges
    # comp_struct.Methods.MakeContBEdges = MakeContBEdges
    # comp_struct.Methods.FindEdgeOrient = FindEdgeOrient
    # comp_struct.Methods.AddNodesCubic = AddNodesCubic

    # Assign basis function methods
    comp_struct.Methods.FindPos = FindPos_sp_SAFE
    comp_struct.Methods.L1L2_int_matrix = L1L2_int_matrix
    comp_struct.Methods.L1L2L3_int_matrix = L1L2L3_int_matrix
    comp_struct.Methods.NL_matrix = NL_matrix
    comp_struct.Methods.dNL_matrices = dNL_matrices
    comp_struct.Methods.ConvolveMatrices = ConvolveMatrices
    comp_struct.Methods.ConvolveEdgeMatrices = ConvolveEdgeMatrices
    comp_struct.Methods.NLEdge_matrix = NLEdge_matrix

    # Assign basic matrix assembly
    comp_struct.Methods.AssembleBasicMatrices = AssembleBasicMatrices_sp_SAFE

    # Assign physical property methods per domain
    comp_struct.Methods.PreparePhysProp = {}
    for ii_d in range(comp_struct.Data['N_domain']):
        domain_type = comp_struct.Model.DomainType[ii_d].lower()
        if domain_type == 'fluid':
            comp_struct.Methods.PreparePhysProp[ii_d] = PreparePhysProp_fluid_sp_SAFE
        elif domain_type in ['htti', 'htr', 'vti', 'vtr']:
            comp_struct.Methods.PreparePhysProp[ii_d] = PreparePhysProp_HTTI_sp_SAFE
        else:
            raise ValueError(f"Unsupported domain type: {domain_type}")

    # Assign element matrices methods
    comp_struct.Methods.dxNL_matrix = dxNL_matrix
    comp_struct.Methods.dyNL_matrix = dyNL_matrix

    comp_struct.Methods.MatricesParts_sp_SAFE = {}
    comp_struct.Methods.MatricesPartsPML_sp_SAFE = {}
    comp_struct.Methods.MatricesPartsABC_sp_SAFE = {}
    comp_struct.Methods.getPhysProps = {}
    comp_struct.Methods.KM_matrix = {}
    comp_struct.Methods.KM_el_matrix = {}

    for ii_d in range(comp_struct.Data['N_domain']):
        domain_type = comp_struct.Model.DomainType[ii_d].lower()
        if domain_type == 'fluid':
            comp_struct.Methods.MatricesParts_sp_SAFE[ii_d] = MatricesParts_fluid_sp_SAFE_cubic
            comp_struct.Methods.getPhysProps[ii_d] = getPhysProps_fluid
            comp_struct.Methods.KM_matrix[ii_d] = KM_matrix_fluid
            comp_struct.Methods.KM_el_matrix[ii_d] = KM_el_matrix_fluid
        elif domain_type in ['htti', 'htr', 'vti', 'vtr']:
            comp_struct.Methods.MatricesParts_sp_SAFE[ii_d] = MatricesParts_HTTI_sp_SAFE_cubic
            comp_struct.Methods.MatricesPartsPML_sp_SAFE[ii_d] = MatricesParts_HTTI_PML_sp_SAFE
            comp_struct.Methods.MatricesPartsABC_sp_SAFE[ii_d] = MatricesParts_HTTI_ABC_sp_SAFE
            comp_struct.Methods.getPhysProps[ii_d] = getPhysProps_HTTI
            comp_struct.Methods.KM_matrix[ii_d] = KM_matrix_HTTI
            comp_struct.Methods.KM_el_matrix[ii_d] = KM_el_matrix_HTTI
        else:
            raise ValueError(f"Unsupported domain type: {domain_type}")

    # Handle additional domain (PML/ABC)
    if comp_struct.Model.AddDomain_Exist == 'yes':
        n_domain = comp_struct.Data['N_domain'] - 1
        add_domain_type = comp_struct.Model.AddDomainType.lower()

        if add_domain_type == 'pml':
            comp_struct.Methods.KM_el_matrix[n_domain] = KM_el_matrix_HTTI_PML
        elif add_domain_type == 'abc':
            comp_struct.Methods.KM_el_matrix[n_domain] = KM_el_matrix_HTTI_ABC
        elif add_domain_type == 'pml+abc':
            comp_struct.Methods.KM_el_matrix[n_domain] = KM_el_matrix_HTTI_PML_ABC
        else:
            raise ValueError(f"External Domain Type '{add_domain_type}' is not set correctly!")

    # Assign interface condition methods
    comp_struct.Methods.IC_Matrices_sp_SAFE = {}
    comp_struct.Methods.IC_matrix = {}
    comp_struct.Methods.IC_el_matrix = {}
    comp_struct.Methods.AssembleFullMatrices = {}

    for ii_int in range(comp_struct.Data['N_domain'] - 1):
        type1 = comp_struct.Model.DomainType[ii_int].lower()
        type2 = comp_struct.Model.DomainType[ii_int + 1].lower()

        if (type1 == 'fluid' and type2 in ['htti', 'htr', 'vti', 'vtr']) or \
                (type1 in ['htti', 'htr', 'vti', 'vtr'] and type2 == 'fluid'):
            comp_struct.Methods.IC_Matrices_sp_SAFE[ii_int] = ICMatrices_fluid_HTTI_SAFE_cubic
            comp_struct.Methods.IC_matrix[ii_int] = IC_matrix_FS
            comp_struct.Methods.IC_el_matrix[ii_int] = IC_el_matrix_FS
            comp_struct.Methods.AssembleFullMatrices[ii_int] = AssembleFullMatrices_fs_SAFE_cubic
        elif (type1 == 'fluid' and type2 == 'fluid') or \
                (type1 in ['htti', 'htr', 'vti', 'vtr'] and type2 in ['htti', 'htr', 'vti', 'vtr']):
            comp_struct.Methods.IC_Matrices_sp_SAFE[ii_int] = ICMatrices_ff_ss_SAFE_cubic
            comp_struct.Methods.AssembleFullMatrices[ii_int] = AssembleFullMatrices_ff_ss_SAFE_cubic
        else:
            raise ValueError(f"Unsupported interface: {type1}-{type2}")

    # Assign outer boundary condition method
    ii_int = comp_struct.Data['N_domain'] - 1
    bc_type = comp_struct.Model.BCType[ii_int].lower()

    if bc_type == 'rigid':
        comp_struct.Methods.AssembleFullMatrices[ii_int] = AssembleFullMatrices_rigid_SAFE_cubic
    elif bc_type == 'free':
        comp_struct.Methods.AssembleFullMatrices[ii_int] = AssembleFullMatrices_free_SAFE_cubic
    else:
        raise ValueError(f"Unsupported boundary condition: {bc_type}")

    # Redundant variable removal method
    comp_struct.Methods.RemoveRedundantVariables = RemoveRedundantVariables_SAFE

    return comp_struct


def St2_PrepareModel_sp_SAFE(input_param: InputParam) -> CompStruct:
    """
    Main Step 2 function - prepares model for computation.
    From St2_PrepareModel_sp_SAFE.m
    """
    print('\n=== Step 2: Preparing Model ===\n')

    # Retain all necessary information from Input structure
    comp_struct = CompStruct(
        Config=input_param.Config,
        Model=input_param.Model,
        Advanced=input_param.Advanced,
        Methods=input_param.Methods,
        f_grid=input_param.Model.f_array,
        Data={},
        Misc={}
    )

    # Prepare input data for computations
    print('  Preparing model parameters...')
    comp_struct = St2_1_PrepareModelParams_sp_SAFE(comp_struct)

    # Assign methods for solving the problem
    print('  Assigning computation methods...')
    comp_struct = St2_2_PrepareModelMethods_sp_SAFE(comp_struct)

    print('\n=== Step 2 Completed ===\n')

    # Summary
    print(f"  Domains processed: {comp_struct.Data['N_domain']}")
    print(f"  Variables per domain: {comp_struct.Data['DVarNum']}")
    print(f"  Unit conversions: F={comp_struct.Misc['F_conv']}, S={comp_struct.Misc['S_conv']}\n")

    return comp_struct