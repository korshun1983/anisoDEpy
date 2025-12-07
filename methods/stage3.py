# methods/stage3.py
"""
===============================================================================
Stage 3: Basic Matrices Preparation
Replicates St3_1_PrepareBasicMatrices_sp_SAFE.m
Core FEM matrix assembly for SAFE method
===============================================================================
"""

import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Tuple, Any
from scipy.io import savemat
from pathlib import Path
import logging

from core.config import CompStruct, InputParam
from routines.meshgen import prepare_mesh
from routines.matrix_assembly import (
    assemble_basic_matrices,
    matrices_parts_htti,
    matrices_parts_fluid,
    ic_matrices_fluid_htti,
    ic_matrices_ff_ss,
    assemble_full_matrices_fs,
    assemble_full_matrices_ff_ss,
    assemble_full_matrices_rigid,
    remove_redundant_variables
)

logger = logging.getLogger(__name__)


def prepare_basic_matrices(CompStruct: CompStruct, InputParam: InputParam) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Main matrix assembly orchestrator.
    Replicates St3_1_PrepareBasicMatrices_sp_SAFE.m

    Returns:
        tuple: (BasicMatrices, FEMatrices, FullMatrices, CompStruct)
    """
    logger.info("      Stage 3.1: Preparing basic matrices...")

    # =============================================================================
    # 1. Assemble basic matrices (Lx, Ly, Lz operators)
    # =============================================================================
    BasicMatrices = assemble_basic_matrices(CompStruct)

    # =============================================================================
    # 2. Update domain geometry based on wavelength scaling
    # =============================================================================
    BasicMatrices['f_grid'] = CompStruct.f_grid

    # Current frequency index
    current_freq = CompStruct.f_grid[CompStruct.if_grid - 1]

    # Reference velocity for wavelength scaling
    var_vel = (CompStruct.Asymp['V_SH'] * 1e3) / (current_freq * CompStruct.Misc['F_conv'])

    num_layers = len(InputParam.Model['DomainType'])
    nl = num_layers

    # Apply scaling logic based on domain configuration
    CompStruct = _scale_domain_geometry(CompStruct, InputParam, var_vel, nl)

    # =============================================================================
    # 3. Generate mesh
    # =============================================================================
    mesh_results = prepare_mesh(CompStruct)
    FEMatrices = {}
    FEMatrices.update(mesh_results)

    logger.info(f"        Mesh: {len(FEMatrices['MeshNodes'])} nodes, "
                f"{len(FEMatrices['MeshTri'])} elements")

    # =============================================================================
    # 4. Process each domain
    # =============================================================================
    FEMatrices['PhysProp'] = {}
    FEMatrices['DElements'] = {}
    FEMatrices['DEMeshProps'] = {}
    FEMatrices['DNodes'] = {}
    FEMatrices['DNodesRem'] = {}
    FEMatrices['DNodesComp'] = {}
    FEMatrices['DTakeFromVarPos'] = {}
    FEMatrices['DPutToVarPos'] = {}
    FEMatrices['DZeroVarPos'] = {}

    FEMatrices['K1Matrix_d'] = {}
    FEMatrices['K2Matrix_d'] = {}
    FEMatrices['K3Matrix_d'] = {}
    FEMatrices['MMatrix_d'] = {}
    FEMatrices['PMatrix_d'] = {}

    for ii_d in range(1, CompStruct.Data['N_domain'] + 1):
        logger.info(f"        Processing domain {ii_d}/{CompStruct.Data['N_domain']}")

        # Prepare physical properties for this domain
        FEMatrices['PhysProp'][ii_d] = _prepare_physprop(CompStruct, ii_d)

        # Extract elements belonging to this domain
        FEMatrices['DElements'][ii_d] = FEMatrices['MeshTri'][
            np.isin(FEMatrices['MeshTri'][:, 10], ii_d)
        ]

        # Extract mesh properties for this domain
        FEMatrices['DEMeshProps'][ii_d] = _extract_domain_mesh_props(FEMatrices, ii_d)

        # Find nodes belonging to this domain
        DNodesEl = FEMatrices['DElements'][ii_d][:, :10].flatten()
        FEMatrices['DNodes'][ii_d] = np.unique(DNodesEl)

        # Initialize position arrays
        FEMatrices['DNodesRem'][ii_d] = np.array([], dtype=int)
        FEMatrices['DNodesComp'][ii_d] = np.array([], dtype=int)
        FEMatrices['DTakeFromVarPos'][ii_d] = np.array([], dtype=int)
        FEMatrices['DPutToVarPos'][ii_d] = np.array([], dtype=int)
        FEMatrices['DZeroVarPos'][ii_d] = np.array([], dtype=int)

        # Compute matrix blocks for this domain
        FEMatrices = _compute_domain_matrices(CompStruct, FEMatrices, ii_d)

    # =============================================================================
    # 5. Process interfaces between domains
    # =============================================================================
    FEMatrices['BNodes'] = {}
    n_interfaces = CompStruct.Data['N_domain'] - 1

    for ii_int in range(1, n_interfaces + 1):
        logger.info(f"        Processing interface {ii_int}/{n_interfaces}")

        # Find boundary nodes
        FEMatrices['BNodes'][ii_int] = _extract_boundary_nodes(FEMatrices, ii_int)

        # Assemble interface matrices
        ii_d1 = ii_int
        ii_d2 = ii_int + 1

        FEMatrices = _assemble_interface_matrices(
            CompStruct, BasicMatrices, FEMatrices, ii_int, ii_d1, ii_d2
        )

    # =============================================================================
    # 6. Apply outer boundary conditions
    # =============================================================================
    logger.info("        Applying outer boundary conditions...")

    ii_int = CompStruct.Data['N_domain']
    ii_d1 = ii_int
    ii_d2 = ii_int + 1

    # Extract outer boundary nodes
    FEMatrices['BNodes'][ii_int] = _extract_boundary_nodes(FEMatrices, ii_int)

    # Assemble boundary matrices
    FEMatrices, FullMatrices = _assemble_boundary_matrices(
        CompStruct, BasicMatrices, FEMatrices, ii_int, ii_d1, ii_d2
    )

    # =============================================================================
    # 7. Remove redundant variables (e.g., rigid BC nodes)
    # =============================================================================
    logger.info("        Removing redundant variables...")

    FEMatrices, FullMatrices = remove_redundant_variables(
        CompStruct, BasicMatrices, FEMatrices, FullMatrices
    )

    logger.info("      Stage 3.1 complete")

    # Return empty dicts for BasicMatrices and FullMatrices for now
    # These will be populated as we implement more routines
    return {}, FEMatrices, {}, CompStruct


def _scale_domain_geometry(CompStruct: CompStruct, InputParam: InputParam,
                           var_vel: float, nl: int) -> CompStruct:
    """
    Replicates geometry scaling logic for PML/ABC layers
    """
    # Case 1: Wavelength scaling with additional domain
    if (CompStruct.Model['LDomain_in_LSH'] == 'yes' and
            CompStruct.Model['AddDomain_Exist'] == 'yes'):

        CompStruct.Model['AddDomainL_m'] = CompStruct.Model['AddDomainL'] * var_vel

        if CompStruct.Model['AddDomainLoc'] == 'ext':
            # Update second-to-last domain
            CompStruct.Model['DomainRx'][nl - 1] = (
                    InputParam.Model['DomainRx'][nl - 2] +
                    InputParam.Model['DomainRx'][nl - 1] * var_vel
            )
            CompStruct.Model['DomainRy'][nl - 1] = (
                    InputParam.Model['DomainRy'][nl - 2] +
                    InputParam.Model['DomainRy'][nl - 1] * var_vel
            )
            # Add external layer
            CompStruct.Model['DomainRx'][nl] = (
                    CompStruct.Model['DomainRx'][nl - 1] + CompStruct.Model['AddDomainL_m']
            )
            CompStruct.Model['DomainRy'][nl] = (
                    CompStruct.Model['DomainRy'][nl - 1] + CompStruct.Model['AddDomainL_m']
            )

        elif CompStruct.Model['AddDomainLoc'] == 'int':
            CompStruct.Model['DomainRx'][nl] = (
                    CompStruct.Model['DomainRx'][nl - 1] +
                    InputParam.Model['DomainRx'][nl] * var_vel
            )
            CompStruct.Model['DomainRy'][nl] = (
                    CompStruct.Model['DomainRy'][nl - 1] +
                    InputParam.Model['DomainRy'][nl] * var_vel
            )

    # Case 2: Meter scaling with additional domain
    elif (CompStruct.Model['LDomain_in_LSH'] == 'none' and
          CompStruct.Model['AddDomain_Exist'] == 'yes'):

        CompStruct.Model['AddDomainL_m'] = CompStruct.Model['AddDomainL']

        if CompStruct.Model['AddDomainLoc'] == 'ext':
            CompStruct.Model['DomainRx'][nl] = (
                    CompStruct.Model['DomainRx'][nl - 1] + CompStruct.Model['AddDomainL_m']
            )
            CompStruct.Model['DomainRy'][nl] = (
                    CompStruct.Model['DomainRy'][nl - 1] + CompStruct.Model['AddDomainL_m']
            )

        elif CompStruct.Model['AddDomainLoc'] == 'int':
            varx = CompStruct.Model['DomainRx'][nl] - CompStruct.Model['AddDomainL_m']
            vary = CompStruct.Model['DomainRy'][nl] - CompStruct.Model['AddDomainL_m']
            if varx <= CompStruct.Model['DomainRx'][nl - 1] or vary <= CompStruct.Model['DomainRy'][nl - 1]:
                raise ValueError("Error in specifying model geometry for internal domain!")

    # Case 3: Wavelength scaling without additional domain
    elif (CompStruct.Model['LDomain_in_LSH'] == 'yes' and
          CompStruct.Model['AddDomain_Exist'] == 'no'):

        CompStruct.Model['DomainRx'][nl] = (
                InputParam.Model['DomainRx'][nl - 1] +
                InputParam.Model['DomainRx'][nl] * var_vel
        )
        CompStruct.Model['DomainRy'][nl] = (
                InputParam.Model['DomainRy'][nl - 1] +
                InputParam.Model['DomainRy'][nl] * var_vel
        )

    # Case 4: Meter scaling without additional domain (no changes)
    else:
        pass

    return CompStruct


def _prepare_physprop(CompStruct: CompStruct, domain_id: int) -> Dict[str, Any]:
    """
    Prepare physical properties for a domain
    """
    method_name = CompStruct.Methods['PreparePhysProp'][domain_id]

    if method_name == 'prepare_physprop_fluid':
        from routines.physics import prepare_physprop_fluid
        return prepare_physprop_fluid(CompStruct, domain_id)
    elif method_name == 'prepare_physprop_htti':
        from routines.physics import prepare_physprop_htti
        return prepare_physprop_htti(CompStruct, domain_id)
    else:
        raise ValueError(f"Unknown PreparePhysProp method: {method_name}")


def _extract_domain_mesh_props(FEMatrices: Dict, domain_id: int) -> Dict[str, np.ndarray]:
    """
    Extract mesh properties for a specific domain
    """
    domain_elements = FEMatrices['DElements'][domain_id]
    element_indices = np.isin(FEMatrices['MeshTri'][:, 10], domain_id)

    props = {
        'DS': FEMatrices['MeshProps']['DS'][:, element_indices],
        'delta': FEMatrices['MeshProps']['delta'][:, element_indices],
        'a': FEMatrices['MeshProps']['a'][:, element_indices],
        'b': FEMatrices['MeshProps']['b'][:, element_indices],
        'c': FEMatrices['MeshProps']['c'][:, element_indices],
    }
    return props


def _compute_domain_matrices(CompStruct: CompStruct, FEMatrices: Dict, domain_id: int) -> Dict:
    """
    Compute stiffness/mass matrix blocks for a domain
    """
    method_name = CompStruct.Methods['MatricesParts_sp_SAFE'][domain_id]

    if method_name == 'matrices_parts_htti':
        FEMatrices = matrices_parts_htti(CompStruct, FEMatrices, domain_id)
    elif method_name == 'matrices_parts_fluid':
        FEMatrices = matrices_parts_fluid(CompStruct, FEMatrices, domain_id)
    else:
        raise ValueError(f"Unknown MatricesParts method: {method_name}")

    return FEMatrices


def _extract_boundary_nodes(FEMatrices: Dict, interface_id: int) -> np.ndarray:
    """
    Extract nodes on a specific boundary interface
    """
    boundary_edges = FEMatrices['BoundaryEdges']
    interface_mask = boundary_edges[:, 2] == interface_id
    boundary_edge_nodes = boundary_edges[interface_mask, :2]

    if boundary_edge_nodes.size == 0:
        return np.array([], dtype=int)

    return np.unique(boundary_edge_nodes.flatten())


def _assemble_interface_matrices(CompStruct: CompStruct, BasicMatrices: Dict,
                                 FEMatrices: Dict, interface_id: int,
                                 d1: int, d2: int) -> Dict:
    """
    Assemble interface matrices between two domains
    """
    method_name = CompStruct.Methods['IC_Matrices_sp_SAFE'][interface_id]

    if method_name == 'ic_matrices_fluid_htti':
        FEMatrices = ic_matrices_fluid_htti(CompStruct, BasicMatrices, FEMatrices, interface_id, d1, d2)
    elif method_name == 'ic_matrices_ff_ss':
        FEMatrices = ic_matrices_ff_ss(CompStruct, BasicMatrices, FEMatrices, interface_id, d1, d2)
    else:
        raise ValueError(f"Unknown IC_Matrices method: {method_name}")

    return FEMatrices


def _assemble_boundary_matrices(CompStruct: CompStruct, BasicMatrices: Dict,
                                FEMatrices: Dict, interface_id: int,
                                d1: int, d2: int) -> Tuple[Dict, Dict]:
    """
    Apply outer boundary conditions
    """
    method_name = CompStruct.Methods['AssembleFullMatrices'][interface_id]

    FullMatrices = {}

    if method_name == 'assemble_full_matrices_fs':
        FEMatrices, FullMatrices = assemble_full_matrices_fs(
            CompStruct, BasicMatrices, FEMatrices, FullMatrices, interface_id, d1, d2
        )
    elif method_name == 'assemble_full_matrices_ff_ss':
        FEMatrices, FullMatrices = assemble_full_matrices_ff_ss(
            CompStruct, BasicMatrices, FEMatrices, FullMatrices, interface_id, d1, d2
        )
    elif method_name == 'assemble_full_matrices_rigid':
        FEMatrices, FullMatrices = assemble_full_matrices_rigid(
            CompStruct, BasicMatrices, FEMatrices, FullMatrices, interface_id, d1, d2
        )
    else:
        raise ValueError(f"Unknown AssembleFullMatrices method: {method_name}")

    return FEMatrices, FullMatrices