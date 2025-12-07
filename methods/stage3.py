# methods/stage3.py
"""
===============================================================================
Stage 3: Complete Matrix Assembly
Replicates St3_1_PrepareBasicMatrices_sp_SAFE.m and orchestrates full assembly
===============================================================================
"""

import numpy as np
import scipy.sparse as sp
from typing import Dict, Tuple, Any
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
    assemble_full_matrices_free,
    merge_coincident_dofs,
    remove_redundant_variables,
    prepare_physprop_htti,
    prepare_physprop_fluid
)

logger = logging.getLogger(__name__)


def run_stage3_matrix_assembly(CompStruct: CompStruct, InputParam: InputParam) -> Tuple[CompStruct, Dict]:
    """
    Main Stage 3 orchestrator.
    Replicates the complete logic of St3_1_PrepareBasicMatrices_sp_SAFE.m
    """
    logger.info("=== Stage 3: Matrix Assembly ===")

    # =============================================================================
    # 1. Generate mesh and initialize FEMatrices structure
    # =============================================================================
    logger.info("      Generating mesh...")
    mesh_results = prepare_mesh(CompStruct)

    FEMatrices = {
        'MeshNodes': mesh_results['MeshNodes'],
        'MeshTri': mesh_results['MeshTri'],
        'BoundaryEdges': mesh_results['BoundaryEdges'],
        'MeshProps': mesh_results['MeshProps'],
        'PhysProp': {},
        'DElements': {},
        'DEMeshProps': {},
        'DNodes': {},
        'DNodesRem': {},
        'DNodesComp': {},
        'DTakeFromVarPos': {},
        'DPutToVarPos': {},
        'DZeroVarPos': {},
        'BNodes': {},
        'BNodesFull': {},
        'K1Matrix_d': {},
        'K2Matrix_d': {},
        'K3Matrix_d': {},
        'MMatrix_d': {},
        'PMatrix_d': {},
        'PMatrixD12': {},
        'PMatrixD21': {},
        'ZeroD12': {},
        'ZeroD21': {}
    }

    # =============================================================================
    # 2. Assemble basic matrices (Lx, Ly, Lz, convolutions, etc.)
    # =============================================================================
    logger.info("      Assembling basic matrices...")
    BasicMatrices = assemble_basic_matrices(CompStruct)
    CompStruct.Methods['BasicMatrices'] = BasicMatrices

    # =============================================================================
    # 3. Process each domain sequentially
    # =============================================================================
    n_domains = CompStruct.Data['N_domain']
    for ii_d in range(1, n_domains + 1):
        logger.info(f"      Processing domain {ii_d}/{n_domains}")

        # Extract domain elements
        domain_elements = FEMatrices['MeshTri'][
            FEMatrices['MeshTri'][:, 10] == ii_d
            ].T

        # Store domain data
        FEMatrices['DElements'][ii_d] = domain_elements
        FEMatrices['DEMeshProps'][ii_d] = _extract_domain_mesh_props(FEMatrices, ii_d)

        # Find domain nodes
        DNodesEl = domain_elements[:10, :].flatten()
        FEMatrices['DNodes'][ii_d] = np.unique(DNodesEl)

        # Initialize DOF management arrays
        FEMatrices['DNodesRem'][ii_d] = np.array([], dtype=int)
        FEMatrices['DNodesComp'][ii_d] = np.array([], dtype=int)
        FEMatrices['DTakeFromVarPos'][ii_d] = np.array([], dtype=int)
        FEMatrices['DPutToVarPos'][ii_d] = np.array([], dtype=int)
        FEMatrices['DZeroVarPos'][ii_d] = np.array([], dtype=int)

        # Prepare physical properties
        domain_type = CompStruct.Model['DomainType'][ii_d - 1]
        if domain_type == 'HTTI':
            FEMatrices['PhysProp'][ii_d] = prepare_physprop_htti(CompStruct, ii_d)
        elif domain_type == 'fluid':
            FEMatrices['PhysProp'][ii_d] = prepare_physprop_fluid(CompStruct, ii_d)
        else:
            raise ValueError(f"Unknown domain type: {domain_type}")

        # Compute domain matrices
        if domain_type == 'HTTI':
            FEMatrices = matrices_parts_htti(CompStruct, FEMatrices, ii_d)
        else:
            FEMatrices = matrices_parts_fluid(CompStruct, FEMatrices, ii_d)

    # =============================================================================
    # 4. Process interfaces between domains
    # =============================================================================
    logger.info("      Assembling interfaces...")
    n_interfaces = n_domains - 1

    for ii_int in range(1, n_interfaces + 1):
        logger.debug(f"        Interface {ii_int}: domains {ii_int}-{ii_int + 1}")

        ii_d1 = ii_int
        ii_d2 = ii_int + 1

        # Extract boundary nodes for this interface
        FEMatrices['BNodes'][ii_int] = _extract_boundary_nodes(FEMatrices, ii_int)

        # Determine interface type and assemble
        type1 = CompStruct.Model['DomainType'][ii_d1 - 1]
        type2 = CompStruct.Model['DomainType'][ii_d2 - 1]

        if type1 != type2:  # Fluid-solid
            FEMatrices = ic_matrices_fluid_htti(CompStruct, BasicMatrices, FEMatrices, ii_int, ii_d1, ii_d2)
        else:  # Solid-solid or fluid-fluid
            FEMatrices = ic_matrices_ff_ss(CompStruct, BasicMatrices, FEMatrices, ii_int, ii_d1, ii_d2)

    # =============================================================================
    # 5. Apply outer boundary conditions
    # =============================================================================
    logger.info("      Applying outer boundary conditions...")

    # Get boundary type from model config
    outer_bc = CompStruct.Model.get('OuterBoundaryType', 'rigid')
    ii_int = n_domains  # Interface ID for outer boundary
    ii_d1 = n_domains  # Outer domain

    # Extract outer boundary nodes
    FEMatrices['BNodes'][ii_int] = _extract_boundary_nodes(FEMatrices, ii_int)

    # Store boundary type for assembly
    CompStruct.Model['SubdomainType'] = CompStruct.Model.get('SubdomainType', [])
    while len(CompStruct.Model['SubdomainType']) < n_domains:
        CompStruct.Model['SubdomainType'].append(outer_bc)

    # =============================================================================
    # 6. Global matrix assembly (sequential domain-by-domain)
    # =============================================================================
    logger.info("      Assembling global matrices...")

    FullMatrices = {}

    for ii_d in range(1, n_domains + 1):
        if ii_d == 1:
            # Initialize with first domain
            FullMatrices = {
                'K1': FEMatrices['K1Matrix_d'][ii_d],
                'K2': FEMatrices['K2Matrix_d'][ii_d],
                'K3': FEMatrices['K3Matrix_d'][ii_d],
                'M': FEMatrices['MMatrix_d'][ii_d],
                'P': FEMatrices['PMatrix_d'][ii_d]
            }
            continue

        # Get interface info
        interface_id = ii_d - 1
        subdomain_type = CompStruct.Model['SubdomainType'][interface_id - 1]

        # Select assembler
        assemblers = {
            'fluid-solid': assemble_full_matrices_fs,
            'solid-solid': assemble_full_matrices_ff_ss,
            'fluid-fluid': assemble_full_matrices_ff_ss,
            'rigid': assemble_full_matrices_rigid,
            'free': assemble_full_matrices_free
        }

        assembler = assemblers.get(subdomain_type, assemble_full_matrices_fs)

        # Assemble next domain
        FEMatrices, FullMatrices = assembler(
            CompStruct, BasicMatrices, FEMatrices, FullMatrices,
            interface_id, ii_d - 1, ii_d
        )

    # =============================================================================
    # 7. Merge coincident DOFs for solid-solid interfaces
    # =============================================================================
    logger.debug("      Merging coincident DOFs...")
    FullMatrices = merge_coincident_dofs(FullMatrices, FEMatrices)

    # =============================================================================
    # 8. Remove constrained DOFs (rigid boundaries, etc.)
    # =============================================================================
    logger.info("      Removing redundant variables...")
    FEMatrices, FullMatrices = remove_redundant_variables(
        CompStruct, BasicMatrices, FEMatrices, FullMatrices
    )

    # =============================================================================
    # 9. Store results in CompStruct
    # =============================================================================
    CompStruct.FEMatrices = FEMatrices
    CompStruct.FullMatrices = FullMatrices

    logger.info("      Stage 3 complete")
    return CompStruct, FullMatrices


# =============================================================================
# Helper Functions
# =============================================================================

def _extract_domain_mesh_props(FEMatrices: Dict, domain_id: int) -> Dict[str, np.ndarray]:
    """Extract mesh properties for a specific domain"""
    element_mask = FEMatrices['MeshTri'][:, 10] == domain_id

    return {
        'DS': FEMatrices['MeshProps']['DS'][:, element_mask],
        'delta': FEMatrices['MeshProps']['delta'][:, element_mask],
        'dxL': FEMatrices['MeshProps']['dxL'][:, element_mask],
        'dyL': FEMatrices['MeshProps']['dyL'][:, element_mask],
        'a': FEMatrices['MeshProps']['a'][:, element_mask],
        'b': FEMatrices['MeshProps']['b'][:, element_mask],
        'c': FEMatrices['MeshProps']['c'][:, element_mask],
    }


def _extract_boundary_nodes(FEMatrices: Dict, interface_id: int) -> np.ndarray:
    """
    Extract nodes on boundary interface_id
    Critical: matches MATLAB's boundary edge numbering
    """
    boundary_edges = FEMatrices['BoundaryEdges']
    interface_mask = boundary_edges[2, :] == interface_id

    if not np.any(interface_mask):
        return np.array([], dtype=int)

    edge_nodes = boundary_edges[:2, interface_mask].T.flatten()
    return np.unique(edge_nodes)