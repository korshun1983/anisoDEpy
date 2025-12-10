# methods/stage3.py
"""
===============================================================================
Stage 3: Complete Matrix Assembly - Main Orchestrator
Replicates St3_1_PrepareBasicMatrices_sp_SAFE.m workflow
===============================================================================
Добавлено:
- Улучшенная валидация доменных маркеров
- Проверка размерностей после сборки
===============================================================================
"""

import numpy as np
from typing import Dict, Tuple, Any
import logging

from routines.matrix_assembly import (
    assemble_basic_matrices,
    prepare_physprop_htti,
    prepare_physprop_fluid,
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
    dispatch_method
)

logger = logging.getLogger(__name__)


def run_stage3_matrix_assembly(CompStruct: Any) -> Tuple[Any, Dict]:
    """Main Stage 3 orchestrator - replicates full MATLAB workflow"""
    logger.info("=== Stage 3: Matrix Assembly ===")

    # 1. Initialize FEMatrices structure from mesh
    logger.info("      Initializing FEMatrices from mesh...")

    # ВАЛИДАЦИЯ: проверяем наличие необходимых полей
    required_mesh_keys = ['MeshNodes', 'BoundaryEdges', 'MeshTri', 'MeshProps']
    for key in required_mesh_keys:
        if key not in CompStruct.FEMatrices:
            raise KeyError(f"Missing required mesh key: {key}")

    FEMatrices = {
        'MeshNodes': CompStruct.FEMatrices['MeshNodes'],
        'BoundaryEdges': CompStruct.FEMatrices['BoundaryEdges'],
        'MeshTri': CompStruct.FEMatrices['MeshTri'],
        'MeshProps': CompStruct.FEMatrices['MeshProps'],
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

    CompStruct.FEMatrices = FEMatrices

    logger.info(f"        MeshTri shape: {CompStruct.FEMatrices['MeshTri'].shape}")

    # ВАЛИДАЦИЯ: проверяем доменные маркеры
    domain_markers = CompStruct.FEMatrices['MeshTri'][-1, :]
    unique_markers = np.unique(domain_markers)
    expected_domains = CompStruct.Data.N_domain

    logger.info(f"        Unique domain markers: {unique_markers}")
    logger.info(f"        Expected domains: 1..{expected_domains}")

    if not all(1 <= m <= expected_domains for m in unique_markers):
        logger.warning(f"Domain markers out of range! Got {unique_markers}, expected 1..{expected_domains}")

    # 2. Assemble basic matrices
    logger.info("      Assembling basic matrices...")
    BasicMatrices = assemble_basic_matrices(CompStruct)
    CompStruct.Methods['BasicMatrices'] = BasicMatrices

    # 3. Process each domain sequentially
    n_domains = CompStruct.Data.N_domain

    for ii_d in range(1, n_domains + 1):
        logger.info(f"      Processing domain {ii_d}/{n_domains}")

        # Extract domain elements
        mask = CompStruct.FEMatrices['MeshTri'][-1, :] == ii_d

        if np.sum(mask) == 0:
            logger.error(f"No elements found for domain {ii_d}")
            logger.error(f"Domain markers in MeshTri: {np.unique(CompStruct.FEMatrices['MeshTri'][-1, :])}")
            raise ValueError(f"Domain {ii_d} has no elements - check mesh generation and domain markers")

        domain_elements = CompStruct.FEMatrices['MeshTri'][:-1, mask].T

        FEMatrices['DElements'][ii_d] = domain_elements
        FEMatrices['DEMeshProps'][ii_d] = _extract_domain_mesh_props(CompStruct, ii_d)

        # Find domain nodes
        DNodesEl = domain_elements.flatten()
        FEMatrices['DNodes'][ii_d] = np.unique(DNodesEl)
        CompStruct.FEMatrices['DNodes'] = FEMatrices['DNodes']

        # Initialize DOF management
        FEMatrices['DNodesRem'][ii_d] = np.array([], dtype=int)
        FEMatrices['DNodesComp'][ii_d] = np.array([], dtype=int)
        FEMatrices['DTakeFromVarPos'][ii_d] = np.array([], dtype=int)
        FEMatrices['DPutToVarPos'][ii_d] = np.array([], dtype=int)
        FEMatrices['DZeroVarPos'][ii_d] = np.array([], dtype=int)

        # Prepare physical properties
        method_name = CompStruct.Methods['PreparePhysProp'][ii_d - 1]
        method_func = dispatch_method(method_name)
        FEMatrices['PhysProp'][ii_d] = method_func(CompStruct, ii_d)

        # Compute domain matrices
        domain_type = CompStruct.Model['DomainType'][ii_d - 1].lower()
        if domain_type == 'htti':
            FEMatrices = matrices_parts_htti(CompStruct, FEMatrices, ii_d)
        elif domain_type == 'fluid':
            FEMatrices = matrices_parts_fluid(CompStruct, FEMatrices, ii_d)
        else:
            raise ValueError(f"Unknown domain type: {domain_type}")

    # 4. Process interfaces between domains
    logger.info("      Assembling interfaces...")
    n_interfaces = n_domains - 1

    for ii_int in range(1, n_interfaces + 1):
        logger.debug(f"        Interface {ii_int}: domains {ii_int}-{ii_int + 1}")

        ii_d1 = ii_int
        ii_d2 = ii_int + 1

        FEMatrices['BNodes'][ii_int] = _extract_boundary_nodes(FEMatrices, ii_int)

        type1 = CompStruct.Model['DomainType'][ii_d1 - 1].lower()
        type2 = CompStruct.Model['DomainType'][ii_d2 - 1].lower()

        if type1 != type2:
            FEMatrices = ic_matrices_fluid_htti(CompStruct, FEMatrices, ii_int, ii_d1, ii_d2)
        else:
            FEMatrices = ic_matrices_ff_ss(CompStruct, FEMatrices, ii_int, ii_d1, ii_d2)

    # 5. Apply outer boundary conditions
    logger.info("      Applying outer boundary conditions...")

    outer_bc = CompStruct.Model['BCType'][-1].lower()
    ii_int = n_domains
    ii_d1 = n_domains

    FEMatrices['BNodes'][ii_int] = _extract_boundary_nodes(FEMatrices, ii_int)

    # 6. Global matrix assembly (sequential domain-by-domain)
    logger.info("      Assembling global matrices...")

    FullMatrices = {}

    for ii_d in range(1, n_domains + 1):
        if ii_d == 1:
            FullMatrices = {
                'K1': FEMatrices['K1Matrix_d'][ii_d],
                'K2': FEMatrices['K2Matrix_d'][ii_d],
                'K3': FEMatrices['K3Matrix_d'][ii_d],
                'M': FEMatrices['MMatrix_d'][ii_d],
                'P': FEMatrices['PMatrix_d'][ii_d]
            }
            continue

        interface_id = ii_d - 1

        if interface_id < n_interfaces:
            type1 = CompStruct.Model.DomainType[ii_d - 2].lower()
            type2 = CompStruct.Model.DomainType[ii_d - 1].lower()

            if type1 != type2:
                assembler = assemble_full_matrices_fs
            else:
                assembler = assemble_full_matrices_ff_ss
        else:
            if outer_bc == 'rigid':
                assembler = assemble_full_matrices_rigid
            elif outer_bc == 'free':
                assembler = assemble_full_matrices_free
            else:
                raise ValueError(f"Unsupported outer BC: {outer_bc}")

        FEMatrices, FullMatrices = assembler(
            CompStruct, FEMatrices, FullMatrices, interface_id, ii_d - 1, ii_d
        )

    # 7. Merge coincident DOFs for solid-solid interfaces
    logger.debug("      Merging coincident DOFs...")
    FullMatrices = merge_coincident_dofs(FullMatrices, FEMatrices)

    # 8. Remove constrained DOFs (rigid boundaries, etc.)
    logger.info("      Removing redundant variables...")
    FEMatrices, FullMatrices = remove_redundant_variables(
        CompStruct, FEMatrices, FullMatrices
    )

    # ВАЛИДАЦИЯ: проверяем финальные размерности
    final_dofs = FullMatrices['M'].shape[0]
    logger.info(f"      Final system size: {final_dofs} DOFs")

    if final_dofs == 0:
        raise RuntimeError("All DOFs were removed! Check boundary conditions.")

    # 9. Store results back in CompStruct
    CompStruct.FEMatrices = FEMatrices
    CompStruct.FullMatrices = FullMatrices

    logger.info("      Stage 3 complete")
    return CompStruct, FullMatrices


# =============================================================================
# Helper Functions
# =============================================================================

def _extract_domain_mesh_props(CompStruct: Any, domain_id: int) -> Dict[str, np.ndarray]:
    """Extract mesh properties for a specific domain"""
    element_mask = CompStruct.FEMatrices['MeshTri'][-1, :] == domain_id

    return {
        'DS': CompStruct.FEMatrices['MeshProps']['DS'][:, element_mask],
        'delta': CompStruct.FEMatrices['MeshProps']['delta'][:, element_mask],
        'dxL': CompStruct.FEMatrices['MeshProps']['dxL'][:, element_mask],
        'dyL': CompStruct.FEMatrices['MeshProps']['dyL'][:, element_mask],
        'a': CompStruct.FEMatrices['MeshProps']['a'][:, element_mask],
        'b': CompStruct.FEMatrices['MeshProps']['b'][:, element_mask],
        'c': CompStruct.FEMatrices['MeshProps']['c'][:, element_mask],
    }


def _extract_boundary_nodes(FEMatrices: Dict, interface_id: int) -> np.ndarray:
    """Extract nodes on a specific boundary/interface"""
    boundary_edges = FEMatrices['BoundaryEdges']
    interface_mask = boundary_edges[2, :] == interface_id

    if not np.any(interface_mask):
        logger.warning(f"No boundary edges found for interface {interface_id}")
        return np.array([], dtype=int)

    edge_nodes = boundary_edges[:2, interface_mask].T.flatten()
    return np.unique(edge_nodes)