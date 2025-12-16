"""
PrepareMesh_sp_SAFE.py
======================
EXACT MATLAB equivalent of PrepareMesh_sp_SAFE.m
"""

import numpy as np
from utils import debug_print
from .mesh_generator import prepare_mesh_bh, meshfaces, add_nodes_cubic

def prepare_mesh_sp_safe(CompStruct):
    """
    Stage 2: Prepare mesh for computational domains.
    """
    debug_print("  PrepareMesh_sp_SAFE: Preparing mesh...", level=3)

    # Initialize mesh field
    if 'Mesh' not in CompStruct:
        CompStruct['Mesh'] = {}

    # Ensure ext_boundary_shape exists
    if 'ext_boundary_shape' not in CompStruct['Mesh']:
        CompStruct['Mesh']['ext_boundary_shape'] = 'cir'

    # Stage 2.1: Generate boundary geometry (returns DomainFaces!)
    MeshNodes, BoundaryEdges, DomainFaces, CompStruct = prepare_mesh_bh(CompStruct)

    # Stage 2.2: Generate actual mesh
    hmax = CompStruct['Mesh'].get('hmax', 0.16)
    MeshNodes, MeshTri, MeshFaceNums, CompStruct = meshfaces(
        MeshNodes, BoundaryEdges, DomainFaces, CompStruct, hmax
    )

    # CRITICAL: Sort triangles by subdomain (MATLAB: MeshTri = (sortrows(MeshTri',4))')
    sort_idx = np.argsort(MeshTri[3, :])
    MeshTri = MeshTri[:, sort_idx]
    MeshFaceNums = MeshFaceNums[sort_idx]

    # Stage 2.3: Add cubic interpolation nodes
    MeshNodes, MeshTri, MeshProps = add_nodes_cubic(MeshNodes, MeshTri)

    # Find boundary edges (returns array of edge node pairs)
    BoundaryEdges = CompStruct['Methods']['FindBEdges'](MeshNodes, MeshTri, CompStruct)

    debug_print(f"    Mesh complete: {MeshNodes.shape[1]} nodes, {MeshTri.shape[1]} elements", level=4)

    return MeshNodes, BoundaryEdges, MeshTri, MeshProps, CompStruct