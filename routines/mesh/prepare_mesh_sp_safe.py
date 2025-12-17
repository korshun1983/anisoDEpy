# D:\Работа\python\anisoDEpy\stage2\prepare_mesh_sp_safe.py
"""
PrepareMesh_sp_SAFE.py
======================
EXACT MATLAB equivalent of PrepareMesh_sp_SAFE.m
NO CYCLIC IMPORTS!
"""

import numpy as np
from utils import debug_print

from routines.mesh.mesh_generator import (
    prepare_mesh_bh,
    meshfaces,
    add_nodes_cubic,
    find_bedges
)

def prepare_mesh_sp_safe(CompStruct):
    """
    Stage 2: Prepare mesh for computational domains.
    """
    debug_print("  PrepareMesh_sp_SAFE: Preparing mesh...", level=3)

    # Инициализация
    if 'Mesh' not in CompStruct:
        CompStruct['Mesh'] = {}

    if 'Methods' not in CompStruct:
        raise KeyError("CompStruct['Methods'] not found! Run st2_2_prepare_model_methods_sp_safe first.")

    # Stage 2.1: Generate boundary geometry
    debug_print("    Stage 2.1: Boundary geometry...", level=4)
    MeshNodes, BoundaryEdges, DomainFaces, CompStruct = prepare_mesh_bh(CompStruct)

    # Stage 2.2: Generate actual mesh
    debug_print("    Stage 2.2: Mesh generation...", level=4)
    hmax = CompStruct['Mesh'].get('hmax', 0.16)
    MeshNodes, MeshTri, MeshFaceNums, CompStruct = CompStruct['Methods']['MeshFaces'](
        MeshNodes, BoundaryEdges, DomainFaces, CompStruct, hmax
    )
    if MeshNodes.size == 0 or MeshTri.size == 0:
        raise RuntimeError("Mesh generation failed: empty mesh")

    # CRITICAL: Sort triangles by subdomain (MATLAB: MeshTri = (sortrows(MeshTri',4))')
    sort_idx = np.argsort(MeshTri[3, :])
    MeshTri = MeshTri[:, sort_idx]
    MeshFaceNums = MeshFaceNums[sort_idx]

    # Stage 2.3: Add cubic interpolation nodes
    debug_print("    Stage 2.3: Adding cubic nodes...", level=4)
    MeshNodes, MeshTri, MeshProps = add_nodes_cubic(MeshNodes, MeshTri)

    # Find boundary edges
    debug_print("    Stage 2.4: Finding boundary edges...", level=4)
    BoundaryEdges = find_bedges(MeshNodes, MeshTri, CompStruct)

    debug_print(f"    Mesh complete: {MeshNodes.shape[1]} nodes, {MeshTri.shape[1]} elements", level=4)

    if CompStruct.get('Advanced', {}).get('VisualizeMesh', False):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 10))
        plt.triplot(MeshNodes[0, :], MeshNodes[1, :], MeshTri[:3, :].T - 1)
        plt.title(f"Mesh: {MeshNodes.shape[1]} nodes, {MeshTri.shape[1]} elements")
        plt.axis('equal')
        plt.savefig('mesh_debug.png', dpi=150, bbox_inches='tight')
        debug_print(f"  Mesh visualization saved: mesh_debug.png", level=2)

    return MeshNodes, BoundaryEdges, MeshTri, MeshProps, CompStruct