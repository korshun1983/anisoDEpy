"""
St2_2_PrepareModelMethods_sp_safe.py
====================================
NO CYCLIC IMPORTS - RELATIVE IMPORTS INSIDE PACKAGE
"""
from utils import debug_print

from routines.mesh.prepare_mesh_sp_safe import prepare_mesh_sp_safe


from routines.mesh.mesh_generator import (
    prepare_mesh_bh,
    meshfaces,
    add_nodes_cubic,
    find_bedges,
    find_edge_orient,
    make_cont_bedges
)

from routines.physics.prepare_physprop_fluid_sp_safe import prepare_physprop_fluid_sp_safe
from routines.physics.prepare_physprop_htti_sp_safe import prepare_physprop_htti_sp_safe


def st2_2_prepare_model_methods_sp_safe(CompStruct):
    """Assign methods for mesh generation."""
    debug_print("  St2_2: Assigning computational methods...", level=3)

    if 'Methods' not in CompStruct:
        CompStruct['Methods'] = {}

    Methods = CompStruct['Methods']
    Methods['PrepareMesh'] = prepare_mesh_sp_safe
    Methods['PrepareMeshBH'] = prepare_mesh_bh
    Methods['MeshFaces'] = meshfaces
    Methods['AddNodesCubic'] = add_nodes_cubic
    Methods['FindBEdges'] = find_bedges
    Methods['MakeContBEdges'] = make_cont_bedges
    Methods['FindEdgeOrient'] = find_edge_orient

    n_domain = CompStruct['Data']['N_domain']
    Methods['PreparePhysProp'] = [None] * n_domain
    for ii_d in range(n_domain):
        domain_type = CompStruct['Model']['DomainType'][ii_d]
        Methods['PreparePhysProp'][ii_d] = (
            prepare_physprop_fluid_sp_safe if domain_type == 'fluid'
            else prepare_physprop_htti_sp_safe
        )

    debug_print("  St2_2: Methods assigned", level=3)
    return CompStruct