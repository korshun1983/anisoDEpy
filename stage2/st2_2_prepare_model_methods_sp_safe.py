"""
St2_2_PrepareModelMethods_sp_SAFE.py
====================================
EXACT MATLAB equivalent - MESH GENERATION ONLY
"""

import os
from utils import debug_print


def st2_2_prepare_model_methods_sp_safe(CompStruct):
    """
    Assign methods for mesh generation.
    """
    debug_print("  St2_2: Assigning computational methods...", level=3)

    if 'Methods' not in CompStruct:
        CompStruct['Methods'] = {}

    Methods = CompStruct['Methods']

    # FIX: Import each function from its own module
    from routines.mesh.prepare_mesh_sp_safe import prepare_mesh_sp_safe
    Methods['PrepareMesh'] = prepare_mesh_sp_safe
    debug_print("    Assigned PrepareMesh", level=4)

    from routines.mesh.prepare_mesh_bh import prepare_mesh_bh
    Methods['PrepareMeshBH'] = prepare_mesh_bh
    debug_print("    Assigned PrepareMeshBH", level=4)

    # FIX: Each function in its own file
    from routines.mesh.find_bedges import find_bedges
    Methods['FindBEdges'] = find_bedges
    debug_print("    Assigned FindBEdges", level=4)

    from routines.mesh.make_cont_bedges import make_cont_bedges
    Methods['MakeContBEdges'] = make_cont_bedges
    debug_print("    Assigned MakeContBEdges", level=4)

    from routines.mesh.find_edge_orient import find_edge_orient
    Methods['FindEdgeOrient'] = find_edge_orient
    debug_print("    Assigned FindEdgeOrient", level=4)

    from routines.mesh.add_nodes_cubic import add_nodes_cubic
    Methods['AddNodesCubic'] = add_nodes_cubic
    debug_print("    Assigned AddNodesCubic", level=4)

    # Physical properties
    n_domain = CompStruct['Data']['N_domain']
    Methods['PreparePhysProp'] = [None] * n_domain

    from routines.physics.prepare_physprop_fluid_sp_safe import prepare_physprop_fluid_sp_safe
    from routines.physics.prepare_physprop_htti_sp_safe import prepare_physprop_htti_sp_safe

    for ii_d in range(n_domain):
        domain_type = CompStruct['Model']['DomainType'][ii_d]
        if domain_type == 'fluid':
            Methods['PreparePhysProp'][ii_d] = prepare_physprop_fluid_sp_safe
        elif domain_type == 'HTTI':
            Methods['PreparePhysProp'][ii_d] = prepare_physprop_htti_sp_safe

    debug_print("  St2_2: Mesh generation methods assigned", level=3)

    return CompStruct