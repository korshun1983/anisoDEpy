"""
Mesh generation package with production-ready functions.
"""

from .mesh_generator import (
    prepare_mesh_bh,
    meshfaces,
    add_nodes_cubic,
    find_bedges,
    find_edge_orient,
    make_cont_bedges,
    _inpolygon
)

__all__ = [
    'prepare_mesh_bh',
    'meshfaces',
    'add_nodes_cubic',
    'find_bedges',
    'find_edge_orient',
    'make_cont_bedges'
]