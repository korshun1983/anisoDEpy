"""
AddNodesCubic.py
================
Add nodes for cubic elements.
"""

import numpy as np
from utils import debug_print

def add_nodes_cubic(MeshNodes, MeshTri):
    """
    Add nodes for cubic interpolation (10 nodes per triangle)
    Returns modified MeshNodes and MeshTri with 10-node connectivity
    """
    debug_print("      AddNodesCubic: Adding cubic interpolation nodes...", level=5)

    n_tri = MeshTri.shape[1]
    n_original_nodes = MeshNodes.shape[1]

    # Track new nodes to avoid duplicates
    midpoint_cache = {}  # (min_node, max_node) -> new_node_index
    new_nodes = []

    # Pre-allocate 10-node connectivity array
    cubic_tri = np.zeros((10, n_tri), dtype=int)

    for i in range(n_tri):
        # Get triangle vertices (1-based MATLAB indices)
        n1, n2, n3 = MeshTri[:3, i]

        # Vertex nodes (first 3)
        cubic_tri[0, i] = n1
        cubic_tri[1, i] = n2
        cubic_tri[2, i] = n3

        # Edge midpoints (nodes 4-6)
        # Edge 12
        key = tuple(sorted((n1, n2)))
        if key not in midpoint_cache:
            midpoint_cache[key] = n_original_nodes + len(new_nodes)
            p1 = MeshNodes[:, n1-1]
            p2 = MeshNodes[:, n2-1]
            new_nodes.append(0.5 * (p1 + p2))
        cubic_tri[3, i] = midpoint_cache[key]

        # Edge 23
        key = tuple(sorted((n2, n3)))
        if key not in midpoint_cache:
            midpoint_cache[key] = n_original_nodes + len(new_nodes)
            p1 = MeshNodes[:, n2-1]
            p2 = MeshNodes[:, n3-1]
            new_nodes.append(0.5 * (p1 + p2))
        cubic_tri[4, i] = midpoint_cache[key]

        # Edge 31
        key = tuple(sorted((n3, n1)))
        if key not in midpoint_cache:
            midpoint_cache[key] = n_original_nodes + len(new_nodes)
            p1 = MeshNodes[:, n3-1]
            p2 = MeshNodes[:, n1-1]
            new_nodes.append(0.5 * (p1 + p2))
        cubic_tri[5, i] = midpoint_cache[key]

        # Centroid (nodes 7-10, all same for now)
        p1 = MeshNodes[:, n1-1]
        p2 = MeshNodes[:, n2-1]
        p3 = MeshNodes[:, n3-1]
        centroid = (p1 + p2 + p3) / 3.0
        centroid_index = n_original_nodes + len(new_nodes)
        new_nodes.append(centroid)

        cubic_tri[6:10, i] = centroid_index

    # Append new nodes
    if new_nodes:
        new_nodes_array = np.column_stack(new_nodes)
        MeshNodes = np.hstack([MeshNodes, new_nodes_array])

    MeshProps = {
        'n_nodes_per_element': 10,
        'n_elements': n_tri,
        'n_new_nodes': len(new_nodes)
    }

    return MeshNodes, cubic_tri, MeshProps