# routines/meshgen.py
"""
===============================================================================
Mesh Generation for SAFE Method using pygmsh + gmsh
Generates high-order triangular meshes for cylindrical domains
===============================================================================
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import gmsh
import pygmsh

logger = logging.getLogger(__name__)


def prepare_mesh(CompStruct: Any) -> Dict[str, Any]:
    """
    Main mesh generation routine using pygmsh.
    Replicates PrepareMesh_sp_SAFE.m with high-order element support.

    Returns:
        Dict containing MeshNodes, BoundaryEdges, MeshTri, MeshProps
    """
    logger.info("        Generating 2D mesh with pygmsh...")

    # Check boundary shape preference
    boundary_shape = CompStruct.Mesh.get('ext_boundary_shape', 'cir').lower()
    if boundary_shape not in ['cir', 'rect']:
        logger.warning(f"Invalid boundary shape '{boundary_shape}', defaulting to 'cir'")
        boundary_shape = 'cir'

    # Generate mesh using gmsh
    mesh_data = generate_gmsh_mesh(CompStruct, boundary_shape)

    # Convert to cubic (3rd order) elements if required
    if CompStruct.Advanced['N_nodes'] == 10:
        logger.info("          Converting to cubic elements...")
        mesh_data = convert_to_cubic_elements(mesh_data)

    logger.info(f"          Mesh: {mesh_data['MeshNodes'].shape[1]} nodes, "
                f"{mesh_data['MeshTri'].shape[1]} elements")

    return mesh_data


def generate_gmsh_mesh(CompStruct: Any, boundary_shape: str) -> Dict[str, Any]:
    """
    Generate 2D mesh using gmsh with physical group labeling for domains.

    Args:
        CompStruct: Computation structure with domain parameters
        boundary_shape: 'cir' (circular) or 'rect' (rectangular) outer boundary

    Returns:
        Dict with mesh data
    """
    # Initialize gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)  # Suppress output

    model = gmsh.model()
    model.add("safe_waveguide")

    # Extract domain parameters
    domain_rx = np.array(CompStruct.Model['DomainRx'])
    domain_ry = np.array(CompStruct.Model['DomainRy'])
    n_layers = len(domain_rx) - 1

    # Create geometry
    factory = model.occ  # OpenCASCADE kernel for robust geometry

    # Define boundaries for each layer
    boundary_tags = []

    for i in range(n_layers + 1):
        rx = domain_rx[i]
        ry = domain_ry[i]

        if i == 0:  # Inner borehole (if radius > 0)
            if rx > 0 and ry > 0:
                if boundary_shape == 'cir':
                    tag = factory.addDisk(0, 0, 0, rx, ry)
                else:
                    tag = factory.addRectangle(-rx / 2, -ry / 2, 0, rx, ry)
                boundary_tags.append(tag)
            else:
                # Point or line at center - treat specially
                boundary_tags.append(None)
        else:
            if boundary_shape == 'cir':
                tag = factory.addDisk(0, 0, 0, rx, ry)
            else:
                tag = factory.addRectangle(-rx / 2, -ry / 2, 0, rx, ry)
            boundary_tags.append(tag)

    # Create domains by boolean difference
    domain_tags = []

    for i in range(n_layers):
        if boundary_tags[i] is None:
            # First domain is just the first layer
            domain_tag = boundary_tags[i + 1]
        else:
            # Subtract inner from outer
            outer_tag = boundary_tags[i + 1]
            inner_tag = boundary_tags[i]
            domain_dim_tags, _ = factory.cut(
                [(2, outer_tag)], [(2, inner_tag)], removeObject=True, removeTool=False
            )
            domain_tag = domain_dim_tags[0][1]

        domain_tags.append(domain_tag)

        # Assign physical group for domain (will be used as element marker)
        model.addPhysicalGroup(2, [domain_tag], tag=i + 1)

    # Add outer boundary physical group
    outer_boundary_tag = factory.getBoundary([(2, domain_tags[-1])], oriented=False)
    if outer_boundary_tag:
        model.addPhysicalGroup(1, [tag[1] for tag in outer_boundary_tag], tag=n_layers + 1)

    # Synchronize geometry
    factory.synchronize()

    # Set meshing algorithm
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal Delaunay

    # Set element order
    if CompStruct.Advanced['N_nodes'] == 10:
        gmsh.option.setNumber("Mesh.ElementOrder", 3)  # 3rd order (cubic)
    elif CompStruct.Advanced['N_nodes'] == 6:
        gmsh.option.setNumber("Mesh.ElementOrder", 2)  # 2nd order (quadratic)
    else:
        gmsh.option.setNumber("Mesh.ElementOrder", 1)  # Linear

    # Set mesh size from hmax parameter
    hmax = CompStruct.Mesh['hmax']
    if isinstance(hmax, list):
        hmax = hmax[0]  # Use first layer value

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", hmax / 2)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", hmax)

    # Generate mesh
    model.mesh.generate(2)

    # Extract nodes
    node_tags, node_coords, _ = model.mesh.getNodes()
    MeshNodes = node_coords.reshape(-1, 3).T[:2, :]  # Only x,y coordinates

    # Extract triangles
    element_types, element_tags, node_tags_list = model.mesh.getElements()

    # Find triangular elements (type 2 = 3-node, type 9 = 6-node, type 21 = 10-node)
    tri_mask = np.isin(element_types, [2, 9, 21])
    if not np.any(tri_mask):
        raise RuntimeError("No triangular elements found in mesh")

    # Get triangle nodes
    tri_node_tags = node_tags_list[tri_mask][0]

    # Reshape based on element order
    n_nodes_per_tri = len(tri_node_tags) // len(element_tags[tri_mask][0])
    triangles = tri_node_tags.reshape(-1, n_nodes_per_tri).T

    # Adjust to 0-based indexing
    triangles = triangles - 1

    # Get domain markers (physical groups)
    domain_markers = np.zeros(triangles.shape[1], dtype=int)

    for domain_id in range(1, n_layers + 1):
        elem_dim_tags = model.mesh.getElementsForPhysicalGroup(2, domain_id)
        if len(elem_dim_tags) > 0:
            elem_indices = elem_dim_tags[1] - 1  # 0-based
            elem_indices = elem_indices[elem_indices < len(domain_markers)]
            domain_markers[elem_indices] = domain_id

    # Append domain markers as 4th row
    MeshTri = np.vstack([triangles[:3, :], domain_markers])

    # Extract boundary edges
    boundary_edges = []
    for dim in [1]:  # 1D elements (edges)
        edge_dim_tags = model.mesh.getElementsForPhysicalGroup(1, n_layers + 1)
        if len(edge_dim_tags) > 1:
            edge_node_tags = edge_dim_tags[2][0]
            edge_nodes = edge_node_tags.reshape(-1, 2).T - 1
            n_edges = edge_nodes.shape[1]
            boundary_tag = np.ones(n_edges, dtype=int) * (n_layers + 1)
            boundary_edges.append(np.vstack([edge_nodes, boundary_tag]))

    if boundary_edges:
        BoundaryEdges = np.hstack(boundary_edges)
    else:
        BoundaryEdges = np.empty((3, 0), dtype=int)

    # Clean up
    gmsh.finalize()

    # Create MeshProps placeholder
    MeshProps = create_mesh_props(MeshNodes, MeshTri, CompStruct)

    return {
        'MeshNodes': MeshNodes,
        'BoundaryEdges': BoundaryEdges,
        'MeshTri': MeshTri,
        'MeshProps': MeshProps
    }


def create_mesh_props(MeshNodes: np.ndarray, MeshTri: np.ndarray,
                      CompStruct: Any) -> Dict[str, np.ndarray]:
    """
    Compute mesh properties needed for matrix assembly:
    shape function coefficients, element areas, etc.
    """
    n_tri = MeshTri.shape[1]

    # Triangle areas using shoelace formula
    areas = np.zeros(n_tri)
    a_coeff = np.zeros((6, n_tri))
    b_coeff = np.zeros((6, n_tri))
    c_coeff = np.zeros((6, n_tri))

    for i in range(n_tri):
        nodes = MeshTri[:3, i].astype(int)
        p1 = MeshNodes[:, nodes[0]]
        p2 = MeshNodes[:, nodes[1]]
        p3 = MeshNodes[:, nodes[2]]

        # Area
        area = 0.5 * abs(
            p1[0] * (p2[1] - p3[1]) +
            p2[0] * (p3[1] - p1[1]) +
            p3[0] * (p1[1] - p2[1])
        )
        areas[i] = max(area, 1e-12)

        # Shape function coefficients for linear elements
        # a_i = x_j*y_k - x_k*y_j
        # b_i = y_j - y_k
        # c_i = x_k - x_j

        a_coeff[0, i] = p2[0] * p3[1] - p3[0] * p2[1]
        a_coeff[1, i] = p3[0] * p1[1] - p1[0] * p3[1]
        a_coeff[2, i] = p1[0] * p2[1] - p2[0] * p1[1]

        b_coeff[0, i] = p2[1] - p3[1]
        b_coeff[1, i] = p3[1] - p1[1]
        b_coeff[2, i] = p1[1] - p2[1]

        c_coeff[0, i] = p3[0] - p2[0]
        c_coeff[1, i] = p1[0] - p3[0]
        c_coeff[2, i] = p2[0] - p1[0]

    # For cubic elements, expand coefficients (placeholder for full cubic)
    if CompStruct.Advanced['N_nodes'] == 10:
        # Expand to 6 per node (for 3 variables)
        a_coeff = np.vstack([a_coeff, np.zeros((3, n_tri))])
        b_coeff = np.vstack([b_coeff, np.zeros((3, n_tri))])
        c_coeff = np.vstack([c_coeff, np.zeros((3, n_tri))])

    return {
        'DS': np.ones((6, n_tri)),  # Will be refined for cubic
        'delta': np.ones((6, n_tri)),
        'a': a_coeff,
        'b': b_coeff,
        'c': c_coeff,
        'area': areas
    }


def convert_to_cubic_elements(mesh_data: Dict) -> Dict:
    """
    Convert gmsh's quadratic/cubic elements to the format expected by SAFE.
    If gmsh already generated cubic elements, just reformat.
    """
    MeshTri = mesh_data['MeshTri']

    # Check if we already have enough nodes
    if MeshTri.shape[0] >= 10:
        logger.info("          gmsh already generated cubic elements")
        return mesh_data

    # If not, we need to add nodes (simplified version)
    logger.warning("          Manual cubic conversion may be needed")

    return mesh_data