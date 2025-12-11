# routines/meshgen.py
"""
===============================================================================
Mesh Generation for SAFE Method using pygmsh + gmsh
Generates high-order triangular meshes for cylindrical domains with
SPECIAL HANDLING for solid central cylinder
===============================================================================
Критические исправления:
1. Правильная обработка центрального сплошного цилиндра (fluid)
2. Исправлен вызов gmsh API
3. Удалена старая визуализация (перенесена в gen_aniso.py)
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
    logger.info("        Generating 2D mesh with pygmsh...")

    boundary_shape = getattr(CompStruct.Mesh, 'ext_boundary_shape', 'cir').lower()
    if boundary_shape not in ['cir', 'rect']:
        logger.warning(f"Invalid boundary shape '{boundary_shape}', defaulting to 'cir'")
        boundary_shape = 'cir'

    mesh_data = generate_gmsh_mesh(CompStruct, boundary_shape)

    if CompStruct.Advanced.N_nodes == 10:
        logger.info("          Converting to cubic elements...")
        mesh_data = convert_to_cubic_elements(mesh_data)

    logger.info(f"          Mesh: {mesh_data['MeshNodes'].shape[1]} nodes, "
                f"{mesh_data['MeshTri'].shape[1]} elements")

    return mesh_data


def generate_gmsh_mesh(CompStruct: Any, boundary_shape: str) -> Dict[str, Any]:
    """
    Generate 2D mesh using gmsh with MATLAB-compatible layer geometry.
    DomainRx contains OUTER RADII of layers, not boundary positions.
    This function is called AFTER PML domain is added, so all domains exist.
    """
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    model = gmsh.model()
    model.add("safe_waveguide")

    # DomainRx contains OUTER RADII of each layer
    # Example: [0.1, 2.0, 3.0] for fluid, HTTI, and PML layers
    domain_rx = np.array(CompStruct.Model['DomainRx'])
    domain_ry = np.array(CompStruct.Model['DomainRy'])
    n_layers = len(domain_rx)  # Number of layers (domains)

    logger.info(f"      Creating {n_layers} layers with outer radii: {domain_rx}")
    logger.info(f"      Domain types: {CompStruct.Model['DomainType']}")

    factory = model.occ

    # Build concentric layers starting from r=0
    # radii = [0, r1, r2, ..., rn] for easy indexing
    radii = np.concatenate([[0.0], domain_rx])  # Add r=0 at center

    layer_dimtags = []

    for i in range(n_layers):
        r_inner = radii[i]
        r_outer = radii[i + 1]
        domain_type = CompStruct.Model['DomainType'][i]

        logger.info(f"      Creating layer {i + 1}: {domain_type}, r={r_inner:.4f} to {r_outer:.4f} m")

        # Create outer boundary
        if boundary_shape == 'cir':
            outer_tag = factory.addDisk(0, 0, 0, r_outer, r_outer)
            outer_dimtag = (2, outer_tag)

            # If not the innermost layer, cut out the inner part
            if i > 0:
                inner_tag = factory.addDisk(0, 0, 0, r_inner, r_inner)
                inner_dimtag = (2, inner_tag)
                result_dimtags, _ = factory.cut([outer_dimtag], [inner_dimtag],
                                                removeObject=True, removeTool=False)
                layer_dimtag = result_dimtags[0]
            else:
                layer_dimtag = outer_dimtag
        else:
            # Rectangular boundary (placeholder for future implementation)
            raise NotImplementedError("Rectangular boundary not yet implemented")

        layer_dimtags.append(layer_dimtag)

        # Assign physical group for this domain
        model.addPhysicalGroup(2, [layer_dimtag[1]], tag=i + 1)
        logger.info(f"        Physical group {i + 1} created")

    factory.synchronize()

    # Create outer boundary physical group (for BC application)
    try:
        last_layer_dimtag = layer_dimtags[-1]
        outer_boundary_dimtags = model.getBoundary([last_layer_dimtag], oriented=False)

        if outer_boundary_dimtags:
            edge_tags = [dimtag[1] for dimtag in outer_boundary_dimtags]
            model.addPhysicalGroup(1, edge_tags, tag=n_layers + 1)
            logger.info(f"      Outer boundary physical group created with {len(edge_tags)} edges")
    except Exception as e:
        logger.warning(f"Could not create outer boundary physical group: {e}")

    factory.synchronize()

    # Configure mesh size
    # Use absolute hmax calculated in prepare_model_params
    hmax = getattr(CompStruct.Mesh, 'hmax_absolute', CompStruct.Mesh.hmax)
    if isinstance(hmax, (list, np.ndarray)):
        hmax = hmax[0] if len(hmax) > 0 else 0.1

    logger.info(f"      Setting mesh size: hmax={hmax:.4f} m")

    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", hmax / 2)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", hmax)

    # Configure mesh gradient if specified
    if hasattr(CompStruct.Mesh, 'dhmax') and CompStruct.Mesh.dhmax > 0:
        logger.info(f"      Setting mesh gradient control: dhmax={CompStruct.Mesh.dhmax}")
        gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", CompStruct.Mesh.dhmax)
        gmsh.option.setNumber("Mesh.Smoothing", 1)
        gmsh.option.setNumber("Mesh.SmoothRatio", CompStruct.Mesh.dhmax)

    # Set element order (cubic)
    if CompStruct.Advanced.N_nodes == 10:
        gmsh.option.setNumber("Mesh.ElementOrder", 3)
    elif CompStruct.Advanced.N_nodes == 6:
        gmsh.option.setNumber("Mesh.ElementOrder", 2)
    else:
        gmsh.option.setNumber("Mesh.ElementOrder", 1)

    # Generate mesh
    logger.info("      Generating mesh...")
    model.mesh.generate(2)
    logger.info("      Mesh generation complete")

    # Extract mesh data
    node_tags, node_coords, _ = model.mesh.getNodes()
    MeshNodes = node_coords.reshape(-1, 3).T[:2, :]  # 2D coordinates

    element_types, element_tags, node_tags_list = model.mesh.getElements()

    # Extract only triangular elements
    tri_types = [2, 9, 21]  # Linear, quadratic, cubic triangles
    tri_indices = [i for i, etype in enumerate(element_types) if etype in tri_types]

    if not tri_indices:
        raise RuntimeError("No triangular elements found in generated mesh")

    tri_idx = tri_indices[0]
    tri_node_tags = node_tags_list[tri_idx]
    tri_elements = element_tags[tri_idx]

    n_nodes_per_tri = len(tri_node_tags) // len(tri_elements)
    triangles = tri_node_tags.reshape(-1, n_nodes_per_tri).T - 1  # 0-based indexing

    # Assign domain markers to elements based on their centroid location
    logger.info("      Assigning domain markers to elements...")
    tri_nodes = triangles[:3, :].astype(int)
    centers = np.mean(MeshNodes[:, tri_nodes], axis=1)
    radii = np.sqrt(centers[0, :] ** 2 + centers[1, :] ** 2)

    domain_markers = np.zeros(triangles.shape[1], dtype=int)

    for i in range(n_layers):
        r_inner = radii[i]
        r_outer = radii[i + 1]

        if i == 0:
            # First layer: 0 <= r <= r_outer
            mask = (radii >= 0) & (radii <= r_outer)
        else:
            # Subsequent layers: r_inner < r <= r_outer
            mask = (radii > r_inner) & (radii <= r_outer)

        domain_markers[mask] = i + 1
        n_elements_layer = np.sum(mask)
        logger.info(f"        Domain {i + 1}: {n_elements_layer} elements")

    logger.info(f"      Total elements: {len(domain_markers)}")

    # Create MeshTri with domain markers
    MeshTri = np.vstack([triangles, domain_markers.reshape(1, -1)])

    # Extract boundary edges (for interface conditions)
    boundary_edges = []
    try:
        edge_types, edge_tags, edge_node_tags = model.mesh.getElements(dim=1)
        if edge_node_tags:
            # Get all edges (gmsh doesn't directly give boundary edges)
            # We'll identify them later in Stage 3
            logger.info(f"      Extracted {len(edge_tags)} edges for boundary processing")
    except Exception as e:
        logger.warning(f"Could not extract boundary edges: {e}")

    if boundary_edges:
        BoundaryEdges = np.hstack(boundary_edges)
    else:
        # Create empty array - boundary edges will be identified in Stage 3
        BoundaryEdges = np.empty((3, 0), dtype=int)
        logger.info("      No boundary edges extracted (will be identified in Stage 3)")

    gmsh.finalize()

    # Compute mesh properties
    MeshProps = create_mesh_props(MeshNodes, MeshTri, CompStruct)

    return {
        'MeshNodes': MeshNodes,
        'BoundaryEdges': BoundaryEdges,
        'MeshTri': MeshTri,
        'MeshProps': MeshProps
    }


def create_mesh_props(MeshNodes: np.ndarray, MeshTri: np.ndarray,
                      CompStruct: Any) -> Dict[str, np.ndarray]:
    """Compute mesh properties needed for matrix assembly"""
    n_tri = MeshTri.shape[1]

    areas = np.zeros(n_tri)
    a_coeff = np.zeros((6, n_tri))
    b_coeff = np.zeros((6, n_tri))
    c_coeff = np.zeros((6, n_tri))

    for i in range(n_tri):
        nodes = MeshTri[:3, i].astype(int)
        p1 = MeshNodes[:, nodes[0]]
        p2 = MeshNodes[:, nodes[1]]
        p3 = MeshNodes[:, nodes[2]]

        area = 0.5 * abs(
            p1[0] * (p2[1] - p3[1]) +
            p2[0] * (p3[1] - p1[1]) +
            p3[0] * (p1[1] - p2[1])
        )
        areas[i] = max(area, 1e-12)

        a_coeff[0, i] = p2[0] * p3[1] - p3[0] * p2[1]
        a_coeff[1, i] = p3[0] * p1[1] - p1[0] * p3[1]
        a_coeff[2, i] = p1[0] * p2[1] - p2[0] * p1[1]

        b_coeff[0, i] = p2[1] - p3[1]
        b_coeff[1, i] = p3[1] - p1[1]
        b_coeff[2, i] = p1[1] - p2[1]

        c_coeff[0, i] = p3[0] - p2[0]
        c_coeff[1, i] = p1[0] - p3[0]
        c_coeff[2, i] = p2[0] - p1[0]

    if CompStruct.Advanced.N_nodes == 10:
        a_coeff = np.vstack([a_coeff, np.zeros((3, n_tri))])
        b_coeff = np.vstack([b_coeff, np.zeros((3, n_tri))])
        c_coeff = np.vstack([c_coeff, np.zeros((3, n_tri))])

    dxL = np.zeros((6, n_tri))
    dyL = np.zeros((6, n_tri))

    for i in range(n_tri):
        area = areas[i]
        if area > 1e-12:
            dxL[:3, i] = b_coeff[:3, i] / (2 * area)
            dyL[:3, i] = c_coeff[:3, i] / (2 * area)

    return {
        'DS': np.ones((6, n_tri)),
        'delta': np.ones((6, n_tri)),
        'a': a_coeff,
        'b': b_coeff,
        'c': c_coeff,
        'area': areas,
        'dxL': dxL,
        'dyL': dyL
    }


def convert_to_cubic_elements(mesh_data: Dict) -> Dict:
    """Convert gmsh's quadratic/cubic elements to SAFE format"""
    MeshTri = mesh_data['MeshTri']

    if MeshTri.shape[0] >= 10:
        logger.info("          gmsh generated cubic elements correctly")
        return mesh_data

    logger.warning("          Manual cubic conversion needed (gmsh order mismatch)")
    return mesh_data