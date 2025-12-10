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
    Generate 2D mesh using gmsh with physical group labeling for domains.
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    model = gmsh.model()
    model.add("safe_waveguide")

    domain_rx = np.array(CompStruct.Model['DomainRx'])
    domain_ry = np.array(CompStruct.Model['DomainRy'])
    n_layers = len(domain_rx) - 1

    logger.info(f"      Creating {n_layers} domains from {len(domain_rx)} radii")
    logger.info(f"      Radii: {domain_rx}")

    factory = model.occ

    boundary_tags = []
    boundary_dimtags = []

    # Центральный цилиндр
    if domain_rx[0] > 0.0:
        if boundary_shape == 'cir':
            tag = factory.addDisk(0, 0, 0, domain_rx[0], domain_ry[0])
            boundary_dimtags.append((2, tag))
        else:
            tag = factory.addRectangle(-domain_rx[0] / 2, -domain_ry[0] / 2, 0, domain_rx[0], domain_ry[0])
            boundary_dimtags.append((2, tag))
        boundary_tags.append(tag)
        logger.info(f"      Central solid cylinder: r={domain_rx[0]}")
    else:
        boundary_tags.append(None)
        boundary_dimtags.append(None)
        logger.info("      Central point (no geometry)")

    # Внешние границы
    for i in range(1, n_layers + 1):
        rx = domain_rx[i]
        ry = domain_ry[i]

        if boundary_shape == 'cir':
            tag = factory.addDisk(0, 0, 0, rx, ry)
            boundary_dimtags.append((2, tag))
        else:
            tag = factory.addRectangle(-rx / 2, -ry / 2, 0, rx, ry)
            boundary_dimtags.append((2, tag))
        boundary_tags.append(tag)
        logger.info(f"      Outer boundary {i}: r={rx}")

    factory.synchronize()

    domain_tags = []
    domain_dimtags = []

    for i in range(n_layers):
        if i == 0 and domain_rx[0] > 0.0:
            domain_tag = boundary_tags[1]
            domain_dimtags.append(boundary_dimtags[1])
            logger.info(f"      Domain 1 (solid): using boundary tag {domain_tag}")
        else:
            outer_dimtag = boundary_dimtags[i + 1]
            inner_dimtag = boundary_dimtags[i]

            if inner_dimtag is None:
                domain_dimtags.append(outer_dimtag)
                domain_tag = boundary_tags[i + 1]
            else:
                domain_dim_tags, _ = factory.cut(
                    [outer_dimtag], [inner_dimtag], removeObject=True, removeTool=False
                )
                if len(domain_dim_tags) > 0:
                    domain_dimtags.append(domain_dim_tags[0])
                    domain_tag = domain_dim_tags[0][1]
                else:
                    logger.error(f"Failed to create domain {i + 1}")
                    continue

        domain_tags.append(domain_tag)
        model.addPhysicalGroup(2, [domain_tag], tag=i + 1)
        logger.info(f"      Domain {i + 1} physical group created")

    # Внешняя граница для ABC/PML
    try:
        if domain_dimtags:
            last_domain_dimtag = domain_dimtags[-1]
            outer_boundary_dimtags = model.getBoundary([last_domain_dimtag], oriented=False)

            if outer_boundary_dimtags:
                edge_tags = [dimtag[1] for dimtag in outer_boundary_dimtags]
                model.addPhysicalGroup(1, edge_tags, tag=n_layers + 1)
                logger.info(f"      Outer boundary physical group created")
    except Exception as e:
        logger.warning(f"Could not create outer boundary physical group: {e}")

    factory.synchronize()

    physical_groups = gmsh.model.getPhysicalGroups()
    domain_groups = [pg for pg in physical_groups if pg[0] == 2]
    logger.info(f"      Created {len(domain_groups)} domain physical groups")

    gmsh.option.setNumber("Mesh.Algorithm", 6)

    if CompStruct.Advanced.N_nodes == 10:
        gmsh.option.setNumber("Mesh.ElementOrder", 3)
    elif CompStruct.Advanced.N_nodes == 6:
        gmsh.option.setNumber("Mesh.ElementOrder", 2)
    else:
        gmsh.option.setNumber("Mesh.ElementOrder", 1)

    hmax = CompStruct.Mesh.hmax
    if isinstance(hmax, list):
        hmax = hmax[0]

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", hmax / 2)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", hmax)

    logger.info("      Generating mesh...")
    model.mesh.generate(2)

    node_tags, node_coords, _ = model.mesh.getNodes()
    MeshNodes = node_coords.reshape(-1, 3).T[:2, :]

    element_types, element_tags, node_tags_list = model.mesh.getElements()

    tri_types = [2, 9, 21]
    tri_indices = [i for i, etype in enumerate(element_types) if etype in tri_types]

    if not tri_indices:
        raise RuntimeError("No triangular elements found")

    tri_idx = tri_indices[0]
    tri_node_tags = node_tags_list[tri_idx]
    tri_elements = element_tags[tri_idx]

    n_nodes_per_tri = len(tri_node_tags) // len(tri_elements)
    triangles = tri_node_tags.reshape(-1, n_nodes_per_tri).T - 1

    logger.info("      Assigning domain markers based on geometry...")
    domain_markers = np.zeros(triangles.shape[1], dtype=int)

    tri_nodes = triangles[:3, :].astype(int)
    centers = np.mean(MeshNodes[:, tri_nodes], axis=1)
    radii = np.sqrt(centers[0, :] ** 2 + centers[1, :] ** 2)

    for i in range(n_layers):
        r_inner = domain_rx[i]
        r_outer = domain_rx[i + 1]

        if i == 0 and r_inner == 0.0:
            mask = radii <= r_outer
        else:
            mask = (radii > r_inner) & (radii <= r_outer)

        domain_markers[mask] = i + 1
        logger.info(f"        Domain {i + 1}: {np.sum(mask)} elements (r={r_inner:.4f} to {r_outer:.4f})")

    unique_markers = np.unique(domain_markers)
    logger.info(f"      Assigned domain markers: {unique_markers}")

    MeshTri = np.vstack([triangles, domain_markers.reshape(1, -1)])
    logger.info(f"      Final MeshTri shape: {MeshTri.shape}")
    logger.info(f"      Domain distribution: {np.bincount(domain_markers)}")

    boundary_edges = []
    try:
        edge_types, edge_tags, edge_node_tags = gmsh.model.getElementsForPhysicalGroup(1, n_layers + 1)
        if len(edge_tags) > 0:
            edge_node_tags = edge_node_tags[0]
            edge_nodes = edge_node_tags.reshape(-1, 2).T - 1
            n_edges = edge_nodes.shape[1]
            boundary_tag = np.ones(n_edges, dtype=int) * (n_layers + 1)
            boundary_edges.append(np.vstack([edge_nodes, boundary_tag]))
            logger.info(f"      Extracted {n_edges} boundary edges")
    except Exception as e:
        logger.warning(f"Could not get boundary edges: {e}")

    if boundary_edges:
        BoundaryEdges = np.hstack(boundary_edges)
    else:
        BoundaryEdges = np.empty((3, 0), dtype=int)
        logger.warning("No boundary edges found")

    gmsh.finalize()

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