"""
mesh_generator.py
=================
PRODUCTION-READY SAFE mesh generator
Features: Robust Gmsh init, automatic fallback, detailed status, list-to-array fix
"""

import numpy as np
from typing import Dict, Tuple, List
from utils import debug_print
import time

# Gmsh/pygmsh imports with robust initialization
PYGMSH_AVAILABLE = False
gmsh = None
try:
    import gmsh
    gmsh.option.setNumber("General.Terminal", 0)
    if not gmsh.isInitialized():
        gmsh.initialize()
    import pygmsh
    PYGMSH_AVAILABLE = True
    debug_print("[OK] pygmsh/Gmsh backend is ready", level=3)
except Exception as e:
    debug_print(f"[FAILED] pygmsh/Gmsh not available: {e}", level=1)
    debug_print("-> Using scipy.spatial.Delaunay fallback", level=1)
    debug_print("-> For better meshes: pip install gmsh pygmsh", level=1)
    PYGMSH_AVAILABLE = False

from scipy.spatial import Delaunay


def prepare_mesh_bh(CompStruct: Dict) -> Tuple[np.ndarray, np.ndarray, Dict, Dict]:
    """EXACT MATLAB equivalent of PrepareMeshBH.m"""
    debug_print("    PrepareMeshBH: Generating boundary geometry...", level=4)

    n_domain = CompStruct['Data']['N_domain']
    nodes_list = np.array([], dtype=float).reshape(0, 2)
    edges_list = np.array([], dtype=int).reshape(0, 2)
    domain_faces = {}
    total_edge_count = 0

    for ii_d in range(n_domain):
        # Extract parameters
        Rx = CompStruct['Model']['DomainRx'][ii_d]
        Ry = CompStruct['Model']['DomainRy'][ii_d]
        ThetaRot = CompStruct['Model']['DomainTheta'][ii_d]
        Ecc = CompStruct['Model']['DomainEcc'][ii_d]
        EccAngle = CompStruct['Model']['DomainEccAngle'][ii_d]
        Nth = CompStruct['Model']['DomainNth'][ii_d]

        debug_print(f"      Domain {ii_d}: Rx={Rx:.3f}, Ry={Ry:.3f}, Nth={Nth}", level=5)

        # Angular grid
        dtheta = np.pi / Nth
        theta = np.arange(-np.pi, np.pi - dtheta/2, dtheta)

        # Check rectangular boundary
        boundary_rec = False
        if CompStruct['Mesh']['ext_boundary_shape'].lower() == 'rec':
            add_loc = CompStruct['Model']['AddDomainLoc']
            if (add_loc == 'ext' and ii_d >= n_domain - 1) or (add_loc == 'int' and ii_d == n_domain - 1):
                boundary_rec = True

        # Generate boundary nodes
        if not boundary_rec:
            Xc = Ecc * np.cos(EccAngle)
            Yc = Ecc * np.sin(EccAngle)
            XBgrid = Xc + Rx * np.cos(theta) * np.cos(ThetaRot) - Ry * np.sin(theta) * np.sin(ThetaRot)
            YBgrid = Yc + Rx * np.cos(theta) * np.sin(ThetaRot) + Ry * np.sin(theta) * np.cos(ThetaRot)
        else:
            # Rectangular boundary
            n_samples = max(Nth // 4, 4)
            XBgrid = np.concatenate([
                np.linspace(-Rx, Rx, n_samples, endpoint=False),
                np.full(n_samples, Rx),
                np.linspace(Rx, -Rx, n_samples, endpoint=False),
                np.full(n_samples, -Rx)
            ])
            YBgrid = np.concatenate([
                np.full(n_samples, -Ry),
                np.linspace(-Ry, Ry, n_samples, endpoint=False),
                np.full(n_samples, Ry),
                np.linspace(Ry, -Ry, n_samples, endpoint=False)
            ])

        domain_nodes = np.column_stack([XBgrid, YBgrid])
        domain_nodes, _ = np.unique(domain_nodes, axis=0, return_index=True)

        if len(domain_nodes) < 3:
            raise ValueError(f"Domain {ii_d}: Insufficient boundary nodes ({len(domain_nodes)})")

        # Define edges (1-based)
        n_nodes = len(domain_nodes)
        local_edges = np.column_stack([
            np.arange(1, n_nodes + 1),
            np.concatenate([np.arange(2, n_nodes + 1), [1]])
        ])

        # Track global edge indices
        n_local_edges = len(local_edges)
        domain_faces[ii_d] = list(range(total_edge_count + 1, total_edge_count + n_local_edges + 1))
        total_edge_count += n_local_edges

        # Update global lists
        if len(edges_list) == 0:
            global_edges = local_edges
            edges_list = global_edges
            nodes_list = domain_nodes
        else:
            node_offset = len(nodes_list)
            global_edges = local_edges + node_offset
            edges_list = np.vstack([edges_list, global_edges])
            nodes_list = np.vstack([nodes_list, domain_nodes])

    # --- CRITICAL FIX: Convert edges_list to numpy array using np.asarray ---
    edges_array = np.asarray(edges_list).astype(int) - 1

    debug_print("[OK] Boundary geometry complete:", level=3)
    debug_print(f"      - Total nodes: {len(nodes_list)}", level=4)
    debug_print(f"      - Total edges: {len(edges_array)}", level=4)
    debug_print(f"      - Domains: {len(domain_faces)}", level=4)
    for domain_id, edge_range in domain_faces.items():
        debug_print(f"        Domain {domain_id}: {len(edge_range)} edges", level=5)

    return nodes_list, edges_array, domain_faces, CompStruct


def meshfaces(nodes, edges, domain_faces, CompStruct, hmax=0.16):
    """Unified interface for mesh generator"""
    if PYGMSH_AVAILABLE:
        try:
            from .gmsh_builder import build_mesh_gmsh
            debug_print("    MeshFaces: Using advanced Gmsh builder...", level=4)
            return build_mesh_gmsh(CompStruct, nodes, edges, domain_faces)
        except Exception as e:
            debug_print(f"[FAILED] Gmsh builder: {e}", level=0)
            debug_print("    Falling back to scipy...", level=1)

    # Fallback to scipy
    return _meshfaces_scipy(nodes, edges, domain_faces, CompStruct, hmax)


def _meshfaces_pygmsh(nodes: np.ndarray, edges: np.ndarray, domain_faces: Dict,
                      CompStruct: Dict, hmax: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Generate mesh using pygmsh"""
    with pygmsh.geo.Geometry() as geom:
        point_ids = [geom.add_point([x, y, 0.0], mesh_size=hmax) for x, y in nodes]

        for domain_id in sorted(domain_faces.keys()):
            face_edges = domain_faces[domain_id]
            curves = []

            for edge_idx in face_edges:
                edge_idx_0 = edge_idx - 1
                start_idx = edges[edge_idx_0, 0]
                end_idx = edges[edge_idx_0, 1]
                line = geom.add_line(point_ids[start_idx], point_ids[end_idx])
                curves.append(line)

            if len(curves) >= 3:
                curve_loop = geom.add_curve_loop(curves)
                geom.add_plane_surface(curve_loop)

        mesh = geom.generate_mesh()

    mesh_points = mesh.points[:, :2]
    triangles = mesh.cells_dict["triangle"]

    centroids = np.mean(mesh_points[triangles], axis=1)
    domain_numbers = _assign_domains(centroids, nodes, edges, domain_faces)

    mesh_tri = np.vstack([triangles.T + 1, domain_numbers])

    return mesh_points.T, mesh_tri, domain_numbers, CompStruct


def _meshfaces_scipy(nodes, edges, domain_faces, CompStruct, hmax=None):
    """Wrapper for Gmsh builder (rename this function!)"""
    try:
        from .gmsh_builder import build_mesh_gmsh
        debug_print("        Using Gmsh builder...", level=5)
        # Gmsh doesn't need pre-generated nodes/edges
        return build_mesh_gmsh(CompStruct)
    except Exception as e:
        debug_print(f"Gmsh failed: {e}, falling back to SciPy", level=1)
        return _meshfaces_scipy_fallback(nodes, edges, domain_faces, CompStruct, hmax)

def _meshfaces_scipy_fallback(nodes, edges, domain_faces, CompStruct, hmax):
    # Generate internal nodes for each domain
    all_nodes = [nodes]
    node_offset = len(nodes)

    for domain_id in sorted(domain_faces.keys()):
        face_edges = domain_faces[domain_id]
        boundary_nodes = [edges[edge_idx - 1, 0] for edge_idx in face_edges]
        boundary_nodes.append(edges[face_edges[-1] - 1, 1])

        polygon = nodes[boundary_nodes]

        # Calculate bounding box
        min_x, max_x = polygon[:, 0].min(), polygon[:, 0].max()
        min_y, max_y = polygon[:, 1].min(), polygon[:, 1].max()

        # Generate internal grid based on hmax
        if hmax is None:
            hmax = 0.16

        nx = max(3, int((max_x - min_x) / hmax))
        ny = max(3, int((max_y - min_y) / hmax))

        # Create internal grid
        x_grid = np.linspace(min_x, max_x, nx)
        y_grid = np.linspace(min_y, max_y, ny)
        xx, yy = np.meshgrid(x_grid, y_grid)
        internal_points = np.column_stack([xx.ravel(), yy.ravel()])

        # Keep only points inside polygon and away from boundary
        from matplotlib.path import Path
        path = Path(polygon)
        inside_mask = path.contains_points(internal_points)

        # Remove points too close to boundary
        for bp in polygon:
            dist = np.sqrt((internal_points[:, 0] - bp[0]) ** 2 +
                           (internal_points[:, 1] - bp[1]) ** 2)
            inside_mask &= (dist > hmax * 0.5)

        internal_points = internal_points[inside_mask]

        if len(internal_points) > 0:
            debug_print(f"          Domain {domain_id}: +{len(internal_points)} internal nodes", level=5)
            all_nodes.append(internal_points)

    # Combine all nodes
    nodes_all = np.vstack(all_nodes)

    # Remove duplicates
    nodes_unique, unique_idx = np.unique(nodes_all, axis=0, return_index=True)
    nodes_final = nodes_unique

    debug_print(f"        Total nodes: {len(nodes_final)} (boundary: {len(nodes)})", level=5)

    # Triangulate
    tri = Delaunay(nodes_final)
    mesh_points = nodes_final
    triangles = tri.simplices

    # Assign domain numbers by centroid location
    centroids = np.mean(nodes_final[triangles], axis=1)
    domain_numbers = _assign_domains(centroids, nodes, edges, domain_faces)

    # Build final MeshTri array
    mesh_tri = np.vstack([triangles.T + 1, domain_numbers])

    return mesh_points.T, mesh_tri, domain_numbers, CompStruct

def _assign_domains(centroids: np.ndarray, nodes: np.ndarray, edges: np.ndarray,
                    domain_faces: Dict) -> np.ndarray:
    """Assign domain numbers to triangles"""
    debug_print("      Assigning domains to triangles...", level=5)

    n_tri = len(centroids)
    domain_numbers = -np.ones(n_tri, dtype=int)

    for domain_id in sorted(domain_faces.keys(), reverse=True):
        face_edges = domain_faces[domain_id]
        boundary_nodes = [edges[edge_idx - 1, 0] for edge_idx in face_edges]
        boundary_nodes.append(edges[face_edges[-1] - 1, 1])

        polygon = nodes[boundary_nodes]
        mask = _inpolygon(centroids, polygon)
        domain_numbers[mask] = domain_id

    if np.any(domain_numbers == -1):
        outermost = max(domain_faces.keys())
        domain_numbers[domain_numbers == -1] = outermost

    return domain_numbers


def _inpolygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Point-in-polygon test"""
    n_points = len(points)
    inside = np.zeros(n_points, dtype=bool)

    px, py = points[:, 0], points[:, 1]
    poly_x, poly_y = polygon[:, 0], polygon[:, 1]

    n_vertices = len(polygon)
    for i in range(n_vertices):
        x1, y1 = poly_x[i], poly_y[i]
        x2, y2 = poly_x[(i + 1) % n_vertices], poly_y[(i + 1) % n_vertices]

        edge_crosses = ((y1 > py) != (y2 > py))
        x_intersect = x1 + (py - y1) * (x2 - x1) / (y2 - y1 + 1e-12)
        inside[edge_crosses & (px < x_intersect)] = ~inside[edge_crosses & (px < x_intersect)]

    return inside


def add_nodes_cubic(MeshNodes: np.ndarray, MeshTri: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Convert to 10-node cubic SAFE elements.
    """
    debug_print("      AddNodesCubic: Adding cubic interpolation nodes...", level=5)

    n_tri = MeshTri.shape[1]
    n_original = MeshNodes.shape[1]

    # Validation
    max_node_idx = np.max(MeshTri[:3, :])
    if max_node_idx > n_original:
        debug_print(f"ERROR: MeshTri contains node index {max_node_idx} but only {n_original} nodes exist!", level=0)
        debug_print("This usually indicates an off-by-one error in mesh generation.", level=0)
        raise IndexError(f"Invalid node index {max_node_idx} (max allowed: {n_original})")

    # Ensure integer indices
    MeshTri = MeshTri.astype(int)

    midpoint_cache = {}
    new_nodes = []
    cubic_tri = np.zeros((10, n_tri), dtype=int)

    for i in range(n_tri):
        # Get vertex nodes (1-based from MeshTri)
        n1, n2, n3 = MeshTri[0, i], MeshTri[1, i], MeshTri[2, i]

        # Store vertex nodes
        cubic_tri[0, i] = n1
        cubic_tri[1, i] = n2
        cubic_tri[2, i] = n3

        # Edge midpoints (convert to 0-based for accessing MeshNodes)
        for edge_idx, (a, b) in enumerate([(n1, n2), (n2, n3), (n3, n1)], start=3):
            key = tuple(sorted((a, b)))
            if key not in midpoint_cache:
                midpoint_cache[key] = n_original + len(new_nodes)  # 0-based index for new node
                # Use 0-based indices: a-1, b-1
                new_nodes.append(0.5 * (MeshNodes[:, a - 1] + MeshNodes[:, b - 1]))
            cubic_tri[edge_idx, i] = midpoint_cache[key] + 1  # 1-based for MeshTri

        # Compute centroid
        p1 = MeshNodes[:, n1 - 1]
        p2 = MeshNodes[:, n2 - 1]
        p3 = MeshNodes[:, n3 - 1]
        centroid = (p1 + p2 + p3) / 3.0

        # Interior nodes 6-9 (at 1/3 from vertices to centroid)
        for j, vertex in enumerate([p1, p2, p3], start=6):
            interior_point = (2 / 3) * vertex + (1 / 3) * centroid
            new_node_idx = n_original + len(new_nodes) + 1
            new_nodes.append(interior_point)
            cubic_tri[j, i] = new_node_idx

        # Centroid node 10
        centroid_idx = n_original + len(new_nodes) + 1
        new_nodes.append(centroid)
        cubic_tri[9, i] = centroid_idx

    # Append new nodes
    if new_nodes:
        MeshNodes = np.hstack([MeshNodes, np.column_stack(new_nodes)])

    MeshProps = {
        'n_nodes_per_element': 10,
        'n_elements': n_tri,
        'n_new_nodes_added': len(new_nodes)
    }

    # Domain numbers - preserve if they exist (4th row)
    if MeshTri.shape[0] >= 4:
        cubic_tri[3, :] = MeshTri[3, :]

    debug_print(f"        Added {len(new_nodes)} nodes ({n_original} → {MeshNodes.shape[1]})", level=5)
    return MeshNodes, cubic_tri, MeshProps


def find_bedges(MeshNodes: np.ndarray, MeshTri: np.ndarray, CompStruct: Dict) -> np.ndarray:
    """Identify boundary edges"""
    debug_print("      FindBEdges: Identifying boundary edges...", level=5)

    tri = MeshTri[:3, :].T

    edges = np.vstack([
        np.column_stack([tri[:, 0], tri[:, 1]]),
        np.column_stack([tri[:, 1], tri[:, 2]]),
        np.column_stack([tri[:, 2], tri[:, 0]])
    ])

    edges = np.sort(edges, axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]

    debug_print(f"      Found {len(boundary_edges)} boundary edges", level=5)

    return boundary_edges.T


def find_edge_orient(DBEdges: np.ndarray, MeshTri: np.ndarray, MeshNodes: np.ndarray) -> np.ndarray:
    """Determine edge orientation"""
    debug_print("      FindEdgeOrient: Determining edge orientation...", level=5)

    if DBEdges.size == 0:
        return DBEdges

    node1, node2 = DBEdges[0], DBEdges[1]

    node1_mask = np.any(MeshTri[:3, :] == node1, axis=0)
    node1_cols = np.where(node1_mask)[0]

    tri_idx = -1
    for col in node1_cols:
        if np.any(MeshTri[:3, col] == node2):
            tri_idx = col
            break

    if tri_idx == -1:
        debug_print(f"        Warning: No triangle for edge ({node1}, {node2})", level=5)
        return DBEdges

    tri_nodes = MeshTri[:3, tri_idx]
    third_node = tri_nodes[(tri_nodes != node1) & (tri_nodes != node2)][0]

    mid = (MeshNodes[:, node1-1] + MeshNodes[:, node2-1]) / 2
    inner_vec = mid - MeshNodes[:, third_node-1]
    tangent_vec = MeshNodes[:, node2-1] - MeshNodes[:, node1-1]

    cross_z = tangent_vec[0] * inner_vec[1] - tangent_vec[1] * inner_vec[0]

    if cross_z > 0:
        DBEdges = DBEdges[[1, 0]]
        debug_print(f"        Flipped edge ({node1}, {node2}) → ({node2}, {node1})", level=5)

    return DBEdges


def make_cont_bedges(DBEdges: np.ndarray, MeshTri: np.ndarray, MeshNodes: np.ndarray, CompStruct: Dict) -> np.ndarray:
    """Create continuous boundary edge loops"""
    debug_print("      MakeContBEdges: Creating continuous boundary...", level=5)

    if DBEdges.size == 0:
        return DBEdges

    DBEdges = DBEdges.T if DBEdges.shape[0] != 2 else DBEdges

    debug_print(f"        Starting with {DBEdges.shape[1]} edges...", level=5)

    first_edge = find_edge_orient(DBEdges[:, 0].copy(), MeshTri, MeshNodes)
    cont_edges = [first_edge]
    DBEdges = np.delete(DBEdges, 0, axis=1)

    current_node = first_edge[1]
    max_iter = DBEdges.shape[1] * 2
    iteration = 0

    while DBEdges.shape[1] > 0 and iteration < max_iter:
        iteration += 1

        connecting = np.where(np.any(DBEdges == current_node, axis=0))[0]
        if len(connecting) == 0:
            break

        next_idx = connecting[0]
        next_edge = DBEdges[:, next_idx]

        oriented = find_edge_orient(next_edge.copy(), MeshTri, MeshNodes)
        if oriented[0] != current_node:
            oriented = oriented[[1, 0]]

        cont_edges.append(oriented)
        current_node = oriented[1]
        DBEdges = np.delete(DBEdges, next_idx, axis=1)

    result = np.column_stack(cont_edges) if cont_edges else np.array([[]])
    debug_print(f"        Connected {len(cont_edges)} edges into continuous boundary", level=5)

    return result

def cleanup_gmsh():
    """Очистка ресурсов Gmsh"""
    global gmsh, PYGMSH_AVAILABLE
    if PYGMSH_AVAILABLE and gmsh and gmsh.isInitialized():
        gmsh.finalize()
        debug_print("[OK] Gmsh resources cleaned up", level=3)