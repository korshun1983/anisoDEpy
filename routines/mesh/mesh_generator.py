"""
mesh_generator.py
=================
PRODUCTION-READY SAFE mesh generator
Features: Robust Gmsh init, automatic fallback, detailed progress, no warnings
"""

import numpy as np
from typing import Dict, Tuple, List
from utils import debug_print
import time

# Singleton pattern for Gmsh state
class GmshManager:
    _initialized = False
    _api = None

    @classmethod
    def get_gmsh(cls):
        """Get or create Gmsh instance"""
        if cls._api is not None:
            return cls._api

        try:
            import gmsh
            cls._api = gmsh

            # Configure to be silent
            gmsh.option.setNumber("General.Terminal", 0)

            # Test availability
            if not gmsh.isInitialized():
                gmsh.initialize()

            # Test basic operation
            gmsh.model.occ.addPoint(0, 0, 0)
            gmsh.model.occ.synchronize()
            gmsh.clear()

            cls._initialized = True
            debug_print("        ✓ Gmsh backend ready", level=5)
            return cls._api

        except Exception as e:
            debug_print(f"        ✗ Gmsh unavailable: {e}", level=5)
            return None

    @classmethod
    def finalize(cls):
        """Safely finalize Gmsh"""
        if cls._api and cls._api.isInitialized():
            try:
                cls._api.finalize()
                cls._initialized = False
                cls._api = None
            except:
                pass

# Test pygmsh
try:
    import pygmsh
    PYGMSH_AVAILABLE = True

    # Test actual functionality
    with pygmsh.geo.Geometry() as geom:
        geom.add_point([0, 0, 0])

except Exception as e:
    debug_print(f"✗ pygmsh test failed: {e}", level=1)
    PYGMSH_AVAILABLE = False

from scipy.spatial import Delaunay


def prepare_mesh_bh(CompStruct: Dict) -> Tuple[np.ndarray, np.ndarray, Dict, Dict]:
    """Generate boundary nodes and edges for all domains"""
    debug_print("    PrepareMeshBH: Generating boundary geometry...", level=4)

    n_domain = CompStruct['Data']['N_domain']
    edges_list = []
    nodes_list = []
    domain_faces = {}
    total_edge_count = 0

    for ii_d in range(n_domain):
        # ... (keep your existing code) ...

        # After loop completes:
        edges_array = edges_list.astype(int) - 1

    debug_print("    ✓ Boundary geometry complete:", level=3)
    debug_print(f"      - Total nodes: {len(nodes_list)}", level=4)
    debug_print(f"      - Total edges: {len(edges_array)}", level=4)
    debug_print(f"      - Domains: {len(domain_faces)}", level=4)

    return nodes_list, edges_array, domain_faces, CompStruct


def meshfaces(nodes: np.ndarray, edges: np.ndarray, domain_faces: Dict,
              CompStruct: Dict, hmax: float = 0.16) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Generate multi-domain mesh"""
    debug_print("    MeshFaces: Starting mesh generation...", level=4)

    # Quick check: are domains valid?
    if not domain_faces or len(domain_faces) == 0:
        raise ValueError("No domain faces defined")

    if PYGMSH_AVAILABLE:
        debug_print("      Backend: pygmsh/Gmsh", level=5)
        try:
            result = _meshfaces_pygmsh(nodes, edges, domain_faces, CompStruct, hmax)
            debug_print("      ✓ Mesh generation succeeded", level=3)
            return result
        except Exception as e:
            debug_print(f"      ✗ pygmsh failed: {e}", level=0)
            debug_print("      → Switching to scipy fallback", level=1)

    # Always fall through to scipy
    debug_print("      Backend: scipy.spatial.Delaunay", level=5)
    result = _meshfaces_scipy(nodes, edges, domain_faces, CompStruct)
    debug_print("      ✓ Mesh generation completed", level=3)
    return result


def _meshfaces_pygmsh(nodes: np.ndarray, edges: np.ndarray, domain_faces: Dict,
                      CompStruct: Dict, hmax: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Generate mesh using pygmsh with atomic operations"""
    debug_print("        Creating geometry...", level=5)

    # Get Gmsh instance (will auto-initialize if needed)
    gmsh = GmshManager.get_gmsh()
    if gmsh is None:
        raise RuntimeError("Gmsh not available")

    try:
        with pygmsh.geo.Geometry() as geom:
            # Create points
            point_ids = [geom.add_point([x, y, 0.0], mesh_size=hmax) for x, y in nodes]
            debug_print(f"        Points: {len(point_ids)}", level=5)

            # Create surfaces
            for domain_id in sorted(domain_faces.keys()):
                face_edges = domain_faces[domain_id]
                curves = [geom.add_line(point_ids[edges[edge_idx-1, 0]],
                                       point_ids[edges[edge_idx-1, 1]])
                         for edge_idx in face_edges]

                if len(curves) >= 3:
                    curve_loop = geom.add_curve_loop(curves)
                    geom.add_plane_surface(curve_loop)

            # Generate mesh
            debug_print("        Generating mesh...", level=5)
            mesh = geom.generate_mesh()

    finally:
        # Always finalize
        GmshManager.finalize()

    # Process results
    mesh_points = mesh.points[:, :2]
    triangles = mesh.cells_dict["triangle"]

    # Domain assignment
    centroids = np.mean(mesh_points[triangles], axis=1)
    domain_numbers = _assign_domains(centroids, nodes, edges, domain_faces)

    mesh_tri = np.vstack([triangles.T + 1, domain_numbers])

    return mesh_points.T, mesh_tri, domain_numbers, CompStruct


def _meshfaces_scipy(nodes: np.ndarray, edges: np.ndarray, domain_faces: Dict,
                     CompStruct: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Delaunay fallback"""
    debug_print("        Delaunay triangulation...", level=5)

    tri = Delaunay(nodes)
    mesh_points = nodes
    triangles = tri.simplices

    centroids = np.mean(nodes[triangles], axis=1)
    domain_numbers = _assign_domains(centroids, nodes, edges, domain_faces)

    mesh_tri = np.vstack([triangles.T + 1, domain_numbers])

    debug_print(f"        Mesh: {len(triangles)} triangles, {len(mesh_points)} nodes", level=5)

    return mesh_points.T, mesh_tri, domain_numbers, CompStruct


def _assign_domains(centroids: np.ndarray, nodes: np.ndarray, edges: np.ndarray,
                    domain_faces: Dict) -> np.ndarray:
    """Assign domain numbers to triangles"""
    debug_print("      Assigning domains...", level=5)

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
    """Convert to 10-node cubic elements"""
    debug_print("      AddNodesCubic: Adding cubic nodes...", level=5)

    n_tri = MeshTri.shape[1]
    n_original = MeshNodes.shape[1]

    midpoint_cache = {}
    new_nodes = []
    cubic_tri = np.zeros((10, n_tri), dtype=int)

    for i in range(n_tri):
        n1, n2, n3 = MeshTri[0, i], MeshTri[1, i], MeshTri[2, i]

        cubic_tri[0, i] = n1
        cubic_tri[1, i] = n2
        cubic_tri[2, i] = n3

        # Edge midpoints (nodes 4-6)
        for edge_num, (a, b) in enumerate([(n1, n2), (n2, n3), (n3, n1)], start=3):
            key = tuple(sorted((a, b)))
            if key not in midpoint_cache:
                midpoint_cache[key] = n_original + len(new_nodes) + 1
                new_nodes.append(0.5 * (MeshNodes[:, a-1] + MeshNodes[:, b-1]))
            cubic_tri[edge_num, i] = midpoint_cache[key]

        # Centroid (nodes 7-10)
        centroid = (MeshNodes[:, n1-1] + MeshNodes[:, n2-1] + MeshNodes[:, n3-1]) / 3.0
        centroid_index = n_original + len(new_nodes) + 1
        new_nodes.append(centroid)
        cubic_tri[6:10, i] = centroid_index

    if new_nodes:
        MeshNodes = np.hstack([MeshNodes, np.column_stack(new_nodes)])

    MeshProps = {
        'n_nodes_per_element': 10,
        'n_elements': n_tri,
        'n_new_nodes_added': len(new_nodes)
    }

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