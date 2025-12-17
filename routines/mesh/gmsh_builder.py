# D:\Работа\python\anisoDEpy\routines\mesh\gmsh_builder.py
"""
GMSH mesh builder for SAFE
Uses pre-generated boundary geometry from prepare_mesh_bh
"""

import gmsh
import numpy as np
from typing import Dict, Tuple, Optional
from utils import debug_print


def get_min_velocity(CompStruct: Dict) -> float:
    """Calculate minimum wave velocity in the model for mesh sizing."""
    Model = CompStruct['Model']
    min_velocity = float('inf')

    for i, domain_type in enumerate(Model['DomainType']):
        params = Model['DomainParam'][i]
        if domain_type.lower() == 'fluid':
            rho = params[0]
            K = params[1]  # Bulk modulus in Pa
            velocity = np.sqrt(K / rho)
            debug_print(f"      Domain {i} (fluid): vp = {velocity:.1f} m/s", level=5)
        else:  # HTTI solid
            rho = params[0]
            c66 = params[5]  # Shear modulus in Pa
            velocity = np.sqrt(c66 / rho)
            debug_print(f"      Domain {i} (HTTI): vs ≈ {velocity:.1f} m/s", level=5)

        if velocity < min_velocity:
            min_velocity = velocity

    if min_velocity == float('inf'):
        raise ValueError("Could not calculate min_velocity - check DomainParam")

    debug_print(f"    Minimum velocity: {min_velocity:.1f} m/s", level=4)
    return min_velocity


def debug_domain_calculation(CompStruct: Dict, frequency: float, min_velocity: float):
    """Debug function to show domain radius calculations."""
    Model = CompStruct['Model']

    debug_print("=" * 60, level=4)
    debug_print(f"DEBUG: Domain Calculation for {frequency} kHz", level=4)
    debug_print("=" * 60, level=4)

    debug_print(f"  Original parameters:", level=4)
    debug_print(f"    DomainRx: {Model['DomainRx']}", level=4)
    debug_print(f"    DomainRy: {Model['DomainRy']}", level=4)
    debug_print(f"    AddDomainLoc: {Model['AddDomainLoc']}", level=4)
    debug_print(f"    AddDomainType: {Model['AddDomainType']}", level=4)
    debug_print(f"    AddDomainL: {Model['AddDomainL']}", level=4)
    debug_print(f"    LDomain_in_LSH: {Model.get('LDomain_in_LSH', 'none')}", level=4)

    wavelength = min_velocity / (frequency * 1000)  # kHz to Hz
    debug_print(f"  Calculation parameters:", level=4)
    debug_print(f"    Frequency: {frequency} kHz", level=4)
    debug_print(f"    Min velocity: {min_velocity:.1f} m/s", level=4)
    debug_print(f"    Wavelength: {wavelength:.6f} m", level=4)

    domain_rx = Model['DomainRx'].copy()
    domain_ry = Model['DomainRy'].copy()

    if 'AddDomainType' in Model and Model['AddDomainType'].lower() != 'none':
        if Model['AddDomainLoc'].lower() == 'ext':
            abc_rx = domain_rx[-1] + Model['AddDomainL']
            abc_ry = domain_ry[-1] + Model['AddDomainL']
            domain_rx = np.append(domain_rx, abc_rx)
            domain_ry = np.append(domain_ry, abc_ry)

    debug_print(f"  Final domain radii:", level=4)
    for i, (rx, ry) in enumerate(zip(domain_rx, domain_ry)):
        domain_type = "main" if i < len(Model['DomainType']) else "ABC"
        debug_print(f"    Domain {i + 1} ({domain_type}): Rx = {rx:.3f}, Ry = {ry:.3f}", level=4)

    debug_print("=" * 60, level=4)


def _inpolygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Point-in-polygon test using ray casting algorithm."""
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


def _assign_domains(centroids: np.ndarray, nodes: np.ndarray, edges: np.ndarray,
                    domain_faces: Dict) -> np.ndarray:
    """Assign domain numbers to triangles by checking if centroid is inside each domain polygon."""
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


def build_mesh_gmsh(CompStruct: Dict, nodes: Optional[np.ndarray] = None,
                    edges: Optional[np.ndarray] = None,
                    domain_faces: Optional[Dict] = None,
                    frequency: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Build 2-D mesh using Gmsh optimized for SAFE.
    Compatible with meshfaces() API - accepts pre-generated boundary geometry.
    If nodes/edges/domain_faces not provided, uses internal generation.
    """
    try:
        # Initialize Gmsh
        if not gmsh.isInitialized():
            gmsh.initialize()
            debug_print("    Gmsh: Initializing...", level=4)
        else:
            debug_print("    Gmsh: Reusing existing instance...", level=3)

        gmsh.model.remove()
        gmsh.model.add("WaveGuide")

        Model = CompStruct['Model']
        Mesh = CompStruct.get('Mesh', {})
        hmax = Mesh.get('hmax', 0.16)

        # Choose generation method
        if nodes is not None and edges is not None and domain_faces is not None:
            debug_print(f"    Using {len(nodes)} boundary nodes, {len(edges)} edges", level=4)
            return _build_from_boundary_geometry(nodes, edges, domain_faces, hmax, CompStruct)
        else:
            debug_print("    No boundary geometry provided, using internal generation", level=3)
            return _build_from_internal_geometry(CompStruct, frequency)

    except Exception as e:
        debug_print(f"ERROR in Gmsh builder: {e}", level=0)
        import traceback
        traceback.print_exc()
        raise


def _build_from_boundary_geometry(nodes: np.ndarray, edges: np.ndarray,
                                  domain_faces: Dict, hmax: float,
                                  CompStruct: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Build mesh from pre-generated boundary geometry."""

    Model = CompStruct['Model']

    # === 1. Create points ===
    point_tags = []
    for i, (x, y) in enumerate(nodes):
        tag = gmsh.model.occ.addPoint(x, y, 0.0, meshSize=hmax)
        point_tags.append(tag)

    # === 2. Create surfaces for each domain ===
    surface_tags = []
    domain_centers = []

    for domain_id in sorted(domain_faces.keys()):
        face_edges = domain_faces[domain_id]
        curve_tags = []

        for edge_idx in face_edges:
            edge_idx_0 = edge_idx - 1
            if edge_idx_0 >= len(edges):
                debug_print(f"ERROR: Edge index {edge_idx} out of bounds!", level=0)
                continue

            start_idx, end_idx = edges[edge_idx_0]
            if start_idx >= len(point_tags) or end_idx >= len(point_tags):
                debug_print(f"ERROR: Point index out of bounds in edge {edge_idx}", level=0)
                continue

            line_tag = gmsh.model.occ.addLine(point_tags[start_idx], point_tags[end_idx])
            curve_tags.append(line_tag)

        if len(curve_tags) < 3:
            debug_print(f"Warning: Domain {domain_id} has only {len(curve_tags)} curves, skipping", level=1)
            continue

        curve_loop = gmsh.model.occ.addCurveLoop(curve_tags)
        surface_tag = gmsh.model.occ.addPlaneSurface([curve_loop])

        surface_tags.append((2, surface_tag))

        com = gmsh.model.occ.getCenterOfMass(2, surface_tag)
        domain_centers.append(np.sqrt(com[0] ** 2 + com[1] ** 2))

        debug_print(f"      Domain {domain_id}: {len(curve_tags)} curves → surface {surface_tag}", level=5)

    if not surface_tags:
        raise RuntimeError("No valid surfaces created from boundary geometry!")

        # === 3. Sort surfaces by radius (inner to outer) ===
    gmsh.model.occ.synchronize()

    # Получаем информацию о поверхностях: (радиус, tag)
    surface_info = []
    for (dim, tag) in surface_tags:
        try:
            com = gmsh.model.occ.getCenterOfMass(2, tag)
            radius = np.sqrt(com[0] ** 2 + com[1] ** 2)
            surface_info.append((radius, tag, (2, tag)))
            debug_print(f"      Entity {tag}: radius={radius:.3f}", level=5)
        except:
            debug_print(f"Warning: Could not get center for entity {tag}", level=4)
            surface_info.append((float('inf'), tag, (2, tag)))

    # Сортируем по радиусу (от внутреннего к внешнему)
    surface_info.sort(key=lambda x: x[0])

    # Создаем список отсортированных поверхностей для fragment
    sorted_surfaces = [info[2] for info in surface_info]  # (2, tag) формат

    # === 3a. Create ring structure (FIXED: make outer domain a ring) ===
    gmsh.model.occ.synchronize()

    if len(sorted_surfaces) > 1:
        debug_print(f"    Creating ring structure from {len(sorted_surfaces)} surfaces...", level=4)

        # Используем fragment для разбиения на непересекающиеся части
        result = gmsh.model.occ.fragment(sorted_surfaces, [],
                                         removeObject=True, removeTool=False)
        gmsh.model.occ.synchronize()

        # result[0] содержит ВСЕ финальные сущности (включая кольцо)
        final_surfaces = result[0] if result[0] else sorted_surfaces
        debug_print(f"      Created {len(final_surfaces)} non-overlapping entities", level=5)
    else:
        final_surfaces = sorted_surfaces
        debug_print(f"    Skipping boolean operations for single surface", level=4)

    # === 3b. Save geometry for debug ===
    if CompStruct.get('Advanced', {}).get('VisualizeGeometry', False):
        geo_file = 'geometry_debug.brep'
        gmsh.write(geo_file)
        debug_print(f"    ✓ Geometry saved to {geo_file}", level=2)

        try:
            debug_print(f"    Opening Gmsh GUI...", level=3)
            gmsh.fltk.run()
        except Exception as e:
            debug_print(f"    Could not open GUI: {e}", level=4)

    # === 4. Create physical groups by radius ===
    gmsh.model.occ.synchronize()

    # Пересчитываем финальные сущности
    all_surfaces = gmsh.model.getEntities(2)
    debug_print(f"    Creating physical groups for {len(all_surfaces)} domains...", level=4)

    # Для каждой сущности определяем домен по радиусу
    final_surface_info = []
    for (dim, tag) in all_surfaces:
        try:
            com = gmsh.model.occ.getCenterOfMass(2, tag)
            radius = np.sqrt(com[0] ** 2 + com[1] ** 2)
            final_surface_info.append((radius, tag))
            debug_print(f"      Entity {tag}: radius={radius:.3f}", level=5)
        except:
            final_surface_info.append((float('inf'), tag))

    # Сортируем по радиусу
    final_surface_info.sort(key=lambda x: x[0])

    # Создаем physical groups в порядке возрастания радиуса
    for i, (radius, tag) in enumerate(final_surface_info):
        gmsh.model.addPhysicalGroup(2, [tag], i + 1)

        # Определяем тип домена по порядку
        if i < len(Model['DomainType']):
            domain_type = Model['DomainType'][i]
            domain_name = f"domain_{i + 1}_{domain_type}"
        else:
            domain_name = f"abc_domain_{i + 1}"

        gmsh.model.setPhysicalName(2, i + 1, domain_name)
        debug_print(f"      Physical group {i + 1}: {domain_name}, radius={radius:.3f}", level=5)

    # === 5. Mesh generation parameters ===
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", hmax)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", hmax * 0.5)
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
    gmsh.option.setNumber("Mesh.ElementOrder", 2)  # Quadratic elements

    debug_print("    Gmsh: Generating mesh...", level=4)
    gmsh.model.mesh.generate(2)
    debug_print("    Gmsh: Mesh generation complete", level=3)

    # === 6. Extract mesh data ===
    node_tags, coord, _ = gmsh.model.mesh.getNodes()
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements()

    n_nodes = len(coord) // 3
    MeshNodes = coord.reshape(n_nodes, 3).T[:2, :]

    # >>>>>>>>>> КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ <<<<<<<<<<
    debug_print(f"    Raw mesh data: {n_nodes} nodes, {len(elem_types)} element types", level=4)
    debug_print(f"    Element types: {list(elem_types)}", level=4)

    # НАЙТИ именно треугольники (Tri3=2, Tri6=9)
    tri_idx = -1
    for idx, etype in enumerate(elem_types):
        if etype in [2, 9]:  # Ищем ЛЮБЫЕ треугольники
            tri_idx = idx
            debug_print(f"    Found triangle elements: type={etype} at index {idx}", level=4)
            break

    if tri_idx == -1:
        debug_print("    ERROR: No triangle elements found in mesh!", level=0)
        debug_print(f"    Available element types: {list(elem_types)}", level=0)
        debug_print(f"    This usually means 1D elements (lines) were generated instead of 2D", level=0)
        raise RuntimeError("Gmsh failed to generate 2D triangle elements. Check geometry validity.")

    # Извлекаем треугольники
    if elem_types[tri_idx] == 9:  # Tri6
        tri_nodes = elem_node_tags[tri_idx].reshape(-1, 6)[:, :3]
        debug_print(f"    Using Tri6 elements: {len(tri_nodes)} triangles", level=4)
    elif elem_types[tri_idx] == 2:  # Tri3
        tri_nodes = elem_node_tags[tri_idx].reshape(-1, 3)
        debug_print(f"    Using Tri3 elements: {len(tri_nodes)} triangles", level=4)
    else:
        raise RuntimeError(f"Unexpected triangle element type: {elem_types[tri_idx]}")
    # >>>>>>>>>> КОНЕЦ ИСПРАВЛЕНИЯ <<<<<<<<<<

    n_elem = len(tri_nodes)
    debug_print(f"    Extracted {n_elem} triangles, {n_nodes} nodes", level=4)

    # === 7. Assign domain numbers to elements ===
    MeshFaceNums = np.zeros(n_elem, dtype=int)

    # Get all entities of size 2 (surfaces)
    all_entities = gmsh.model.getEntities(2)
    entity_tags = [ent[1] for ent in all_entities]

    debug_print(f"    Available entities after boolean ops: {entity_tags}", level=4)

    for i_elem in range(n_elem):
        elem_tag = elem_tags[tri_idx][i_elem]
        elem_info = gmsh.model.mesh.getElement(elem_tag)
        entity_tag = elem_info[1][0]

        # ✓ ПРОВЕРКА: сущность существует и имеет physical group
        if entity_tag in entity_tags:
            phys_groups = gmsh.model.getPhysicalGroupsForEntity(2, entity_tag)
            if phys_groups and len(phys_groups) > 0:
                domain_idx = int(phys_groups[0]) - 1
                MeshFaceNums[i_elem] = domain_idx
            else:
                MeshFaceNums[i_elem] = 0  # Default domain
                debug_print(f"Warning: Entity {entity_tag} has no physical group", level=4)
        else:
            MeshFaceNums[i_elem] = 0  # Default domain
            debug_print(f"Warning: Entity {entity_tag} not found after boolean ops", level=4)

    # === 8. Build MeshTri array ===
    MeshTri = np.vstack([
        tri_nodes.T,  # 1-based node indices
        MeshFaceNums  # Domain numbers
    ]).astype(int)

    debug_print(f"    Generated MeshNodes: {MeshNodes.shape[1]} nodes", level=4)
    debug_print(f"    Generated MeshTri: {MeshTri.shape[1]} elements", level=4)
    debug_print(f"    Domain distribution: {np.bincount(MeshFaceNums)}", level=4)

    return MeshNodes, MeshTri, MeshFaceNums, CompStruct


def _build_from_internal_geometry(CompStruct: Dict, frequency: Optional[float]) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Build mesh from model parameters without boundary geometry."""
    debug_print("    Internal geometry generation from model parameters...", level=3)

    Model = CompStruct['Model']
    Mesh = CompStruct.get('Mesh', {})

    # Determine target frequency
    if frequency is None:
        freq_range = Model.get('f_array_range')
        if freq_range:
            frequency = freq_range.get('end', 15.0)
        else:
            frequency = Model.get('f_array', [1.0])[-1]

    # Calculate wavelength-based mesh size
    min_velocity = get_min_velocity(CompStruct)
    wavelength = min_velocity / (frequency * 1000)  # kHz to Hz
    hmax = wavelength / 6

    debug_print(f"    Gmsh: frequency={frequency} kHz, λ={wavelength:.6f}m, hmax={hmax:.6f}m", level=4)

    # Domain parameters
    domain_rx = Model['DomainRx'].copy()
    domain_ry = Model['DomainRy'].copy()
    domain_theta = Model['DomainTheta'].copy()
    domain_ecc = Model['DomainEcc'].copy()
    domain_ecc_angle = Model['DomainEccAngle'].copy()

    # Apply wavelength scaling
    ldomain_in_lsh = Model.get('LDomain_in_LSH', 'none').lower()
    add_domain_exist = 'AddDomainType' in Model and Model['AddDomainType'].lower() != 'none'

    if ldomain_in_lsh == 'yes' and add_domain_exist:
        if Model['AddDomainLoc'].lower() == 'ext':
            domain_rx[-1] = domain_rx[-2] + domain_rx[-1] * wavelength
            domain_ry[-1] = domain_ry[-2] + domain_ry[-1] * wavelength

            abc_rx = domain_rx[-1] + Model['AddDomainL'] * wavelength
            abc_ry = domain_ry[-1] + Model['AddDomainL'] * wavelength
            domain_rx = np.append(domain_rx, abc_rx)
            domain_ry = np.append(domain_ry, abc_ry)

    elif add_domain_exist:
        if Model['AddDomainLoc'].lower() == 'ext':
            abc_rx = domain_rx[-1] + Model['AddDomainL']
            abc_ry = domain_ry[-1] + Model['AddDomainL']
            domain_rx = np.append(domain_rx, abc_rx)
            domain_ry = np.append(domain_ry, abc_ry)

    # Create surfaces
    surface_tags = []
    for i in range(len(domain_rx)):
        rx = domain_rx[i]
        ry = domain_ry[i]
        ecc = domain_ecc[i] if i < len(domain_ecc) else 0
        ang = domain_ecc_angle[i] * np.pi / 180 if i < len(domain_ecc_angle) else 0

        tag = gmsh.model.occ.addDisk(0, 0, 0, rx, ry)

        if ecc != 0:
            dx = ecc * np.cos(ang)
            dy = ecc * np.sin(ang)
            gmsh.model.occ.translate([(2, tag)], dx, dy, 0)

        surface_tags.append(tag)

    gmsh.model.occ.synchronize()

    # Boolean operations
    if len(surface_tags) > 1:
        sorted_tags = sorted(surface_tags, reverse=True)
        for i in range(len(sorted_tags) - 1):
            gmsh.model.occ.cut([(2, sorted_tags[i])], [(2, sorted_tags[i + 1])],
                               removeObject=True, removeTool=False)
            gmsh.model.occ.synchronize()

    # Physical groups
    all_surfaces = gmsh.model.getEntities(2)
    for i, (dim, tag) in enumerate(all_surfaces):
        gmsh.model.addPhysicalGroup(2, [tag], i + 1)

    # Mesh
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", hmax)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.model.mesh.generate(2)

    # Extract (same as above, copy-paste the CRITICAL FIX here too)
    node_tags, coord, _ = gmsh.model.mesh.getNodes()
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements()

    n_nodes = len(coord) // 3
    MeshNodes = coord.reshape(n_nodes, 3).T[:2, :]

    # >>>>>>>>>> СКОПИРУЙТЕ СЮДА ТО ЖЕ ИСПРАВЛЕНИЕ <<<<<<<<<<
    debug_print(f"    Raw mesh data: {n_nodes} nodes, {len(elem_types)} element types", level=4)
    debug_print(f"    Element types: {list(elem_types)}", level=4)

    tri_idx = -1
    for idx, etype in enumerate(elem_types):
        if etype in [2, 9]:
            tri_idx = idx
            debug_print(f"    Found triangle elements: type={etype} at index {idx}", level=4)
            break

    if tri_idx == -1:
        debug_print("    ERROR: No triangle elements found in mesh!", level=0)
        debug_print(f"    Available element types: {list(elem_types)}", level=0)
        raise RuntimeError("Gmsh failed to generate 2D triangle elements.")

    if elem_types[tri_idx] == 9:
        tri_nodes = elem_node_tags[tri_idx].reshape(-1, 6)[:, :3]
        debug_print(f"    Using Tri6 elements: {len(tri_nodes)} triangles", level=4)
    elif elem_types[tri_idx] == 2:
        tri_nodes = elem_node_tags[tri_idx].reshape(-1, 3)
        debug_print(f"    Using Tri3 elements: {len(tri_nodes)} triangles", level=4)
    else:
        raise RuntimeError(f"Unexpected triangle element type: {elem_types[tri_idx]}")
    # >>>>>>>>>> КОНЕЦ ИСПРАВЛЕНИЯ <<<<<<<<<<

    n_elem = len(tri_nodes)

    MeshFaceNums = np.zeros(n_elem, dtype=int)
    for i_elem in range(n_elem):
        try:
            elem_tag = elem_tags[tri_idx][i_elem]
            elem_info = gmsh.model.mesh.getElement(elem_tag)
            entity_tag = elem_info[1][0]

            phys_groups = gmsh.model.getPhysicalGroupsForEntity(2, entity_tag)

            if phys_groups is not None and len(phys_groups) > 0:
                domain_idx = int(phys_groups[0]) - 1
                MeshFaceNums[i_elem] = domain_idx
            else:
                MeshFaceNums[i_elem] = len(all_surfaces) - 1

        except Exception as e:
            debug_print(f"Warning: Could not get domain for element {i_elem}: {e}", level=4)
            MeshFaceNums[i_elem] = 0

    MeshTri = np.vstack([
        tri_nodes.T,
        MeshFaceNums
    ]).astype(int)

    debug_print(f"    Generated mesh: {MeshNodes.shape[1]} nodes, {n_elem} elements", level=4)
    debug_print(f"    Domain distribution: {np.bincount(MeshFaceNums)}", level=4)

    return MeshNodes, MeshTri, MeshFaceNums, CompStruct


def cleanup_gmsh():
    """Clean Gmsh resources safely"""
    try:
        if gmsh.isInitialized():
            gmsh.finalize()
            debug_print("[OK] Gmsh resources cleaned up", level=3)
    except Exception as e:
        debug_print(f"Warning: Error finalizing Gmsh: {e}", level=4)