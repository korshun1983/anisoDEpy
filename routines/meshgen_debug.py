# routines/meshgen_debug.py
"""
===============================================================================
DEBUG Mesh Generation - Only Inner Domain
===============================================================================
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any
import gmsh

logger = logging.getLogger(__name__)


def prepare_mesh_debug_only_inner(CompStruct: Any) -> Dict[str, Any]:
    """
    Генерирует сетку ТОЛЬКО для внутреннего домена (первый в списке)
    """
    logger.info("        [DEBUG MODE] Generating mesh ONLY for inner domain...")

    # Инициализируем gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    model = gmsh.model()
    model.add("debug_inner_domain")

    # === ИЗВЛЕЧЕНИЕ ТОЛЬКО ПЕРВОГО ДОМЕНА ===
    domain_rx = np.array(CompStruct.Model['DomainRx'])
    domain_ry = np.array(CompStruct.Model['DomainRy'])

    if len(domain_rx) < 2:
        raise ValueError("Need at least 2 radii for debug mode")

    inner_radius = domain_rx[0]
    logger.info(f"      Inner domain radius: {inner_radius} m")

    # === ГЕОМЕТРИЯ: только внутренний цилиндр ===
    factory = model.occ

    # Создаем сплошной цилиндр от 0 до inner_radius
    if inner_radius > 0:
        inner_disk = factory.addDisk(0, 0, 0, inner_radius, inner_radius)
    else:
        inner_disk = factory.addDisk(0, 0, 0, 0.1, 0.1)

    factory.synchronize()

    # === НАСТРОЙКИ СЕТКИ ===
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

    # === ГЕНЕРАЦИЯ ===
    logger.info(f"      Generating mesh with hmax={hmax}...")
    model.mesh.generate(2)

    # === ЭКСТРАКЦИЯ ДАННЫХ ===
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

    # === ОТЛАДКА: показать 3D окно gmsh ===
    logger.info("      Opening GMSH window to visualize mesh...")
    gmsh.fltk.initialize()
    gmsh.fltk.run()

    # === НАЗНАЧЕНИЕ ДОМЕННЫХ МАРКЕРОВ (только Domain 1) ===
    domain_markers = np.ones(triangles.shape[1], dtype=int)
    logger.info(f"      All elements assigned to Domain 1")
    logger.info(f"      Mesh generated: {MeshNodes.shape[1]} nodes, {triangles.shape[1]} elements")

    gmsh.finalize()

    MeshProps = create_mesh_props(MeshNodes, triangles, CompStruct)

    return {
        'MeshNodes': MeshNodes,
        'BoundaryEdges': np.empty((3, 0), dtype=int),
        'MeshTri': np.vstack([triangles, domain_markers.reshape(1, -1)]),
        'MeshProps': MeshProps
    }


def prepare_mesh_debug_only_outer(CompStruct: Any) -> Dict[str, Any]:
    """
    Генерирует сетку ТОЛЬКО для ВТОРОГО (внешнего) домена (полый цилиндр)
    Для проверки генерации полого слоя
    """
    logger.info("        [DEBUG MODE] Generating mesh ONLY for OUTER domain...")

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    model = gmsh.model()
    model.add("debug_outer_domain")

    # === ИЗВЛЕЧЕНИЕ ДВУХ ПЕРВЫХ РАДИУСОВ ===
    domain_rx = np.array(CompStruct.Model['DomainRx'])
    domain_ry = np.array(CompStruct.Model['DomainRy'])

    if len(domain_rx) < 3:
        raise ValueError("Need at least 3 radii for outer domain debug mode (0, r1, r2)")

    r_inner = domain_rx[1]  # Внешний радиус первого домена = внутренний второго
    r_outer = domain_rx[2]  # Внешний радиус второго домена

    logger.info(f"      Outer domain: r_inner={r_inner} m, r_outer={r_outer} m")
    logger.info(f"      Generating HOLLOW cylinder mesh")

    # === ГЕОМЕТРИЯ: полый цилиндр ===
    factory = model.occ

    # Создаем внешний и внутренний диски
    inner_disk = factory.addDisk(0, 0, 0, r_inner, r_inner)
    outer_disk = factory.addDisk(0, 0, 0, r_outer, r_outer)

    # Вычитаем внутренний из внешнего
    ring_dim_tags, _ = factory.cut(
        [(2, outer_disk)], [(2, inner_disk)],
        removeObject=True, removeTool=False
    )

    if len(ring_dim_tags) == 0:
        raise RuntimeError("Failed to create hollow cylinder geometry")

    factory.synchronize()

    # === НАСТРОЙКИ СЕТКИ ===
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

    # === ГЕНЕРАЦИЯ ===
    logger.info(f"      Generating hollow cylinder mesh with hmax={hmax}...")
    model.mesh.generate(2)

    # === ЭКСТРАКЦИЯ ДАННЫХ ===
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

    # === ОТЛАДКА: показать 3D окно gmsh ===
    logger.info("      Opening GMSH window to visualize HOLLOW cylinder...")
    gmsh.fltk.initialize()
    gmsh.fltk.run()

    # === НАЗНАЧЕНИЕ ДОМЕННЫХ МАРКЕРОВ (только Domain 2) ===
    domain_markers = np.full(triangles.shape[1], 2, dtype=int)
    logger.info(f"      All elements assigned to Domain 2 (OUTER)")
    logger.info(f"      Mesh generated: {MeshNodes.shape[1]} nodes, {triangles.shape[1]} elements")

    gmsh.finalize()

    MeshProps = create_mesh_props(MeshNodes, triangles, CompStruct)

    return {
        'MeshNodes': MeshNodes,
        'BoundaryEdges': np.empty((3, 0), dtype=int),
        'MeshTri': np.vstack([triangles, domain_markers.reshape(1, -1)]),
        'MeshProps': MeshProps
    }

def create_mesh_props(MeshNodes: np.ndarray, MeshTri: np.ndarray,
                      CompStruct: Any) -> Dict[str, np.ndarray]:
    """Compute mesh properties"""
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