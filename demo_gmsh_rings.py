#!/usr/bin/env python3
"""
demo_gmsh_rings.py
==================
Standalone demo showing CORRECT way to create nested domains for SAFE.
Key point: Use SINGLE mesh, assign domains by centroid location.
"""

import gmsh
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def create_unified_mesh(radii: list = [1.0, 2.0, 3.0]):
    """
    Create ONE mesh covering all domains, then assign by centroid location.
    Returns 0-based indices for internal consistency.
    """
    print(f"Creating unified mesh for radii: {radii}")

    gmsh.initialize()
    gmsh.model.add("WaveGuide")

    # === 1. Create outermost boundary ===
    r_max = radii[-1]
    center = gmsh.model.occ.addPoint(0, 0, 0)

    # Outer circle points (16 segments for smoothness)
    n_points = 16
    points = []
    arcs = []
    for i in range(n_points):
        angle = i * 2 * np.pi / n_points
        x = r_max * np.cos(angle)
        y = r_max * np.sin(angle)
        p = gmsh.model.occ.addPoint(x, y, 0)
        points.append(p)

    # Outer circle arcs
    for i in range(n_points):
        p1 = points[i]
        p2 = points[(i + 1) % n_points]
        arc = gmsh.model.occ.addCircleArc(p1, center, p2)
        arcs.append(arc)

    # Create outer loop and surface
    outer_loop = gmsh.model.occ.addCurveLoop(arcs)
    outer_surface = gmsh.model.occ.addPlaneSurface([outer_loop])

    gmsh.model.occ.synchronize()

    # === 2. Generate mesh ===
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.3)
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
    gmsh.option.setNumber("Mesh.ElementOrder", 2)  # Quadratic elements

    print(f"  Generating mesh...")
    gmsh.model.mesh.generate(2)

    # === 3. Extract mesh data (0-based indices) ===
    node_tags, coord, _ = gmsh.model.mesh.getNodes()
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements()

    n_nodes = len(coord) // 3
    nodes = coord.reshape(n_nodes, 3)[:, :2]

    # Find triangles
    tri_idx = -1
    for idx, etype in enumerate(elem_types):
        if etype in [2, 9]:
            tri_idx = idx
            break

    if tri_idx == -1:
        raise RuntimeError("No triangles found!")

    # Get triangles (0-based)
    if elem_types[tri_idx] == 9:  # Tri6
        triangles = elem_node_tags[tri_idx].reshape(-1, 6)[:, :3] - 1
    else:  # Tri3
        triangles = elem_node_tags[tri_idx].reshape(-1, 3) - 1

    n_elem = len(triangles)

    # === 4. Assign domains by centroid location ===
    domain_numbers = np.zeros(n_elem, dtype=int)
    centroids = np.mean(nodes[triangles], axis=1)
    centroid_radius = np.sqrt(centroids[:, 0] ** 2 + centroids[:, 1] ** 2)

    # 0: r < radii[0], 1: radii[0] <= r < radii[1], 2: radii[1] <= r <= radii[2]
    domain_numbers[centroid_radius <= radii[0]] = 0
    domain_numbers[(centroid_radius > radii[0]) & (centroid_radius <= radii[1])] = 1
    domain_numbers[(centroid_radius > radii[1]) & (centroid_radius <= radii[2])] = 2

    print(f"  Domain assignment: {np.bincount(domain_numbers)}")

    # Return 0-based triangles and domain numbers
    # Note: vertices are 0-based, domain numbers are in 4th row
    mesh_tri = np.vstack([triangles.T, domain_numbers])

    gmsh.finalize()

    return nodes, mesh_tri, domain_numbers


def add_cubic_nodes(nodes: np.ndarray, mesh_tri: np.ndarray):
    """
    Convert linear triangles to 10-node cubic elements
    Expects 0-based vertex indices in mesh_tri
    Returns SAFE-format with 1-based indices
    """
    print(f"Converting to 10-node cubic elements...")
    print(f"  Input: nodes={nodes.shape}, elements={mesh_tri.shape[1]}")

    n_tri = mesh_tri.shape[1]
    n_original = nodes.shape[0]

    midpoint_cache = {}
    new_nodes = []
    cubic_tri = np.zeros((10, n_tri), dtype=int)

    for i in range(n_tri):
        # Get 0-based vertex indices - ПРЕОБРАЗУЕМ В INT ЗДЕСЬ
        n1 = int(mesh_tri[0, i])
        n2 = int(mesh_tri[1, i])
        n3 = int(mesh_tri[2, i])

        # Validate indices
        if not (0 <= n1 < n_original and 0 <= n2 < n_original and 0 <= n3 < n_original):
            raise ValueError(f"Invalid node indices: {n1}, {n2}, {n3} (max: {n_original - 1})")

        # Store vertex nodes (1-based for SAFE)
        cubic_tri[0, i] = n1 + 1
        cubic_tri[1, i] = n2 + 1
        cubic_tri[2, i] = n3 + 1

        # Edge midpoints - a и b уже int
        for edge_idx, (a, b) in enumerate([(n1, n2), (n2, n3), (n3, n1)], start=3):
            key = tuple(sorted((a, b)))
            if key not in midpoint_cache:
                midpoint_cache[key] = n_original + len(new_nodes)
                new_nodes.append(0.5 * (nodes[a] + nodes[b]))

            cubic_tri[edge_idx, i] = midpoint_cache[key] + 1

        # Compute centroid
        p1 = nodes[n1]
        p2 = nodes[n2]
        p3 = nodes[n3]
        centroid = (p1 + p2 + p3) / 3.0

        # Interior nodes
        for j, vertex in enumerate([p1, p2, p3], start=6):
            interior = (2 / 3) * vertex + (1 / 3) * centroid
            new_nodes.append(interior)
            cubic_tri[j, i] = n_original + len(new_nodes)

        # Centroid node (node 10)
        new_nodes.append(centroid)
        cubic_tri[9, i] = n_original + len(new_nodes)

    # Append new nodes - преобразуем список в numpy array
    if new_nodes:
        new_nodes_array = np.vstack(new_nodes)
        nodes = np.vstack([nodes, new_nodes_array])

    props = {
        'n_nodes_per_element': 10,
        'n_elements': n_tri,
        'n_new_nodes_added': len(new_nodes)
    }

    print(f"  Added {len(new_nodes)} new nodes ({n_original} → {nodes.shape[0]})")

    return nodes, cubic_tri, props


def visualize_mesh(nodes: np.ndarray, mesh_tri: np.ndarray, title: str, filename: str):
    """
    Visualize mesh with domain coloring
    nodes: (N, 2) array
    mesh_tri: (4, n_elem) array - 0-based vertex indices + domain numbers
    """
    print(f"  Visualizing: {filename}")

    fig, ax = plt.subplots(figsize=(10, 10))

    # Get triangles (0-based)
    tri = mesh_tri[:3, :].T.astype(int)
    domain_colors = mesh_tri[3, :]

    unique_domains = np.unique(domain_colors)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_domains)))

    for i, domain_id in enumerate(unique_domains):
        mask = domain_colors == domain_id
        tri_domain = tri[mask]

        # Plot edges
        ax.triplot(nodes[:, 0], nodes[:, 1], tri_domain,
                   color=colors[i], lw=0.5, alpha=0.7)

        # Plot nodes
        nodes_in_domain = np.unique(tri_domain.flatten())
        ax.plot(nodes[nodes_in_domain, 0], nodes[nodes_in_domain, 1],
                'o', color=colors[i], markersize=4, label=f'Domain {domain_id}')

    ax.set_title(f"{title}\n{nodes.shape[0]} nodes, {mesh_tri.shape[1]} elements")
    ax.axis('equal')
    ax.legend()

    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    print("=" * 70)
    print("Demo: Gmsh Nested Domains with Cubic Elements")
    print("=" * 70)

    # 1. Create geometry and mesh
    nodes, mesh_tri, domains = create_unified_mesh([1.0, 2.0, 3.0])

    # 2. Visualize linear mesh
    visualize_mesh(nodes, mesh_tri,
                   "Linear Triangular Mesh",
                   "demo_linear_mesh.png")

    # 3. Convert to cubic elements
    nodes_cubic, mesh_cubic, props = add_cubic_nodes(nodes, mesh_tri)

    # 4. Visualize cubic mesh
    visualize_mesh(nodes_cubic, mesh_cubic,
                   "Cubic (10-node) Triangular Mesh",
                   "demo_cubic_mesh.png")

    print("=" * 70)
    print("✓ Demo completed successfully!")
    print("  Files created:")
    print("    - demo_linear_mesh.png")
    print("    - demo_cubic_mesh.png")
    print("  Open in Gmsh: gmsh geometry_debug.brep (not generated in this version)")
    print("=" * 70)


if __name__ == "__main__":
    main()