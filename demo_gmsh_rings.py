#!/usr/bin/env python3
"""
Demonstration of gmsh capabilities for multi-domain mesh generation.
Features:
1. Three concentric domains created with boolean cut
2. Mesh size parameter controls element density
3. Two generation methods:
   - Linear mesh (3 nodes/element)
   - Direct high-order mesh generation (10 nodes/element)
4. Visualization shows added nodes
"""

import gmsh
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib.collections import PatchCollection

# =============================================================================
# USER CONFIGURATION
# =============================================================================
MESH_SIZE = 0.2  # Controls element size - smaller value = finer mesh

# Visualization parameters
DOMAIN_COLORS = {
    1: '#ff6b6b',  # Red for inner disk
    2: '#4ecdc4',  # Teal for middle ring
    3: '#45b7d1'  # Blue for outer ring
}


def create_geometry_with_cut():
    """Create three concentric circular domains using boolean cut"""
    print(f"Creating geometry with radii: 1.0, 2.0, 3.0")
    print(f"Mesh size parameter: {MESH_SIZE}")

    gmsh.initialize()
    gmsh.model.add("concentric_domains_cut")

    # Create three circles
    circle1 = gmsh.model.occ.addCircle(0, 0, 0, 1.0)
    curve_loop1 = gmsh.model.occ.addCurveLoop([circle1])
    surface1 = gmsh.model.occ.addPlaneSurface([curve_loop1])

    circle2 = gmsh.model.occ.addCircle(0, 0, 0, 2.0)
    curve_loop2 = gmsh.model.occ.addCurveLoop([circle2])
    surface2 = gmsh.model.occ.addPlaneSurface([curve_loop2])

    circle3 = gmsh.model.occ.addCircle(0, 0, 0, 3.0)
    curve_loop3 = gmsh.model.occ.addCurveLoop([circle3])
    surface3 = gmsh.model.occ.addPlaneSurface([curve_loop3])

    gmsh.model.occ.synchronize()

    # Use boolean cut to create rings
    outer_ring, _ = gmsh.model.occ.cut([(2, surface3)], [(2, surface2)],
                                       removeObject=True, removeTool=False)
    middle_ring, _ = gmsh.model.occ.cut([(2, surface2)], [(2, surface1)],
                                        removeObject=True, removeTool=False)

    gmsh.model.occ.synchronize()

    # Get domain IDs
    domain1_id = surface1
    domain2_id = middle_ring[0][1]
    domain3_id = outer_ring[0][1]

    print(f"Domain IDs: {domain1_id}, {domain2_id}, {domain3_id}")

    # Assign physical groups
    pg_disk = gmsh.model.addPhysicalGroup(2, [domain1_id])
    gmsh.model.setPhysicalName(2, pg_disk, "Domain_1")

    pg_ring1 = gmsh.model.addPhysicalGroup(2, [domain2_id])
    gmsh.model.setPhysicalName(2, pg_ring1, "Domain_2")

    pg_ring2 = gmsh.model.addPhysicalGroup(2, [domain3_id])
    gmsh.model.setPhysicalName(2, pg_ring2, "Domain_3")

    gmsh.model.occ.synchronize()

    return [domain1_id, domain2_id, domain3_id], [pg_disk, pg_ring1, pg_ring2]


def generate_linear_mesh(domain_ids):
    """Generate linear triangular mesh using global MESH_SIZE"""
    print(f"\nGenerating linear triangular mesh with size {MESH_SIZE}...")

    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), MESH_SIZE)
    gmsh.model.mesh.setAlgorithm(2, domain_ids[0], 5)  # Frontal-Delaunay

    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(1)

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2)

    print(f"Generated {len(node_tags)} nodes")
    print(f"Generated {len(elem_tags[0])} triangular elements")

    return node_tags, node_coords, elem_types, elem_tags, elem_node_tags


def generate_high_order_mesh(domain_ids, order=3):
    """Generate high-order mesh DIRECTLY without intermediate linear step"""
    print(f"\nGenerating {order}rd-order triangular mesh directly...")

    # Set mesh size and algorithm
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), MESH_SIZE)
    gmsh.model.mesh.setAlgorithm(2, domain_ids[0], 5)

    # Generate 2D mesh (initially linear)
    gmsh.model.mesh.generate(2)

    # Convert to high order directly
    gmsh.model.mesh.setOrder(order)

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2)

    nodes_per_elem = 10 if order == 3 else 3  # 10 nodes for 3rd order

    print(f"Generated {len(node_tags)} nodes")
    print(f"Generated {len(elem_tags[0])} triangular elements")
    print(f"Each element has {nodes_per_elem} nodes")

    return node_tags, node_coords, elem_types, elem_tags, elem_node_tags


def get_elements_grouped_by_domain(physical_group_tags):
    """Group elements and nodes by physical domain"""
    elements_by_domain = {1: [], 2: [], 3: []}
    nodes_by_domain = {1: set(), 2: set(), 3: set()}

    # Get all elements
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2)

    if len(elem_types) == 0 or len(elem_tags) == 0:
        print("No elements found!")
        return elements_by_domain, nodes_by_domain, {'1-2': set(), '2-3': set()}

    # Build element-to-entity map
    element_to_entity = {}
    for domain_num, pg_tag in enumerate(physical_group_tags, 1):
        try:
            entities = gmsh.model.getEntitiesForPhysicalGroup(2, pg_tag)
            if entities.size > 0:
                for entity_id in entities:
                    elem_types_ent, elem_tags_ent, elem_nodes_ent = gmsh.model.mesh.getElements(2, entity_id)
                    if len(elem_types_ent) > 0 and len(elem_tags_ent) > 0:
                        elem_type = elem_types_ent[0]
                        nodes_per_elem = 3 if elem_type == 2 else 10

                        for elem_tag in elem_tags_ent[0]:
                            element_to_entity[elem_tag] = (domain_num, nodes_per_elem)
        except Exception as e:
            print(f"Warning: Could not get elements for physical group {pg_tag}: {e}")

    # Process elements
    tri_elements = elem_tags[0]
    tri_nodes = elem_node_tags[0]

    for i, elem_tag in enumerate(tri_elements):
        if elem_tag in element_to_entity:
            domain, nodes_per_elem = element_to_entity[elem_tag]
            start_idx = i * nodes_per_elem
            end_idx = start_idx + nodes_per_elem
            node_list = tri_nodes[start_idx:end_idx]

            elements_by_domain[domain].append((elem_tag, list(node_list)))
            nodes_by_domain[domain].update(node_list)

    # Find boundary nodes
    boundary_nodes = {
        '1-2': nodes_by_domain[1] & nodes_by_domain[2],
        '2-3': nodes_by_domain[2] & nodes_by_domain[3]
    }

    total_elems = sum(len(v) for v in elements_by_domain.values())
    print(f"Total elements assigned: {total_elems}")

    return elements_by_domain, nodes_by_domain, boundary_nodes


def visualize_linear_mesh(node_coords, elements_by_domain,
                          nodes_by_domain, boundary_nodes):
    """Visualize linear triangular mesh"""
    fig, ax = plt.subplots(figsize=(10, 10))

    coords = node_coords.reshape(-1, 3)[:, :2]

    # Plot elements
    for domain_id in [1, 2, 3]:
        patches = []
        for elem_tag, node_list in elements_by_domain[domain_id]:
            vertices = coords[np.array(node_list[:3], dtype=int) - 1]
            polygon = Polygon(vertices, closed=True)
            patches.append(polygon)

        collection = PatchCollection(
            patches,
            facecolor=DOMAIN_COLORS[domain_id],
            edgecolor='black',
            alpha=0.6,
            linewidth=0.5
        )
        ax.add_collection(collection)

    # Plot nodes (small dots)
    for domain_id in [1, 2, 3]:
        if nodes_by_domain[domain_id]:
            domain_node_coords = coords[np.array(list(nodes_by_domain[domain_id]), dtype=int) - 1]
            ax.scatter(domain_node_coords[:, 0], domain_node_coords[:, 1],
                       c=DOMAIN_COLORS[domain_id], s=15, alpha=0.6,
                       marker='.', zorder=5)

    # Plot boundary nodes
    for boundary_name, bnodes in boundary_nodes.items():
        if bnodes:
            bcoords = coords[np.array(list(bnodes), dtype=int) - 1]
            ax.scatter(bcoords[:, 0], bcoords[:, 1], c='black', s=50,
                       marker='o', linewidth=2, zorder=10)

    # Show domain boundaries
    for radius in [1.0, 2.0, 3.0]:
        circle = Circle((0, 0), radius, fill=False, linestyle='--',
                        edgecolor='gray', alpha=0.5)
        ax.add_patch(circle)

    ax.set_aspect('equal')
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-3.2, 3.2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Linear Triangular Mesh\n(3 nodes/element, mesh size={MESH_SIZE})')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def visualize_high_order_mesh(node_coords, elements_by_domain,
                              nodes_by_domain, boundary_nodes):
    """Visualize high-order mesh with added nodes"""
    fig, ax = plt.subplots(figsize=(10, 10))

    coords = node_coords.reshape(-1, 3)[:, :2]

    # Plot elements as straight-edged triangles (using only vertices)
    for domain_id in [1, 2, 3]:
        patches = []
        for elem_tag, node_list in elements_by_domain[domain_id]:
            # Use only first 3 nodes (vertices) for straight edges
            vertices = coords[np.array(node_list[:3], dtype=int) - 1]
            polygon = Polygon(vertices, closed=True)
            patches.append(polygon)

        collection = PatchCollection(
            patches,
            facecolor=DOMAIN_COLORS[domain_id],
            edgecolor='black',
            alpha=0.4,
            linewidth=0.5
        )
        ax.add_collection(collection)

    # Plot ALL nodes (vertices + edge + internal nodes)
    node_sizes = {1: 15, 2: 12, 3: 10}
    for domain_id in [1, 2, 3]:
        if nodes_by_domain[domain_id]:
            domain_node_coords = coords[np.array(list(nodes_by_domain[domain_id]), dtype=int) - 1]
            ax.scatter(domain_node_coords[:, 0], domain_node_coords[:, 1],
                       c=DOMAIN_COLORS[domain_id], s=node_sizes[domain_id],
                       alpha=0.8, marker='o', edgecolor='black', linewidth=0.5, zorder=5)

    # Emphasize boundary nodes
    for boundary_name, bnodes in boundary_nodes.items():
        if bnodes:
            bcoords = coords[np.array(list(bnodes), dtype=int) - 1]
            ax.scatter(bcoords[:, 0], bcoords[:, 1], c='black', s=50,
                       marker='o', linewidth=3, zorder=10)

    ax.text(0, -3.5, f'3rd-order mesh with {len(node_coords) // 3} nodes\n'
                     f'(mesh size={MESH_SIZE})',
            ha='center', fontsize=10)

    for radius in [1.0, 2.0, 3.0]:
        circle = Circle((0, 0), radius, fill=False, linestyle='--',
                        edgecolor='gray', alpha=0.5)
        ax.add_patch(circle)

    ax.set_aspect('equal')
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-3.2, 3.2)
    ax.set_title(f'3rd-Order Triangular Mesh\n(10 nodes/element, straight edges)')

    plt.tight_layout()
    return fig


def main():
    """Main execution"""
    print("=" * 60)
    print("GMSH Multi-Domain Mesh Generation Demo")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Mesh size: {MESH_SIZE}")

    # Create geometry
    domain_ids, physical_group_tags = create_geometry_with_cut()

    # Part 1: Linear mesh
    print("\n" + "=" * 50)
    print("PART 1: LINEAR TRIANGULAR MESH")
    print("=" * 50)

    node_tags_lin, node_coords_lin, elem_types_lin, elem_tags_lin, elem_node_tags_lin = \
        generate_linear_mesh(domain_ids)

    elements_by_domain_lin, nodes_by_domain_lin, boundary_nodes_lin = \
        get_elements_grouped_by_domain(physical_group_tags)

    fig1 = visualize_linear_mesh(
        node_coords_lin, elements_by_domain_lin,
        nodes_by_domain_lin, boundary_nodes_lin
    )
    plt.savefig('mesh_linear.png', dpi=150, bbox_inches='tight')
    print("Saved: mesh_linear.png")

    # Part 2: High-order mesh (direct generation)
    print("\n" + "=" * 50)
    print("PART 2: 3RD-ORDER MESH (DIRECT GENERATION)")
    print("=" * 50)

    # Clear and regenerate mesh in high order directly
    gmsh.model.mesh.clear()
    node_tags_ho, node_coords_ho, elem_types_ho, elem_tags_ho, elem_node_tags_ho = \
        generate_high_order_mesh(domain_ids, order=3)

    elements_by_domain_ho, nodes_by_domain_ho, boundary_nodes_ho = \
        get_elements_grouped_by_domain(physical_group_tags)

    fig2 = visualize_high_order_mesh(
        node_coords_ho, elements_by_domain_ho,
        nodes_by_domain_ho, boundary_nodes_ho
    )
    plt.savefig('mesh_high_order.png', dpi=150, bbox_inches='tight')
    print("Saved: mesh_high_order.png")

    # Save mesh files
    gmsh.write("concentric_domains_linear.msh")
    print("Saved: concentric_domains_linear.msh")

    gmsh.write("concentric_domains_high_order.msh")
    print("Saved: concentric_domains_high_order.msh")

    # Show plots
    print("\n" + "=" * 50)
    print("Displaying all visualizations...")
    print("=" * 50)
    plt.show()

    # Finalize
    gmsh.finalize()
    print("\nGMSH session finalized.")
    print("\nOutput files:")
    print("  - concentric_domains_linear.msh")
    print("  - concentric_domains_high_order.msh")
    print("  - mesh_linear.png")
    print("  - mesh_high_order.png")


if __name__ == "__main__":
    main()