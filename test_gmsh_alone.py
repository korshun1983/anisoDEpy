#!/usr/bin/env python3
"""
test_gmsh_alone.py
==================
Standalone test for Gmsh mesh generation with visualization
"""

import gmsh
import numpy as np
import matplotlib.pyplot as plt


def test_gmsh():
    """Test Gmsh mesh generation and visualize"""
    print("=" * 60)
    print("Testing Gmsh mesh generation...")
    print("=" * 60)

    # Initialize Gmsh
    gmsh.initialize()
    gmsh.model.add("test")

    # Create geometry: outer circle (radius 1.0)
    outer_tag = gmsh.model.occ.addDisk(0, 0, 0, 1.0, 1.0)

    # Create inner circle (hole, radius 0.3) for more interesting mesh
    inner_tag = gmsh.model.occ.addDisk(0, 0, 0, 0.3, 0.3)

    # Subtract inner from outer
    result = gmsh.model.occ.cut([(2, outer_tag)], [(2, inner_tag)])
    gmsh.model.occ.synchronize()

    # Set mesh parameters
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.15)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.05)
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
    gmsh.option.setNumber("Mesh.ElementOrder", 2)  # Quadratic elements

    # Generate mesh
    print("Generating mesh...")
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.refine()  # Optional refinement

    # Extract mesh data
    node_tags, coord, _ = gmsh.model.mesh.getNodes()
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements()

    # Convert nodes to 2D numpy array
    n_nodes = len(coord) // 3
    nodes = coord.reshape(n_nodes, 3)[:, :2]  # Take only X,Y coordinates

    print(f"✓ Generated {n_nodes} nodes")

    # Find triangle elements (type 9 = Tri6, type 2 = Tri3)
    tri_idx = -1
    for idx, etype in enumerate(elem_types):
        if etype == 9:  # Tri6 (quadratic)
            tri_idx = idx
            debug_print(f"    Found Tri6 elements", level=2)
            break
        elif etype == 2:  # Tri3 (linear)
            tri_idx = idx
            debug_print(f"    Found Tri3 elements", level=2)

    if tri_idx == -1:
        raise RuntimeError("No triangle elements found in mesh!")

    # Extract triangles (convert to 0-based indexing for matplotlib)
    if elem_types[tri_idx] == 9:
        # For Tri6, use only the first 3 vertices
        triangles = elem_node_tags[tri_idx].reshape(-1, 6)[:, :3] - 1
    else:
        triangles = elem_node_tags[tri_idx].reshape(-1, 3) - 1

    n_tri = len(triangles)
    print(f"✓ Generated {n_tri} triangles")

    # Visualize mesh
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot mesh edges
    ax.triplot(nodes[:, 0], nodes[:, 1], triangles,
               'b-', lw=0.5, alpha=0.7, label='Mesh edges')

    # Plot nodes
    ax.plot(nodes[:, 0], nodes[:, 1], 'r.', markersize=3, label='Nodes')

    # Highlight boundary nodes (optional)
    # boundary_nodes = find_boundary_nodes(triangles)  # You could implement this
    # ax.plot(nodes[boundary_nodes, 0], nodes[boundary_nodes, 1], 'go', markersize=4)

    ax.set_title(f'Gmsh Test Mesh\n{n_nodes} nodes, {n_tri} elements', fontsize=14)
    ax.set_xlabel('X coordinate', fontsize=12)
    ax.set_ylabel('Y coordinate', fontsize=12)
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    # Save figure
    output_file = 'test_gmsh_mesh.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Mesh visualization saved to: {output_file}")

    # Show plot (comment out if running on server without display)
    # plt.show()

    # Finalize Gmsh
    gmsh.finalize()
    print("=" * 60)
    print("✓ Gmsh test completed successfully")
    print("=" * 60)


def debug_print(message: str, level: int = 1):
    """Simple debug print function"""
    print(f"[DEBUG] {message}")


if __name__ == "__main__":
    try:
        test_gmsh()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        gmsh.finalize()
        sys.exit(1)