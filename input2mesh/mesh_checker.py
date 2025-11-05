# mesh_checker.py
import gmsh
import json5


def check_mesh_quality(mesh_file):
    """Check quality of generated mesh"""
    gmsh.initialize()
    gmsh.open(mesh_file)

    # Get mesh statistics
    nodes = gmsh.model.mesh.getNodes()
    elements = gmsh.model.mesh.getElements(2)

    print("Mesh Quality Check:")
    print(f"Total nodes: {len(nodes[0])}")
    print(f"Total elements: {sum(len(elem[1]) for elem in elements)}")

    # Check element types
    for elem_type, elem_tags, elem_nodes in elements:
        print(f"Element type {elem_type}: {len(elem_tags)} elements")

    # Check for negative Jacobians (poor quality elements)
    quality = gmsh.model.mesh.getJacobians(2, "element")
    min_quality = min(quality) if quality else 0
    print(f"Minimum element quality: {min_quality:.6f}")

    gmsh.finalize()


if __name__ == "__main__":
    check_mesh_quality("waveguide_mesh.msh")