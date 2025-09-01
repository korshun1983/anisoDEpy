# Example for 2D triangular mesh generation using MeshPy (requires Triangle)
# For 3D, you would use TetGen through MeshPy
from meshpy.triangle import MeshInfo, build

mesh_info = MeshInfo()
mesh_info.set_points([(0, 0), (1, 0), (1, 1), (0, 1)])
mesh_info.set_facets([(0, 1), (1, 2), (2, 3), (3, 0)])

mesh = build(mesh_info)

# Access vertices and elements (triangles)
# print(mesh.points)
# print(mesh.elements)