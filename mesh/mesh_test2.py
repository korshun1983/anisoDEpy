import open3d as o3d
import numpy as np

# Create a point cloud (example: a simple cube's corners)
points = np.array([
    [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
    [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]
])
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Reconstruct a mesh from the point cloud using alpha shapes
# Alpha determines the "tightness" of the reconstruction
alpha = 0.5
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)

# You can also create a basic mesh directly by defining vertices and triangles
# mesh = o3d.geometry.TriangleMesh()
# mesh.vertices = o3d.utility.Vector3dVector(vertices)
# mesh.triangles = o3d.utility.Vector3iVector(triangles)

o3d.visualization.draw_geometries([mesh])