import pymesh
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

# Generate point clouds
num_points = 10000
center = np.array([0, 0, 0])
radius = 1
theta = np.random.uniform(0, 2*np.pi, num_points)
phi = np.random.uniform(0, np.pi, num_points)
x = radius*np.sin(phi)*np.cos(theta) + center[0]
y = radius*np.sin(phi)*np.sin(theta) + center[1]
z = radius*np.cos(phi) + center[2]
point_cloud = np.column_stack((x, y, z))

# Visualize the point clouds
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c='r', marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

# Create a mesh from the point clouds using PyMesh
mesh = pymesh.PyMesh()
mesh.load_point_cloud_from_array(point_cloud)
mesh.reconstruct(eta=5, with_timing=True)

# Visualize the resulting mesh
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(mesh.vertices[:,0], mesh.vertices[:,1], mesh.vertices[:,2],
                triangles=mesh.faces, shade=True, color='w')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()