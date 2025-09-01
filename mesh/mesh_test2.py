import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Define rectangular domain (unit square) ---
width, height = 1.0, 1.0
vertices = np.array([
    [0.0, 0.0, 0.0],
    [width, 0.0, 0.0],
    [width, height, 0.0],
    [0.0, height, 0.0]
])
faces = np.array([[0, 1, 2], [0, 2, 3]])

# Create initial mesh (just a quad split into 2 triangles)
mesh = o3d.geometry.TriangleMesh(
    o3d.utility.Vector3dVector(vertices),
    o3d.utility.Vector3iVector(faces)
)

# --- 2. Subdivide until ~50 triangles ---
# Each midpoint subdivision multiplies #triangles by 4
target_triangles = 50
while len(mesh.triangles) < target_triangles:
    mesh = mesh.subdivide_midpoint(1)

print(f"Generated {len(mesh.triangles)} triangles")

# --- 3. Compute centroids of triangles ---
tri_vertices = np.asarray(mesh.vertices)[np.asarray(mesh.triangles)]
centroids = tri_vertices.mean(axis=1)

# --- 4. Plot mesh + centroids (2D projection) ---
plt.triplot(
    np.asarray(mesh.vertices)[:, 0],
    np.asarray(mesh.vertices)[:, 1],
    np.asarray(mesh.triangles),
    color="k", lw=0.7
)
plt.scatter(centroids[:, 0], centroids[:, 1], color="red", s=10, label="Centroids")
plt.legend()
plt.gca().set_aspect("equal")
plt.show()
