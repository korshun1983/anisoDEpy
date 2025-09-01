import pymesh
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Define rectangular domain (unit square) ---
width, height = 1.0, 1.0
vertices = np.array([
    [0.0, 0.0],
    [width, 0.0],
    [width, height],
    [0.0, height]
])
faces = np.array([[0, 1, 2], [0, 2, 3]])

# Create polygonal domain mesh
domain = pymesh.form_mesh(vertices, faces)

# --- 2. Triangulate with ~50 elements ---
area_constraint = (width * height) / 50.0
tri_mesh = pymesh.triangulate(
    domain,
    cell_size=area_constraint,
    engine="triangle"
)

print(f"Generated {len(tri_mesh.faces)} triangles")

# --- 3. Compute centroids of triangles ---
centroids = np.mean(tri_mesh.vertices[tri_mesh.faces], axis=1)

# --- 4. Plot mesh + centroids ---
plt.triplot(tri_mesh.vertices[:, 0], tri_mesh.vertices[:, 1], tri_mesh.faces, color="k", lw=0.7)
plt.scatter(centroids[:, 0], centroids[:, 1], color="red", s=10, label="Centroids")
plt.legend()
plt.gca().set_aspect("equal")
plt.show()
