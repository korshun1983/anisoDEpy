import meshpy.triangle as triangle
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Define rectangular domain (unit square) ---
width, height = 1.0, 1.0
points = [
    (0.0, 0.0),
    (width, 0.0),
    (width, height),
    (0.0, height)
]
facets = [(0, 1), (1, 2), (2, 3), (3, 0)]

# --- 2. Build info structure ---
info = triangle.MeshInfo()
info.set_points(points)
info.set_facets(facets)

# --- 3. Triangulate with ~50 elements ---
area_constraint = (width * height) / 50.0
mesh = triangle.build(info, max_area=area_constraint)

print(f"Generated {len(mesh.elements)} triangles")

# --- 4. Compute centroids ---
vertices = np.array(mesh.points)
faces = np.array(mesh.elements)
centroids = np.mean(vertices[faces], axis=1)

# --- 5. Plot mesh + centroids ---
plt.triplot(vertices[:, 0], vertices[:, 1], faces, color="k", lw=0.7)
plt.scatter(centroids[:, 0], centroids[:, 1], color="red", s=10, label="Centroids")
plt.legend()
plt.gca().set_aspect("equal")
plt.show()
