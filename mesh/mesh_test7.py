import gmsh
import numpy as np
import matplotlib.pyplot as plt

gmsh.initialize()
gmsh.model.add("rectangle")

# --- 1. Define rectangular domain (unit square) ---
width, height = 1.0, 1.0
p1 = gmsh.model.geo.addPoint(0,     0, 0)
p2 = gmsh.model.geo.addPoint(width, 0, 0)
p3 = gmsh.model.geo.addPoint(width, height, 0)
p4 = gmsh.model.geo.addPoint(0,     height, 0)

l1 = gmsh.model.geo.addLine(p1, p2)
l2 = gmsh.model.geo.addLine(p2, p3)
l3 = gmsh.model.geo.addLine(p3, p4)
l4 = gmsh.model.geo.addLine(p4, p1)

cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
surface = gmsh.model.geo.addPlaneSurface([cl])

gmsh.model.geo.synchronize()

# --- 2. Mesh with ~50 elements ---
# characteristic length controls density
target_triangles = 50
domain_area = width * height
avg_area = domain_area / target_triangles
# edge length ~ sqrt(2 * area) for triangles
char_len = (2 * avg_area) ** 0.5

gmsh.option.setNumber("Mesh.CharacteristicLengthMin", char_len)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", char_len)

gmsh.model.mesh.generate(2)

# --- 3. Extract mesh data ---
nodes = gmsh.model.mesh.getNodes()
node_coords = nodes[1].reshape(-1, 3)
elements = gmsh.model.mesh.getElementsByType(2)[1].reshape(-1, 3) - 1  # 2 = triangle

print(f"Generated {len(elements)} triangles")

# --- 4. Compute centroids ---
centroids = node_coords[elements][:, :, :2].mean(axis=1)

# --- 5. Plot mesh + centroids ---
plt.triplot(node_coords[:,0], node_coords[:,1], elements, color="k", lw=0.7)
plt.scatter(centroids[:,0], centroids[:,1], color="red", s=10, label="Centroids")
plt.legend()
plt.gca().set_aspect("equal")
plt.show()

gmsh.finalize()
