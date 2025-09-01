import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

# 1. Generate a set of initial points
np.random.seed(0) # for reproducibility
points = np.random.rand(20, 2) * 10 # 20 random points in a 10x10 square

# 2. Perform Delaunay triangulation
tri = Delaunay(points)

# 3. Add additional points within each triangle (e.g., centroids)
new_points = []
for simplex in tri.simplices:
    triangle_points = points[simplex]
    centroid = np.mean(triangle_points, axis=0)
    new_points.append(centroid)

# Convert new_points to a NumPy array and combine with original points
new_points = np.array(new_points)
all_points = points
#all_points = np.vstack((points, new_points))

# Re-triangulate with the expanded set of points
tri_refined = Delaunay(all_points)

# 4. Visualize the mesh
plt.figure(figsize=(8, 6))
plt.triplot(all_points[:, 0], all_points[:, 1], tri_refined.simplices, color='blue', alpha=0.7)
plt.plot(all_points[:, 0], all_points[:, 1], 'o', markersize=4, color='red') # Plot all points
plt.plot(new_points[:, 0], new_points[:, 1], 'o', markersize=4, color='red') # Plot new points
plt.title('2D Triangular Mesh with In-Triangle Points')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box') # Ensure equal aspect ratio
plt.show()