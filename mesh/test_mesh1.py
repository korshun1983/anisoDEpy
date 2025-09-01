import numpy as np
import matplotlib.pyplot as plt

# Define vertices (x, y, z coordinates)
x = np.array([0, 1, 0, 1])
y = np.array([0, 0, 1, 1])
z = np.array([0, 0, 0, 1])

# Define triangles (indices of vertices that form each triangle)
# For example, (0, 1, 2) forms a triangle with vertices at index 0, 1, and 2
triangles = np.array([[0, 1, 2], [1, 3, 2]])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(x, y, z, triangles=triangles, cmap='viridis')
plt.show()