import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Define vertices of the cube
vertices = np.array([
    [-0.5, -0.5, -0.5],
    [-0.5, -0.5, 0.5],
    [-0.5, 0.5, -0.5],
    [-0.5, 0.5, 0.5],
    [0.5, -0.5, -0.5],
    [0.5, -0.5, 0.5],
    [0.5, 0.5, -0.5],
    [0.5, 0.5, 0.5]
])

# Define indices of the cube
indices = np.array([
    [0, 1, 3],
    [0, 3, 2],
    [0, 2, 4],
    [2, 6, 4],
    [0, 4, 1],
    [1, 4, 5],
    [2, 3, 6],
    [3, 7, 6],
    [4, 6, 5],
    [5, 6, 7],
    [1, 5, 7],
    [1, 7, 3]
])

# Visualize the cube mesh
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for triangle in indices:
    ax.plot(vertices[triangle, 0], vertices[triangle, 1], vertices[triangle, 2], 'b-')
plt.show()