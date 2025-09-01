import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define cylinder parameters
radius = 1.0
height = 3.0
num_segments = 50  # Number of segments for the circular base
num_height_slices = 50 # Number of slices along the height

# Generate cylindrical coordinates
theta = np.linspace(0, 2 * np.pi, num_segments)
z = np.linspace(0, height, num_height_slices)
theta, z = np.meshgrid(theta, z)

# Convert cylindrical to Cartesian coordinates
x = radius * np.cos(theta)
y = radius * np.sin(theta)

# Create the 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the cylinder surface
ax.plot_surface(x, y, z, cmap='viridis')

# Plot the top and bottom caps
# Bottom cap
x_cap_bottom = radius * np.cos(np.linspace(0, 2 * np.pi, num_segments))
y_cap_bottom = radius * np.sin(np.linspace(0, 2 * np.pi, num_segments))
ax.plot_surface(x_cap_bottom, y_cap_bottom, 0 * x_cap_bottom, color='gray', alpha=0.5)

# Top cap
x_cap_top = radius * np.cos(np.linspace(0, 2 * np.pi, num_segments))
y_cap_top = radius * np.sin(np.linspace(0, 2 * np.pi, num_segments))
ax.plot_surface(x_cap_top, y_cap_top, height * x_cap_top / x_cap_top, color='gray', alpha=0.5)


ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Cylinder Mesh')

plt.show()