import gmsh
import numpy as np
import pyvista as pv

# Initialize Gmsh
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1) # Show output in terminal

# Create a new model
gmsh.model.add("cylinder_mesh")

# Define cylinder parameters
radius = 1.0
height = 2.0
lc = 0.2 # Characteristic length for meshing

# Create points for the cylinder base
p1 = gmsh.model.geo.addPoint(0, 0, 0, lc)
p2 = gmsh.model.geo.addPoint(radius, 0, 0, lc)
p3 = gmsh.model.geo.addPoint(0, radius, 0, lc)
p4 = gmsh.model.geo.addPoint(-radius, 0, 0, lc)
p5 = gmsh.model.geo.addPoint(0, -radius, 0, lc)

# Create arcs for the cylinder base
c1 = gmsh.model.geo.addCircleArc(p2, p1, p3)
c2 = gmsh.model.geo.addCircleArc(p3, p1, p4)
c3 = gmsh.model.geo.addCircleArc(p4, p1, p5)
c4 = gmsh.model.geo.addCircleArc(p5, p1, p2)

# Create curve loop and plane surface for the base
cl1 = gmsh.model.geo.addCurveLoop([c1, c2, c3, c4])
s1 = gmsh.model.geo.addPlaneSurface([cl1])

# Extrude the base to create the cylinder volume
vol = gmsh.model.geo.extrude([(2, s1)], 0, 0, height)

# Synchronize the CAD model with the Gmsh model
gmsh.model.geo.synchronize()

# Set meshing algorithm (optional, but recommended for better quality)
#gmsh.model.mesh.setAlgorithm(3, vol[1][1]) # 3 for Delaunay

# Generate 3D mesh
gmsh.model.mesh.generate(3)

# Save the mesh (optional)
gmsh.write("cylinder.msh")

# Finalize Gmsh
gmsh.finalize()



# Load the generated mesh file
mesh = pv.read("cylinder.msh")

# Extract the unstructured grid (tetrahedra)
tetra_mesh = mesh.extract_unstructured_grid()

# Visualize the mesh
tetra_mesh.plot(show_edges=True, color='lightblue', opacity=0.8)