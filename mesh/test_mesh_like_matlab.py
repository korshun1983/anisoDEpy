#import structures
import math
import numpy
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import gmsh

from structures import BigCompStruct

CompStruct = BigCompStruct()

CompStruct.Model.DomainRx = [0.1000, 8.7796, 13.1195]
CompStruct.Model.DomainRy = [0.1000, 8.7796, 13.1195]
CompStruct.Data.N_domain = 3
CompStruct.Model.DomainTheta = [0, 0, 0]
CompStruct.Model.AddDomainLoc = 'ext'
CompStruct.Model.DomainEcc = [0, 0, 0]
CompStruct.Model.DomainEccAngle = [0, 0, 0]

# ===============================================================================
# Preparing input data for the mesh construction -
# domain boundaries' nodes, edges, faces
# ===============================================================================
Edges = numpy.empty((0,2))
Nodes = numpy.empty((0,2))
ed_num = []
PrevEdgeSize = 0

DomainFaces = []

radii = numpy.empty((3,1))

# construct the lists of nodes, edges, and faces for the domain boundaries
for ii_d in range(CompStruct.Data.N_domain):

    # define the domain geometry
    Rx = CompStruct.Model.DomainRx[ii_d]
    Ry = CompStruct.Model.DomainRy[ii_d]

    radii[ii_d] = Rx #for circular grid

    # find the coordinates of the domain center
    ThetaRot = CompStruct.Model.DomainTheta[ii_d]
    Xc = CompStruct.Model.DomainEcc[ii_d] * math.cos(CompStruct.Model.DomainEccAngle[ii_d])
    Yc = CompStruct.Model.DomainEcc[ii_d] * math.sin(CompStruct.Model.DomainEccAngle[ii_d])

    # define the angular grid for the domain boundary
    DTheta = math.pi / CompStruct.Model.DomainNth[ii_d]
    Theta = numpy.arange(- math.pi, math.pi, DTheta)

    boundary_rec = 'n'
    if CompStruct.Mesh.ext_boundary_shape == 'rec':
        if CompStruct.Model.AddDomainLoc == 'ext' & ii_d >= CompStruct.Data.N_domain - 1:
            boundary_rec = 'y'
        if CompStruct.Model.AddDomainLoc == 'int' & ii_d == CompStruct.Data.N_domain:
            boundary_rec = 'y'

    # define the grid of nodes on the domain boundary (the domain is rotated
    # according to the input model)
    if boundary_rec == 'n':
        XBgrid = Xc + Rx * numpy.cos(Theta) * numpy.cos(ThetaRot) - Ry * numpy.sin(Theta) * numpy.sin(ThetaRot)
        YBgrid = Yc + Rx * numpy.cos(Theta) * numpy.sin(ThetaRot) + Ry * numpy.sin(Theta) * numpy.cos(ThetaRot)

    if boundary_rec == 'y':
        XBgrid = Rx * math.cos(Theta)
        YBgrid = Ry * math.sin(Theta)
        for ia in range(1, 1, Theta):
            ang = Theta[ia]
            if -math.pi <= ang & ang <= -(3. / 4.) * math.pi:
                XBgrid[ia] = -Rx
                YBgrid[ia] = -Rx * math.tan(math.pi + ang)

            if (3. / 4) * math.pi <= ang & ang <= math.pi:
                XBgrid[ia] = -Rx
                YBgrid[ia] = Rx * math.tan(math.pi - ang)

            if -(3. / 4) * math.pi <= ang & ang <= -(1. / 4.) * math.pi:
                XBgrid[ia] = Ry * math.tan(math.pi / 2 + ang)
                YBgrid[ia] = -Ry

            if -(1. / 4) * math.pi <= ang & ang <= (1. / 4.) * math.pi:
                XBgrid[ia] = Rx
                YBgrid[ia] = Rx * math.tan(ang)

            if (1. / 4) * math.pi <= ang & ang <= (3. / 4.) * math.pi:
                XBgrid[ia] = Ry * math.tan(math.pi / 2 - ang)
                YBgrid[ia] = Ry

    DomainBNodes = numpy.empty((len(XBgrid), 2))
    DomainBNodes[:,0] = XBgrid
    DomainBNodes[:,1] = YBgrid

    # define the edges for the domain boundary
    DomainBEdges = numpy.empty((numpy.size(DomainBNodes,axis=1), 2))
    DomainBEdges[:, 0] = numpy.arange(1, numpy.size(DomainBNodes,axis=1) + 1, 1)
    DomainBEdges[:, 1] = numpy.append(numpy.arange(2, numpy.size(DomainBNodes,axis=1) + 1, 1), 1)

    # ed_num=[ed_num; ones(size(node,1),1)*i];

    # define the start and the end edge for the domain ii_d
    FaceStartEdge = numpy.size(Edges, axis=0) - PrevEdgeSize + 1

    FaceEndEdge = numpy.size(numpy.append(Edges, (DomainBEdges + numpy.size(Edges, axis=0)), axis=0), axis=0)

    # define the edges that belong to the face of the domain ii_d
    DomainFaces.append(range(FaceStartEdge, FaceEndEdge + 1, 1))
    PrevEdgeSize = numpy.size(DomainBEdges, axis=0)

    # update the full list of edges and nodes of the domain boundaries
    Edges = numpy.append(Edges, (DomainBEdges + numpy.size(Edges, axis=0)), axis=0)
    Nodes = numpy.append(Nodes, DomainBNodes, axis=0)


# -------------------------------
# Initialize GMSH
gmsh.initialize()

## Alternate (easy) method for concentric circle geometry domain adding
# # Use OpenCASCADE kernel for better geometry handling
# gmsh.option.setNumber("Geometry.OCCBoundsUseStl", 1)
#
# # Create concentric disks
# center_x = Xc
# center_y = Yc
#
# # Create disks
# # disk1 = gmsh.model.occ.addDisk(center_x, center_y, 0, radii[0], radii[0])
# disk2 = gmsh.model.occ.addDisk(center_x, center_y, 0, radii[1], radii[1])
# disk3 = gmsh.model.occ.addDisk(center_x, center_y, 0, radii[2], radii[2])
#
# # Create annular regions by cutting inner disks from outer ones
# # annulus2 = gmsh.model.occ.cut([(2, disk2)], [(2, disk1)])
# annulus3 = gmsh.model.occ.cut([(2, disk3)], [(2, disk2)])
#
# disk1 = gmsh.model.occ.addDisk(center_x, center_y, 0, radii[0], radii[0])
# disk2 = gmsh.model.occ.addDisk(center_x, center_y, 0, radii[1], radii[1])
# annulus2 = gmsh.model.occ.cut([(2, disk2)], [(2, disk1)])
#
# disk1 = gmsh.model.occ.addDisk(center_x, center_y, 0, radii[0], radii[0])

## Method of direct points adding
boundary_pts = numpy.array(Nodes[48:,:])

interior_pts = numpy.array(Nodes[:47,:])

# Set mesh sizes
mesh_size_min = radii[0]*.2
mesh_size_max = radii[2]*.2
gmsh.model.add("custom_polygon_mesh")

# -------------------------------------------------
# Step 1: Create boundary polygon
# -------------------------------------------------
boundary_tags = []
for (x, y) in boundary_pts:
    tag = gmsh.model.geo.addPoint(x, y, 0, mesh_size_max)
    boundary_tags.append(tag)

# Connect points with lines in order
line_tags = []
for i in range(len(boundary_tags)):
    ltag = gmsh.model.geo.addLine(boundary_tags[i],
                                  boundary_tags[(i + 1) % len(boundary_tags)])
    line_tags.append(ltag)

# Form a closed loop and a surface
loop = gmsh.model.geo.addCurveLoop(line_tags)
surface = gmsh.model.geo.addPlaneSurface([loop])

# -------------------------------------------------
# Step 2: Add interior points and embed them
# -------------------------------------------------
interior_tags = []
for (x, y) in interior_pts:
    tag = gmsh.model.geo.addPoint(x, y, 0, mesh_size_min)
    interior_tags.append(tag)

gmsh.model.geo.synchronize()

# Embed these points in the surface so they become mesh nodes
# gmsh.model.mesh.embed(0, interior_tags, 2, [surface])

# -------------------------------------------------
# Step 3: Set meshing options
# -------------------------------------------------
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size_min)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size_max)
gmsh.option.setNumber("Mesh.Algorithm", 6)  # Delaunay

# -------------------------------------------------
# Step 4: Generate the 2D triangular mesh
# -------------------------------------------------
gmsh.model.mesh.generate(2)

# Get all mesh nodes
node_tags, node_coords, _ = gmsh.model.mesh.getNodes()

# Reshape coordinates to (n_nodes, 3)
nodes = node_coords.reshape(-1, 3)

# Get only x, y coordinates (ignore z for 2D)
x = nodes[:, 0]
y = nodes[:, 1]

# Get all 2D elements (triangles)
element_types, element_tags, node_tags_elements = gmsh.model.mesh.getElements(2)

# Find triangles (element type 2 in GMSH)
triangles = []
for i, elem_type in enumerate(element_types):
    if elem_type == 2:  # 3-node triangle
        # Node tags for triangles
        tri_node_tags = node_tags_elements[i]
        # Reshape to (n_triangles, 3)
        triangles = tri_node_tags.reshape(-1, 3) - 1  # Convert to 0-based indexing

# Create matplotlib triangulation
triangulation = tri.Triangulation(x, y, triangles)

# Create the plot
plt.figure(figsize=(12, 10))

# Plot the mesh
plt.triplot(triangulation, 'k-', linewidth=0.5, alpha=0.6)
plt.plot(x, y, 'o', markersize=1, alpha=0.5)

# Customize the plot
plt.title('2D Mesh for Concentric Cylindrical Layers', fontsize=14, fontweight='bold')
plt.xlabel('X coordinate', fontsize=12)
plt.ylabel('Y coordinate', fontsize=12)
plt.grid(True, alpha=0.3)
plt.axis('equal')

plt.show()

# Upgrade element order to cubic (order=3)
gmsh.option.setNumber("Mesh.ElementOrder", 3)
gmsh.model.mesh.setOrder(3)

# Get all mesh nodes after upgrading
node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
node_coords = node_coords.reshape(-1, 3)

# Get triangular elements (type=2 for linear, 9 for quadratic, 21 for cubic triangles)
etype = 21  # 3rd-order triangle
elemTags, elemNodeTags = gmsh.model.mesh.getElementsByType(etype)

# Original (corner) nodes: take the first 3 from each element
orig_nodes = []
new_nodes = []
for i in range(0, len(elemNodeTags), 10):  # 10 nodes per cubic triangle
    # First 3 are corners
    for n in elemNodeTags[i:i+3]:
        orig_nodes.append(n)
    # Last 7 are new (edge + centroid)
    for n in elemNodeTags[i+3:i+10]:
        new_nodes.append(n)

# Remove duplicates
orig_nodes = list(set(orig_nodes))
new_nodes = list(set(new_nodes))

# Map node tags to coordinates
coords_dict = {node_tags[i]: node_coords[i] for i in range(len(node_tags))}

orig_coords = [coords_dict[n][:2] for n in orig_nodes]
new_coords = [coords_dict[n][:2] for n in new_nodes]

# Plotting
fig, ax = plt.subplots()

# --- Draw triangle edges using corner nodes only ---
for i in range(0, len(elemNodeTags), 10):
    n1, n2, n3 = elemNodeTags[i:i+3]  # corner nodes
    x1, y1 = coords_dict[n1][:2]
    x2, y2 = coords_dict[n2][:2]
    x3, y3 = coords_dict[n3][:2]
    ax.plot([x1, x2], [y1, y2], 'k-', lw=0.5)
    ax.plot([x2, x3], [y2, y3], 'k-', lw=0.5)
    ax.plot([x3, x1], [y3, y1], 'k-', lw=0.5)

# --- Plot nodes ---
if orig_coords:
    ox, oy = zip(*orig_coords)
    ax.scatter(ox, oy, color="blue", label="Original nodes (order 2)")

if new_coords:
    nx, ny = zip(*new_coords)
    ax.scatter(nx, ny, color="red", s=15, marker="o", label="New cubic nodes")

ax.set_aspect("equal")
ax.legend()
plt.show()

gmsh.finalize()