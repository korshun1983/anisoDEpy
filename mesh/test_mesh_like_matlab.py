#import structures
import math
import numpy
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
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

# construct the lists of nodes, edges, and faces for the domain boundaries
for ii_d in range(CompStruct.Data.N_domain):

    # define the domain geometry
    Rx = CompStruct.Model.DomainRx[ii_d]
    Ry = CompStruct.Model.DomainRy[ii_d]

    if ii_d == 2:
        R = numpy.sqrt(Rx*Rx+Ry*Ry)

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

# ===============================================================================
# Constructing triangular mesh based on the supplied model geometry
# ===============================================================================
# CurDir = cd(CompStruct.Config.root_path)
# cd('routines\Mesh2d v24\')
# [MeshNodes, MeshTri, MeshFaceNums, ~, ~, ~, CompStruct] = CompStruct.Methods.MeshFaces(Nodes, Edges, DomainFaces, CompStruct)
# cd(CurDir)

# # 2. Perform Delaunay triangulation
# tri = Delaunay(Nodes)
#
# # 3. Add additional points within each triangle (e.g., centroids)
# new_points = []
# for simplex in tri.simplices:
#     triangle_points = Nodes[simplex]
#     centroid = numpy.mean(triangle_points, axis=0)
#     new_points.append(centroid)
#
# # Convert new_points to a NumPy array and combine with original points
# new_points = numpy.array(new_points)
# all_points = Nodes
# # all_points = np.vstack((points, new_points))
#
# # Re-triangulate with the expanded set of points
# tri_refined = Delaunay(all_points)
#
# # 4. Visualize the mesh
# # plt.figure(figsize=(8, 6))
# plt.triplot(all_points[:, 0], all_points[:, 1], tri_refined.simplices, color='blue', alpha=0.7)
# plt.plot(all_points[:, 0], all_points[:, 1], 'o', markersize=4, color='red')  # Plot all points
# plt.plot(new_points[:, 0], new_points[:, 1], 'o', markersize=4, color='red')  # Plot new points
# plt.title('2D Triangular test Mesh')
# plt.xlabel('X-coordinate')
# plt.ylabel('Y-coordinate')
# plt.grid(True)
# plt.gca().set_aspect('equal', adjustable='box')  # Ensure equal aspect ratio
# plt.show()

# -------------------------------
# TODO: adapt options for gmsh mesh construction

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