#import structures
import math
import numpy
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

from structures import BigCompStruct


def PrepareMeshBH(CompStruct: BigCompStruct):


    # Part of the toolbox for solving problems of wave propagation
    # in arbitrary anisotropic inhomogeneous waveguides.
    # For details see User manual
    # and comments to the main script gen_aniso.m
    #
    # PrepareMeshBH M-file
    #      PrepareMeshBH, by itself, prepares the mesh
    #      for elliptically layered boreholes.
    #      It is based on the open source Mesh2D package.
    #
    #   [T.Zharnikov, D.Syresin, SMR v0.3_08.2014]
    #
    # function [p,e,t] = PrepareMeshBH(CompStruct)
    #
    #  Inputs -
    #
    #       CompStuct - structure containing the parameters of the model;
    #
    #       ii_l - number of the layer, for which the physical properties are
    #              introduced
    #
    #  Outputs -
    #
    #       PhysProp - structure, containing physical properties of the ii_l-th
    #       layer
    #
    #  M-files required-
    #
    # Last Modified by Timur Zharnikov SMR v0.3_08.2014

    ################################################################################
    #
    #   Code for PrepareMeshBH, original version written by Denis Syresin, SMR, 2013
    #
    ################################################################################

    # FOR DEBUG ONLY #!!!!!!!!!!!!!!!!!!
    CompStruct.Model.DomainRx = [0.1000, 8.7796, 13.1195]
    CompStruct.Model.DomainRy = [0.1000, 8.7796, 13.1195]
    CompStruct.Data.N_domain = 3
    CompStruct.Model.DomainTheta = [0, 0, 0]
    CompStruct.Model.AddDomainLoc = 'ext'
    CompStruct.Model.DomainEcc = [0,0,0]
    CompStruct.Model.DomainEccAngle = [0,0,0]


    # ===============================================================================
    # Preparing input data for the mesh construction -
    # domain boundaries' nodes, edges, faces
    # ===============================================================================
    Edges = []
    Nodes = []
    ed_num = []
    PrevEdgeSize = 0

    DomainFaces = []

    # construct the lists of nodes, edges, and faces for the domain boundaries
    for ii_d in range(1,1,CompStruct.Data.N_domain):

        # define the domain geometry
        Rx = CompStruct.Model.DomainRx(ii_d)
        Ry = CompStruct.Model.DomainRy(ii_d)

        # find the coordinates of the domain center
        ThetaRot = CompStruct.Model.DomainTheta(ii_d)
        Xc = CompStruct.Model.DomainEcc(ii_d) * math.cos(CompStruct.Model.DomainEccAngle(ii_d))
        Yc = CompStruct.Model.DomainEcc(ii_d) * math.sin(CompStruct.Model.DomainEccAngle(ii_d))

        # define the angular grid for the domain boundary
        DTheta = math.pi / CompStruct.Model.DomainNth(ii_d)
        Theta = numpy.arange(- math.pi, DTheta, math.pi - DTheta)

        boundary_rec = 'n'
        if CompStruct.Mesh.ext_boundary_shape == 'rec':
            if CompStruct.Model.AddDomainLoc == 'ext' & ii_d >= CompStruct.Data.N_domain - 1:
                boundary_rec = 'y'
            if CompStruct.Model.AddDomainLoc == 'int' & ii_d == CompStruct.Data.N_domain:
                boundary_rec = 'y'

        # define the grid of nodes on the domain boundary (the domain is rotated
        # according to the input model)
        if boundary_rec == 'n':
            XBgrid = Xc + Rx * math.cos(Theta) * math.cos(ThetaRot) - Ry * math.sin(Theta) * math.sin(ThetaRot)
            YBgrid = Yc + Rx * math.cos(Theta) * math.sin(ThetaRot) + Ry * math.sin(Theta) * math.cos(ThetaRot)


        if boundary_rec == 'y':
            XBgrid = Rx * math.cos(Theta)
            YBgrid = Ry * math.sin(Theta)
            for ia in range(1,1,Theta):
                ang = Theta(ia)
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




    DomainBNodes = [XBgrid, YBgrid]

    # define the edges for the domain boundary
    DomainBEdges = numpy.empty((len(DomainBNodes),2))
    DomainBEdges[:, 1] = numpy.arange(1, len(DomainBNodes) + 1, 1)
    DomainBEdges[:, 2] = numpy.append(numpy.arange(2, len(DomainBNodes) + 1, 1), 1)

    # ed_num=[ed_num; ones(size(node,1),1)*i];

    # define the start and the end edge for the domain ii_d
    FaceStartEdge = len(Edges) - PrevEdgeSize + 1

    FaceEndEdge = numpy.size(numpy.append(Edges, (DomainBEdges + numpy.size(Edges, axis = 0)), axis = 0), axis = 0)

    # define the edges that belong to the face of the domain ii_d
    DomainFaces.append(range(FaceStartEdge,FaceEndEdge+1,1))
    PrevEdgeSize = numpy.size(DomainBEdges, axis = 0)

    # update the full list of edges and nodes of the domain boundaries
    Edges = numpy.append(Edges, (DomainBEdges + numpy.size(Edges, axis = 0)), axis = 0)
    Nodes = numpy.append(Nodes, DomainBNodes, axis = 0)



    # ===============================================================================
    # Constructing triangular mesh based on the supplied model geometry
    # ===============================================================================
    #CurDir = cd(CompStruct.Config.root_path)
    #cd('routines\Mesh2d v24\')
    #[MeshNodes, MeshTri, MeshFaceNums, ~, ~, ~, CompStruct] = CompStruct.Methods.MeshFaces(Nodes, Edges, DomainFaces, CompStruct)
    #cd(CurDir)

    # 2. Perform Delaunay triangulation
    tri = Delaunay(Nodes)

    # 3. Add additional points within each triangle (e.g., centroids)
    new_points = []
    for simplex in tri.simplices:
        triangle_points = Nodes[simplex]
        centroid = numpy.mean(triangle_points, axis=0)
        new_points.append(centroid)

    # Convert new_points to a NumPy array and combine with original points
    new_points = numpy.array(new_points)
    all_points = Nodes
    # all_points = np.vstack((points, new_points))

    # Re-triangulate with the expanded set of points
    tri_refined = Delaunay(all_points)

    # 4. Visualize the mesh
    #plt.figure(figsize=(8, 6))
    plt.triplot(all_points[:, 0], all_points[:, 1], tri_refined.simplices, color='blue', alpha=0.7)
    plt.plot(all_points[:, 0], all_points[:, 1], 'o', markersize=4, color='red')  # Plot all points
    plt.plot(new_points[:, 0], new_points[:, 1], 'o', markersize=4, color='red')  # Plot new points
    plt.title('2D Triangular Mesh with In-Triangle Points')
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')  # Ensure equal aspect ratio
    plt.show()

    #  [p,t,fnum] = meshfaces(node,edge,face,hdata,options);
    #
    # OUTPUTS
    #
    #  P     = Nx2 array of nodal XY co-ordinates.
    #  T     = Mx3 array of triangles as indicies into P, defined with a
    #          counter-clockwise node ordering.
    #  FNUM  = Mx1 array of face numbers for each triangle in T.
    #
    # INPUTS
    #
    # Blank arguments can be passed using the empty placeholder "[]".
    #
    # NODE defines the XY co-ordinates of the geometry vertices. The vertices
    # for all faces must be specified:
    #
    #  NODE = [x1,y1; x2,y2; etc], xy geometry vertices specified in any order.
    #
    # EDGE defines the connectivity between the points in NODE as a list of
    # edges. Edges for all faces must be specified:
    #
    #  EDGE = [n1 n2; n2 n3; etc], connectivity between nodes to form
    #                              geometry edges.
    #
    # FACE defines the edges included in each geometry face. Each face is a
    # vector of edge numbers, stored in a cell array:
    #
    #  FACE{1} = [e11,e12,etc]
    #  FACE{2} = [e21,e22,etc]
    #
    # HDATA is a structure containing user defined element size information.
    # HDATA can include the following fields:
    #
    #  hdata.hmax  = h0;                   Max allowable global element size.
    #  hdata.edgeh = [e1,h1; e2,h2; etc];  Element size on specified geometry
    #                                      edges.
    #  hdata.fun   = 'fun' or @fun;        User defined size function.
    #  hdata.args  = {arg1, arg2, etc};    Additional arguments for HDATA.FUN.

    # Inserting in the array defining the triangles the information on the
    # domain, which they belong to

    #MeshTri[:, 4] = MeshFaceNums
    #MeshTri = MeshTri

    # Find edges that belong to the boundaries of the domains
    #MeshNodes = MeshNodes
    #BoundaryEdges = CompStruct.Methods.FindBEdges(MeshNodes, MeshTri, CompStruct)
    # [p]=jigglemesh(p,e,t);

    return CompStruct
