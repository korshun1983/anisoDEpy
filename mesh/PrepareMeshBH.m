def PrepareMeshBH(CompStruct):
// %   Part of the toolbox for solving problems of wave propagation
// % in arbitrary anisotropic inhomogeneous waveguides.
// % For details see User manual
// % and comments to the main script gen_aniso.m
// %
// % PrepareMeshBH M-file
// %      PrepareMeshBH, by itself, prepares the mesh
// %      for elliptically layered boreholes.
// %      It is based on the open source Mesh2D package.
// %
// %   [T.Zharnikov, D.Syresin, SMR v0.3_08.2014]
// %
// % function [p,e,t] = PrepareMeshBH(CompStruct)
// %
// %  Inputs -
// %
// %       CompStuct - structure containing the parameters of the model;
// %
// %       ii_l - number of the layer, for which the physical properties are
// %              introduced
// %
// %  Outputs -
// %
// %       PhysProp - structure, containing physical properties of the ii_l-th
// %       layer
// %
// %  M-files required-
// %
// % Last Modified by Timur Zharnikov SMR v0.3_08.2014
//
// %###############################################################################
// %
// %   Code for PrepareMeshBH, original version written by Denis Syresin, SMR, 2013
// %
// %###############################################################################
//
//
// %===============================================================================
// % Preparing input data for the mesh construction -
// % domain boundaries' nodes, edges, faces
// %===============================================================================
Edges = [];
Nodes = [];
// ed_num = [];
PrevEdgeSize = 0;

//% construct the lists of nodes, edges, and faces for the domain boundaries
for ii_d = 1:CompStruct.Data.N_domain
    
    % define the domain geometry
    Rx = CompStruct.Model.DomainRx(ii_d);
    Ry = CompStruct.Model.DomainRy(ii_d);
    
    //% find the coordinates of the domain center
    ThetaRot = CompStruct.Model.DomainTheta(ii_d);
    Xc = CompStruct.Model.DomainEcc(ii_d)*cos(CompStruct.Model.DomainEccAngle(ii_d));
    Yc = CompStruct.Model.DomainEcc(ii_d)*sin(CompStruct.Model.DomainEccAngle(ii_d));

    //% define the angular grid for the domain boundary
    DTheta = pi/CompStruct.Model.DomainNth(ii_d);
    Theta = (( - pi):DTheta:( pi - DTheta ))';
    
    boundary_rec='n';
    if strcmpi(CompStruct.Mesh.ext_boundary_shape,'rec')
       if strcmpi(CompStruct.Model.AddDomainLoc,'ext') && ii_d>=CompStruct.Data.N_domain-1
          boundary_rec='y';
       end
       if strcmpi(CompStruct.Model.AddDomainLoc,'int') && ii_d==CompStruct.Data.N_domain
          boundary_rec='y';
       end
    end
        
    % define the grid of nodes on the domain boundary (the domain is rotated 
    % according to the input model)
    if boundary_rec=='n'
       XBgrid = Xc + Rx*cos(Theta)*cos(ThetaRot) - Ry*sin(Theta)*sin(ThetaRot);
       YBgrid = Yc + Rx*cos(Theta)*sin(ThetaRot) + Ry*sin(Theta)*cos(ThetaRot);
    end
    
    if boundary_rec=='y'
       XBgrid = Rx*cos(Theta); YBgrid = Ry*sin(Theta);
       for ia=1:length(Theta)
           ang=Theta(ia);
           if -pi<=ang && ang<=-(3./4.)*pi
              XBgrid(ia) = -Rx;
              YBgrid(ia) = -Rx*tan(pi+ang);
           end
           if (3./4)*pi<=ang && ang<=pi
              XBgrid(ia) = -Rx;
              YBgrid(ia) =  Rx*tan(pi-ang);
           end
           if -(3./4)*pi<=ang && ang<=-(1./4.)*pi
              XBgrid(ia) =  Ry*tan(pi/2+ang);
              YBgrid(ia) = -Ry;
           end
           if -(1./4)*pi<=ang && ang<=(1./4.)*pi
              XBgrid(ia) = Rx;
              YBgrid(ia) = Rx*tan(ang);
           end
           if (1./4)*pi<=ang && ang<=(3./4.)*pi
              XBgrid(ia) = Ry*tan(pi/2-ang);
              YBgrid(ia) = Ry;
           end
       end
    end
    
    DomainBNodes = [XBgrid, YBgrid];
    
    % define the edges for the domain boundary
    DomainBEdges = [(1:size(DomainBNodes,1))',[(2:size(DomainBNodes,1))'; 1]];
    % ed_num=[ed_num; ones(size(node,1),1)*i];
    
    % define the start and the end edge for the domain ii_d
    FaceStartEdge = size(Edges,1) - PrevEdgeSize + 1 ;
    FaceEndEdge = size([Edges; ( DomainBEdges + size(Edges,1) )],1);
    
    % define the edges that belong to the face of the domain ii_d
    DomainFaces{ii_d} = FaceStartEdge:FaceEndEdge;
    PrevEdgeSize = size(DomainBEdges,1);
    
    % update the full list of edges and nodes of the domain boundaries
    Edges = [Edges; DomainBEdges + size(Edges,1)];
    Nodes = [Nodes; DomainBNodes];
    
end

%===============================================================================
% Constructing triangular mesh based on the supplied model geometry
%===============================================================================
CurDir = cd(CompStruct.Config.root_path);
cd('routines\Mesh2d v24\');
[MeshNodes,MeshTri,MeshFaceNums,~,~,~,CompStruct] = ...
    CompStruct.Methods.MeshFaces(Nodes,Edges,DomainFaces,CompStruct);
cd(CurDir);

// %  [p,t,fnum] = meshfaces(node,edge,face,hdata,options);
// %
// % OUTPUTS
// %
// %  P     = Nx2 array of nodal XY co-ordinates.
// %  T     = Mx3 array of triangles as indicies into P, defined with a
// %          counter-clockwise node ordering.
// %  FNUM  = Mx1 array of face numbers for each triangle in T.
// %
// % INPUTS
// %
// % Blank arguments can be passed using the empty placeholder "[]".
// %
// % NODE defines the XY co-ordinates of the geometry vertices. The vertices
// % for all faces must be specified:
// %
// %  NODE = [x1,y1; x2,y2; etc], xy geometry vertices specified in any order.
// %
// % EDGE defines the connectivity between the points in NODE as a list of
// % edges. Edges for all faces must be specified:
// %
// %  EDGE = [n1 n2; n2 n3; etc], connectivity between nodes to form
// %                              geometry edges.
// %
// % FACE defines the edges included in each geometry face. Each face is a
// % vector of edge numbers, stored in a cell array:
// %
// %  FACE{1} = [e11,e12,etc]
// %  FACE{2} = [e21,e22,etc]
// %
// % HDATA is a structure containing user defined element size information.
// % HDATA can include the following fields:
// %
// %  hdata.hmax  = h0;                   Max allowable global element size.
// %  hdata.edgeh = [e1,h1; e2,h2; etc];  Element size on specified geometry
// %                                      edges.
// %  hdata.fun   = 'fun' or @fun;        User defined size function.
// %  hdata.args  = {arg1, arg2, etc};    Additional arguments for HDATA.FUN.

//% Inserting in the array defining the triangles the information on the
//% domain, which they belong to
MeshTri(:,4) = MeshFaceNums;
MeshTri = MeshTri';

//% Find edges that belong to the boundaries of the domains
MeshNodes = MeshNodes'
BoundaryEdges = CompStruct.Methods.FindBEdges(MeshNodes,MeshTri,CompStruct)
//% [p]=jigglemesh(p,e,t);

return MeshNodes,BoundaryEdges,MeshTri,CompStruct
