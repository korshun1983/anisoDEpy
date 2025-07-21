function [DBEdges] = FindEdgeOrient(DBEdges,MeshTri,MeshNodes)
    
%   Part of the toolbox for solving problems of wave propagation
% in arbitrary anisotropic inhomogeneous waveguides.
% For details see User manual 
% and comments to the main script gen_aniso.m
%
%   MakeContBEdges function makes continuous boundary out of the edges,
% which were identified as the parts of the domain boundary by 
% FindBEdges procedure
% NB!!! This implementation is specific for FE computations
%
%   [D.Syresin, T.Zharnikov, SMR v0.3_08.2014]
%
% function [ContBEdges] = MakeContBEdges(DBEdges,MeshTri,MeshNodes)
%
%  Inputs - 
%
%       CompStruct - structure containing parameters of the model
%
%  Outputs -
%
%       Pos - structure array, indicating positions of the blocks corresponding
%               to the various hierarchy levels (layers, harmonics, variables) 
%               inside the full matrix representation matrix.
%
%  M-files required-
%
% Last Modified by Timur Zharnikov SMR v0.3_08.2014

%###############################################################################
%
%   Code for MakeContBEdges
%
%###############################################################################
%===============================================================================
% Initialization
%===============================================================================


%===============================================================================
% By checking all of the edges in all of the triangles, find the number
% of the domains that they belong to (1 - maximum number, 2 - minimum number)
%===============================================================================

% find triangles containing the first node of the edge
Node1 = DBEdges(1);
[Node1Row,Node1Col] = find(MeshTri(1:3,:) == Node1); 
% among found triangles, select those containing the second node of the edge 
Node2 = DBEdges(2);
[Node2Row,Node2Col] = find(MeshTri(1:3,Node1Col) == Node2);
% select columns for triangle, containing the edge under consideration
% note that there is only one such triangle, because only triangles
% belonging to the domain in question are passed to this procedure!!!
ColumnIndex = Node1Col(Node2Col);
% find the third node of the triangle, containing the edge under consideration
Node3Row = (MeshTri(1:3,ColumnIndex) ~= Node1 & MeshTri(1:3,ColumnIndex) ~= Node2);
Node3 = MeshTri(Node3Row,ColumnIndex);

% Define the vector, which is directed inside the triangle
InnerVector = [(MeshNodes(1,Node1) + MeshNodes(1,Node2))/2 - MeshNodes(1,Node3), ...
    (MeshNodes(2,Node1) + MeshNodes(2,Node2))/2 - MeshNodes(2,Node3)];

% Define the vector, which is directed along the path of the triangle in a
% counterclockwise direction
TangentVector = [MeshNodes(1,Node2) - MeshNodes(1,Node1),MeshNodes(2,Node2) - MeshNodes(2,Node1)];

% Compute the orientation of the cross product of the InnerVector and the
% TangentVector. Since they are in the xy plane, the cross product lies
% in the axial (z) direction and its sign determines the orientation of the
% (InnerVector,TangentVector) pair
SignOrientation = sign(cross([TangentVector 0],[InnerVector 0]));
% depending on the orientation of the pair, switch the direction of the
% edge as necessary so that the path goes in the counterclockwise direction
if (SignOrientation(3) == 1)
    DBEdges(1:2) = DBEdges([2,1]);
end

end