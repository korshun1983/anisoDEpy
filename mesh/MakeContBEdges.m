function [ContDBEdges] = MakeContBEdges(DBEdges,MeshTri,MeshNodes,CompStruct)
    
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
% Starting from the first edge, find its continuation in the array of edges,
% which belong to the boundary of the selected domain
%===============================================================================

DBEdges(:,1) = CompStruct.Methods.FindEdgeOrient(DBEdges(:,1),MeshTri,MeshNodes);
% select the first edge of the boundary
ContDBEdges = DBEdges(:,1);
% remove this edge from the original array (from the search)
DBEdges(:,1) = [];
while DBEdges
    % find the edge, which is the continuation of the previous one
    [NextNodeRow,NextNodeCol] = find(DBEdges(1:2,:) == ContDBEdges(2,end));
    % add the found edge to the ordered array (connected edges) observing
    % the proper order of nodes
    if NextNodeRow == 1
        ContDBEdges = [ContDBEdges  DBEdges(1:3,NextNodeCol)];
        DBEdges(:,NextNodeCol) = [];
    elseif NextNodeRow == 2
        % switch the order of nodes in the edge as necessary
        ContDBEdges = [ContDBEdges  DBEdges([2 1 3],NextNodeCol)];
        DBEdges(:,NextNodeCol) = [];
    else
        disp('error');
        break;
    end
end

end