function [BEdges] = FindBEdges(MeshNodes,MeshTri,CompStruct)
      
%   Part of the toolbox for solving problems of wave propagation
% in arbitrary anisotropic inhomogeneous waveguides.
% For details see User manual 
% and comments to the main script gen_aniso.m
%
%   FindBEdges function identifies the edges, which belong
% to the domain boundaries.
% NB!!! This implementation is specific for FE computations
%
%   [D.Syresin, T.Zharnikov, SMR v0.3_08.2014]
%
% function [BEdges] = FindBEdges(MeshNodes,MeshTri,N_domain)
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
%   Code for FindBEdges
%
%###############################################################################
%===============================================================================
% Initialization
%===============================================================================

% creating the matrix of node connections
EdgeDomain = zeros(max(max(MeshTri(1:3,:))),max(max(MeshTri(1:3,:))),2);

%===============================================================================
% By checking all of the edges in all of the triangles, find the number
% of the domains that they belong to (1 - maximum number, 2 - minimum number)
%===============================================================================

for ii = 1:size(MeshTri,2)
    node1 = min(MeshTri([1,2],ii));
    node2 = max(MeshTri([1,2],ii));
    EdgeDomain(node1,node2,2) = min(MeshTri(4,ii),EdgeDomain(node1,node2,1));
    EdgeDomain(node1,node2,1) = max(MeshTri(4,ii),EdgeDomain(node1,node2,1));
          
    node1 = min(MeshTri([2,3],ii));
    node2 = max(MeshTri([2,3],ii));
    EdgeDomain(node1,node2,2) = min(MeshTri(4,ii),EdgeDomain(node1,node2,1));
    EdgeDomain(node1,node2,1) = max(MeshTri(4,ii),EdgeDomain(node1,node2,1));
    
    node1 = min(MeshTri([3,1],ii));
    node2 = max(MeshTri([3,1],ii));
    EdgeDomain(node1,node2,2) = min(MeshTri(4,ii),EdgeDomain(node1,node2,1));  
    EdgeDomain(node1,node2,1) = max(MeshTri(4,ii),EdgeDomain(node1,node2,1));
end

% find the edges that lie on the boundaries of the domains
EdgeOnBoundary = EdgeDomain(:,:,1) - EdgeDomain(:,:,2);
[BEdRow,BEdCol] = find( EdgeOnBoundary ~= 0 );
BEdges = zeros(3,length(BEdRow));

for ii=1:length(BEdRow)
    % attribute the start and the end nodes of the edge
    BEdges(1,ii) = BEdRow(ii);
    BEdges(2,ii) = BEdCol(ii);
    % identify the smallest domain number, which the edge belongs to
    if EdgeDomain(BEdRow(ii),BEdCol(ii),2) == 0
        % if the edge was encountered in just one triangle - belongs to the
        % outer boundary of the whole computational domain
        BEdges(3,ii) = EdgeDomain(BEdRow(ii),BEdCol(ii),1);
    else
        % identifies the outer boundary of the corresponding domain
        BEdges(3,ii) = EdgeDomain(BEdRow(ii),BEdCol(ii),2);
    end
end

BEdges = sortrows(BEdges',3)';
ContinuousBEdges = [];
for ii_d = 1:1:CompStruct.Data.N_domain
    % make continuous boundary for the domain, 
    % note that only the triangles belonging to the domain in question are
    % selected !!! 
    % Hence there is just one outer edge in the boundary traingles!!!
    ContinuousBEdges_d = CompStruct.Methods.MakeContBEdges...
        (BEdges(:,BEdges(3,:)==ii_d),MeshTri(:,MeshTri(4,:)==ii_d),MeshNodes,CompStruct);
    ContinuousBEdges = [ContinuousBEdges ...
        ContinuousBEdges_d];
end
BEdges = ContinuousBEdges;

end