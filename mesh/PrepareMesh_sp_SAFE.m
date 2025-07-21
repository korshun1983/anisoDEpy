function [MeshNodes,BoundaryEdges,MeshTri,MeshProps,CompStruct] = PrepareMesh_sp_SAFE(CompStruct)

%   Part of the toolbox for solving problems of wave propagation
% in arbitrary anisotropic inhomogeneous waveguides.
% For details see User manual 
% and comments to the main script gen_aniso.m
%
% PrepareMesh_sp_SAFE M-file      
%      PrepareMesh_sp_SAFE, by itself, prepares the mesh 
%      for the computational domains.
%      It is based on the open source Mesh2D package.
%
%   [T.Zharnikov, D.Syresin, SMR v0.3_08.2014]
%
% function [PhysProp] = PrepareMesh_sp_SAFE(CompStruct,ii_l)
%
%  Inputs - 
%
%       CompStuct - structure containing the parameters of the model;
%
%       ii_l - number of the layer, for which the physical properties are
%              introduced
%
%  Outputs -
%
%       PhysProp - structure, containing physical properties of the ii_l-th
%       layer
%
%  M-files required-
%
% Last Modified by Timur Zharnikov SMR v0.3_08.2014

%###############################################################################
%
%   Code for PrepareMesh_sp_SAFE
%
%###############################################################################


%===============================================================================
% Constructing triangular mesh based on the supplied model geometry
%===============================================================================

%warning off all;
if isfield(CompStruct,'Mesh')
   if ~isfield(CompStruct.Mesh,'ext_boundary_shape')
      CompStruct.Mesh.ext_boundary_shape='cir';
   else
      if ~strcmpi(CompStruct.Mesh.ext_boundary_shape,'cir') && ~strcmpi(CompStruct.Mesh.ext_boundary_shape,'rec')
         fprintf(1,'\tMesh.ext_boundary_shape has invalid value!!!\n\n');
         return;
      end 
   end
else
   CompStruct.Mesh.ext_boundary_shape='cir'; 
end
[MeshNodes,BoundaryEdges,MeshTri,CompStruct] = CompStruct.Methods.PrepareMeshBH(CompStruct);

% Check orientation of triangles (Nikitin 25.03.2016)
% numtria=size(MeshTri,2);
% ddd=zeros(length(numtria),1);
% for itr=1:numtria
%     ijk=MeshTri(1:3,itr);
%     ddd(itr)=0.5*det( [[1 MeshNodes(:,ijk(1))']; [1 MeshNodes(:,ijk(2))']; [1 MeshNodes(:,ijk(3))']] );
% end
% ddd_min=min(ddd); ddd_max=max(ddd);

%sort triangles according to their subdomain number
MeshTri = (sortrows(MeshTri',4))';

% add new nodes to each element for cubic interpolation
[MeshNodes,MeshTri,MeshProps] = CompStruct.Methods.AddNodesCubic(MeshNodes,MeshTri);
%hold on;
% plot(p(1,:),p(2,:),'.');

end
