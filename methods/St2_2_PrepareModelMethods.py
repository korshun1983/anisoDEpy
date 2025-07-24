import structures
from structures import BigCompStruct


def St2_2_PrepareModelMethods(CompStruct: BigCompStruct):
    #   Part of the toolbox for solving problems of wave propagation
    # in arbitrary anisotropic inhomogeneous waveguides.
    # For details see User manual
    # and comments to the main script gen_aniso.m
    #
    #   St2_2_PrepareModelMethods_sp_SAFE.m M-file
    #      St2_2_PrepareModelMethods_sp_SAFE.m prepares assigns
    # the methods, which will be used for the computations.
    # This selection is based on the model and the problem, selected as the input.
    #
    # NB! This implementation is specific to spectrum calculation
    # by the spectral method
    #
    #   [T.Zharnikov, D.Syresin, SMR v0.3_08.2014]
    #
    # function [CompStruct] = St2_2_PrepareModelMethods_sp_SAFE(CompStruct)
    #
    #  Inputs -
    #
    #       CompStuct - structure containing the parameters of the model
    #
    #  Outputs -
    #
    #       CompStuct - structure containing the parameters of the model after
    #                   updated with the handles to the methods, which will
    #                   be used for the spectral method computations
    #
    #  M-files required-
    #
    # Last Modified by Timur Zharnikov SMR v0.3_08.2014

    ################################################################################
    #
    #   Code for St2_2_PrepareModelMethods_sp_SM
    #
    ################################################################################
    # ===============================================================================
    # Initialization
    # ===============================================================================

    # ===============================================================================
    # Assigning the methods, which will be used for further steps
    # ===============================================================================

    cd(strcat(CompStruct.Config.root_path, 'routines\'))

    CompStruct.Methods.chebdif =

    @chebdif

    CompStruct.Methods.em_tensor_VTI =

    @em_tensor_VTI

    CompStruct.Methods.rot_c_ij =

    @rot_c_ij

    CompStruct.Methods.rot_matrix =

    @rot_matrix

    cd(CompStruct.Config.root_path)

    cd(CompStruct.Config.solver_path)
    CompStruct.Methods.St2_2_PrepareModelParams =

    @St2_2_PrepareModelParams_sp_SAFE

    CompStruct.Methods.St3_ProblemFormulation =

    @St3_ProblemFormulation_sp_SAFE

    CompStruct.Methods.St3_1_PrepareBasicMatrices =

    @St3_1_PrepareBasicMatrices_sp_SAFE

    CompStruct.Methods.St4_ComputeSolution =

    @St4_ComputeSolution_sp_SAFE

    cd(CompStruct.Config.root_path)

    # Assign the methods, which will compute
    # the set of physical properties, as well as
    # the spectral method matrices and blocks for each layer
    cd(strcat(CompStruct.Config.solver_path, 'routines\mesh\'))

    CompStruct.Methods.PrepareMesh =

    @PrepareMesh_sp_SAFE

    CompStruct.Methods.PrepareMeshBH =

    @PrepareMeshBH

    CompStruct.Methods.FindBEdges =

    @FindBEdges

    CompStruct.Methods.MakeContBEdges =

    @MakeContBEdges

    CompStruct.Methods.FindEdgeOrient =

    @FindEdgeOrient

    CompStruct.Methods.AddNodesCubic =

    @AddNodesCubic

    cd(strcat(CompStruct.Config.solver_path, 'routines\BasicMatrices\'))

    CompStruct.Methods.FindPos =

    @FindPos_sp_SAFE

    CompStruct.Methods.L1L2_int_matrix =

    @L1L2_int_matrix

    CompStruct.Methods.L1L2L3_int_matrix =

    @L1L2L3_int_matrix

    CompStruct.Methods.NL_matrix =

    @NL_matrix

    CompStruct.Methods.dNL_matrices =

    @dNL_matrices

    CompStruct.Methods.ConvolveMatrices =

    @ConvolveMatrices

    CompStruct.Methods.ConvolveEdgeMatrices =

    @ConvolveEdgeMatrices

    CompStruct.Methods.NLEdge_matrix =

    @NLEdge_matrix

    cd(strcat(CompStruct.Config.solver_path, 'routines\'))

    CompStruct.Methods.AssembleBasicMatrices =

    @AssembleBasicMatrices_sp_SAFE

    for ii_d = 1:CompStruct.Data.N_domain
    switch
    CompStruct.Model.DomainType
    {ii_d}
    case
    'fluid'
    CompStruct.Methods.PreparePhysProp
    {ii_d} =

    @PreparePhysProp_fluid_sp_SAFE

    case
    'HTTI'
    CompStruct.Methods.PreparePhysProp
    {ii_d} =

    @PreparePhysProp_HTTI_sp_SAFE

    otherwise
    end
    end

    cd(strcat(CompStruct.Config.solver_path, 'routines\FEMatrices\'))

    CompStruct.Methods.dxNL_matrix =

    @dxNL_matrix

    CompStruct.Methods.dyNL_matrix =

    @dyNL_matrix

    for ii_d = 1:CompStruct.Data.N_domain
    switch
    CompStruct.Model.DomainType
    {ii_d}
    case
    'fluid'
    CompStruct.Methods.MatricesParts_sp_SAFE
    {ii_d} =

    @MatricesParts_fluid_sp_SAFE_cubic

    CompStruct.Methods.getPhysProps
    {ii_d} =

    @getPhysProps_fluid

    CompStruct.Methods.KM_matrix
    {ii_d} =

    @KM_matrix_fluid

    CompStruct.Methods.KM_el_matrix
    {ii_d} =

    @KM_el_matrix_fluid

    case
    'HTTI'
    CompStruct.Methods.MatricesParts_sp_SAFE
    {ii_d} =

    @MatricesParts_HTTI_sp_SAFE_cubic

    CompStruct.Methods.MatricesPartsPML_sp_SAFE
    {ii_d} =

    @MatricesParts_HTTI_PML_sp_SAFE

    CompStruct.Methods.MatricesPartsABC_sp_SAFE
    {ii_d} =

    @MatricesParts_HTTI_ABC_sp_SAFE

    CompStruct.Methods.getPhysProps
    {ii_d} =

    @getPhysProps_HTTI

    CompStruct.Methods.KM_matrix
    {ii_d} =

    @KM_matrix_HTTI

    CompStruct.Methods.KM_el_matrix
    {ii_d} =

    @KM_el_matrix_HTTI

    otherwise
    end
    end
    if strcmpi(CompStruct.Model.AddDomain_Exist, 'yes')
    switch
    CompStruct.Model.AddDomainType
    case
    'pml'
    CompStruct.Methods.KM_el_matrix
    {CompStruct.Data.N_domain} =

    @KM_el_matrix_HTTI_PML

    case
    'abc'
    CompStruct.Methods.KM_el_matrix
    {CompStruct.Data.N_domain} =

    @KM_el_matrix_HTTI_ABC

    case
    'pml+abc'
    CompStruct.Methods.KM_el_matrix
    {CompStruct.Data.N_domain} =

    @KM_el_matrix_HTTI_PML_ABC

    otherwise
    fprintf(1, 'External Domain Type is not set!!!\n\n')
    return
    end


end

# assigning the methods, which are responsible for the interface condtions
# between the neighbour domains

for ii_int = 1:(CompStruct.Data.N_domain - 1)
switch[CompStruct.Model.DomainType
{ii_int}, CompStruct.Model.DomainType
{ii_int + 1}]
case
{['fluid', 'HTTI'], ['HTTI', 'fluid']}
CompStruct.Methods.IC_Matrices_sp_SAFE
{ii_int} =


@ICMatrices_fluid_HTTI_SAFE_cubic


CompStruct.Methods.IC_matrix
{ii_int} =


@IC_matrix_FS


CompStruct.Methods.IC_el_matrix
{ii_int} =


@IC_el_matrix_FS


CompStruct.Methods.AssembleFullMatrices
{ii_int} =


@AssembleFullMatrices_fs_SAFE_cubic


case
{['fluid', 'fluid'], ['HTTI', 'HTTI']}
CompStruct.Methods.IC_Matrices_sp_SAFE
{ii_int} =


@ICMatrices_ff_ss_SAFE_cubic


CompStruct.Methods.AssembleFullMatrices
{ii_int} =


@AssembleFullMatrices_ff_ss_SAFE_cubic


otherwise
end
end

# assign the method, which will address the outer boundary condition

ii_int = CompStruct.Data.N_domain
switch
CompStruct.Model.BCType
{ii_int}
case
'rigid'
CompStruct.Methods.AssembleFullMatrices
{ii_int} =


@AssembleFullMatrices_rigid_SAFE_cubic


case
'free'
CompStruct.Methods.AssembleFullMatrices
{ii_int} =


@AssembleFullMatrices_free_SAFE_cubic


otherwise
end

CompStruct.Methods.RemoveRedundantVariables =


@RemoveRedundantVariables_SAFE


cd(CompStruct.Config.root_path)

end
