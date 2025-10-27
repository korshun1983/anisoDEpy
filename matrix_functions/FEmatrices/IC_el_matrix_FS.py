import numpy as np
from matrix_functions.matrix_definitions import ElMatrices
from structures import BigCompStruct

def IC_el_matrix_FS(BasicMatrices, CompStruct:BigCompStruct, ElPhysProps, EdgeProps, EdgeNodesPos):

    #   Part of the toolbox for solving problems of wave propagation
    # in arbitrary anisotropic inhomogeneous waveguides.
    # For details see User manual
    # and comments to the main script gen_aniso.m
    #
    #   KM_el_matrix_HTTI function computes the matrices, which indicates the expansion
    #  and the coefficients of the expansion of the integrands for K_i and M matrices' elements
    #  into the intepolating functions L_j (L1, L2, L3).
    #  For HTTI solid.
    #  The K_i and M matrices are defined according to Bartoli_Marzani_DiScalea_Viola_JSoundVibration_v295_p685_2006.
    #  N_m = (N1*E_N, ..., Nn*E_N) = kron(N_vector,E_N) (3 x 3Nn matrix)
    #  B2_m = Lz_m * N_m = Lz_m * kron(N_vector,E_N) (6 x 3Nn matrix)
    #  B1_m = Lx_m*d/dx N_m + Ly_m*d/dy N_m = ((Lx_m*d/dx + Ly_m*d/dy)*N1, ..., (Lx_m*d/dx + Ly_m*d/dy)*Nn)
    #
    #  K1_m = B1_adjoint * C_m * B1
    #  K1_m ab :::  = conv(conj(B1_m ia :::), conv(C_ij :::, B1_m jb :::) ) (3Nn x 3Nn matrix)
    #  K2_m = B1_adjoint * C_m * B2 - B2_adjoint * C_m * B1
    #  K2_m ab :::  = conv(conj(B1_m ia :::), conv(C_ij :::, B2_m jb :::) )
    #                   - conv(conj(B2_m ia :::), conv(C_ij :::, B1_m jb :::) ) (3Nn x 3Nn matrix)
    #  K3_m = B2_adjoint * C_m * B2
    #  K3_m ab :::  = conv(conj(B2_m ia :::), conv(C_ij :::, B2_m jb :::) ) (3Nn x 3Nn matrix)
    #  M_m = N_adjoint * Rho_m * N
    #  M_m ab :::  = conv(conj(N_m ia :::), conv(C_ij :::, N_m jb :::) ) (3Nn x 3Nn matrix)
    #
    #  Current implementation is for cubic elements.
    # NB!!! This implementation is specific for FE computations
    #
    #   [T.Zharnikov, SMR v0.3_08.2014]
    #
    # function [B1tCB1Matrix, B1tCB2Matrix, B2tCB1Matrix, B2tCB2Matrix, NtRhoNMatrix]...
    #            = KM_el_matrix_HTTI(CMatrix, RhoMatrix, BasicMatrices, ...
    #                            NLMatrix, dxNLMatrix, dyNLMatrix)
    #
    #  Inputs -
    #
    #       CompStruct - structure containing parameters of the model
    #
    #  Outputs -
    #
    #       Pos - structure array, indicating positions of the blocks corresponding
    #               to the various hierarchy levels (layers, harmonics, variables)
    #               inside the full matrix representation matrix.
    #
    #  M-files required-
    #
    # Last Modified by Timur Zharnikov SMR v0.3_08.2014

    ################################################################################
    #
    #   Code for KM_el_matrix_HTTI
    #
    ################################################################################
    # ===============================================================================
    # Initialization
    # ===============================================================================

    # Identify number of nodes

    # E_matrix = eye(3)
    NFlEdge_nodes = CompStruct.Advanced.NEdge_nodes
    NSEdge_nodes = 3 * CompStruct.Advanced.NEdge_nodes

    # Allocate memory for the matrices.
    # The implementation is for cubic elements, hence expansion is up to the
    # 3rd power in L_i
    ElMatrices.NfltRhonNMatrix = np.zeros((NFlEdge_nodes, NSEdge_nodes))

    # ===============================================================================
    # Compute the elements of FS interface condition matrices according to the above
    # presented formulas
    # ===============================================================================

    # Prepare various properties
    Rho = ElPhysProps.DfEl.RhoVec(EdgeNodesPos.Df)  # fluid
    OnesMatrix = np.ones[(NFlEdge_nodes)]
    # OnesVarMatrix = ones(CompStruct.Data.DVarNum(ii_d))
    IdMatrix = np.eye(3)  # HTTI solid
    # Msize = CompStruct.Data.DVarNum(ii_d)*N_nodes

    # process all of the elements of the stiffness and mass matrices
    # expansion coefficients (N_nodes x N_nodes x ... x ... x ... )
    Rho_nE3_large = np.zeros([NFlEdge_nodes, NFlEdge_nodes, NSEdge_nodes])
    # Rho_large = np.zeros(size(BasicMatrices.NENENEConvMatrixInt))
    BasicMatrices.NENENEConvMatrixIntLarge = np.zeros([NFlEdge_nodes, NFlEdge_nodes, NSEdge_nodes])
    NodeNumbers = np.linspace(1, NFlEdge_nodes, NFlEdge_nodes)
    for ii in range(0,3):  # HTTI solid, CompStruct.Data.DVarNum(ii_d)
        BasicMatrices.NENENEConvMatrixIntLarge[:,:,(ii + 3 * (NodeNumbers - 1)) ] = BasicMatrices.NENENEConvMatrixInt
    #         ( ii + CompStruct.Data.DVarNum(ii_d)*( NodeNumbers - 1 )) ) = ...


    for kk in range(0,NFlEdge_nodes):
        Rho_nE3_kk = Rho(kk) * EdgeProps.Normal.T * IdMatrix
        Rho_nE3_large[:, kk,:] = np.reshape(np.kron(OnesMatrix, Rho_nE3_kk), (NFlEdge_nodes, 1, NSEdge_nodes))

    NfltRhonNMatrix_unsummed = Rho_nE3_large * BasicMatrices.NENENEConvMatrixIntLarge
    ElMatrices.NfltRhonNMatrix = np.squeeze(np.sum(NfltRhonNMatrix_unsummed, axis = 1))

    # # process all of the elements of the 2d N matrix (3 x 3*N_nodes)
    # for nN = 1:NEdge_nodes
    #     NflMatrix(1, nN,:,:) = BasicMatrices.NLEdgeMatrix(nN,:,:)
    #     for jj = 1:3
    #         for ii = 1:3
    #             nNMatrix( ( 3*(nN - 1) + jj ),:,:) = nNMatrix( ( 3*(nN - 1) + jj ),:,:) + ...
    #                 EdgeProps.Normal(ii)*BasicMatrices.E3(ii,jj)*BasicMatrices.NLEdgeMatrix(nN,:,:)
    #
    #
    #
    #
    # # process all of the elements of the 2d
    # # Nt(fluid)RhoNnMatrix matrix ( N_nodes x 3*N_nodes )
    # for aa = 1:(NEdge_nodes)
    #     for bb = 1:(3*NEdge_nodes)
    #         # mass matrix term - the kinetic energy
    #         Rho = squeeze(ElPhysProps.RhoMatrix(:,:))
    #         nNb = squeeze(nNMatrix(bb,:,:))
    #         Nfl_t_a = conj(squeeze(NflMatrix(1,aa,:,:)))
    #         NfltRhonNMatrix(aa,bb,:,:,:) = conv(Nfl_t_a,conv(Rho,nNb))
    #
    return ElMatrices

