import numpy as np
from matrix_functions.matrix_definitions import ElMatrices
from structures import BigCompStruct

def KM_el_matrix_fluid(BasicMatrices, CompStruct:BigCompStruct, FEMatrices, ii_d, ii_el, ElPhysProps, TriProps):

    #   Part of the toolbox for solving problems of wave propagation
    # in arbitrary anisotropic inhomogeneous waveguides.
    # For details see User manual
    # and comments to the main script gen_aniso.m
    #
    #   KM_el_matrix_fluid function computes the matrices, which indicates the expansion
    #  and the coefficients of the expansion of the integrands for K_i and M matrices' elements
    #  into the intepolating functions L_j (L1, L2, L3).
    #  For fluid.
    #  The K_i and M matrices are defined according to Bartoli_Marzani_DiScalea_Viola_JSoundVibration_v295_p685_2006.
    #  N_m = (N1, ..., Nn) = kron(N_vector,E_1) (1 x Nn matrix)
    #  B2_m = Lz_m * N_m = Lz_m * kron(N_vector,E_N) (3 x Nn matrix)
    #  B1_m = Lx_m*d/dx N_m + Ly_m*d/dy N_m = ((Lx_m*d/dx + Ly_m*d/dy)*N1, ..., (Lx_m*d/dx + Ly_m*d/dy)*Nn)
    #  B1_m = Lx_m*sum_i_1^3 ( dxLi*dNvec_i ) + Ly_m*sum_i_1^3 ( dyLi*dNvec_i )
    #  rho = sum rho_k*N_k
    #  Lx_mT*Lx_m = 1
    #  Ly_mT*Ly_m = 1
    #  Lz_mT*Lz_m = 1
    #  Lx_mT*Ly_m = 0
    #  Lx_mT*Lz_m = 0
    #  Ly_mT*Lz_m = 0
    #
    #  K1_m = B1_adjoint * rho * B1
    #  K1_m ab :::  = sum_k rho_k *sum_i_1_3 sum_j_1_3 (dxLi*dxLj*Lx_mT*Lx_m +
    #     dyLi*dyLj*Ly_mT*Ly_m + dxLi*dyLj*Lx_mT*Ly_m + dyLi*dxLj*Ly_mT*Lx_m )*
    #     conv(dN_iT(a) :::, conv(N_k :::, dN_j(b) :::) ) (Nn x Nn matrix)
    #  K2_m = B1_adjoint * rho * B2 - B2_adjoint * rho * B1
    #  K2_m ab :::  is proportional to Lx_mT*Ly_m and hence is equal to 0 (Nn x Nn matrix)
    #  K2_m ab ::: = conv(conj(B1_m ia :::), conv(C_ij :::, B2_m jb :::) )
    #                   - conv(conj(B2_m ia :::), conv(C_ij :::, B1_m jb :::) )
    #  K3_m = B2_adjoint * rho * B2
    #  K3_m ab :::  = conv(conj(B2_m ia :::), conv(C_ij :::, B2_m jb :::) ) (Nn x Nn matrix)
    #  K3_m ab :::  = sum_k rho_k*conv(NT_a :::, conv(N_k :::, N_b :::) ) (Nn x Nn matrix)
    #  M_m = N_adjoint * Rho2_Lambda * N
    #  M_m ab :::  = conv(conj(N_m ia :::), conv(C_ij :::, N_m jb :::) ) (Nn x Nn matrix)
    #  M_m ab :::  = sum_k rho2_lambda_k*conv(NT_a :::, conv(N_k :::, N_b :::) ) (Nn x Nn matrix)
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
    N_nodes = CompStruct.Advanced.N_nodes

    # Allocate memory for the matrices.
    # The implementation is for cubic elements, hence expansion is up to the 3rd power in L_i
    ElMatrices.B1tCB1Matrix = np.zeros((CompStruct.Data.DVarNum(ii_d) * N_nodes, CompStruct.Data.DVarNum(ii_d) * N_nodes))
    ElMatrices.B1tCB2Matrix = np.zeros((CompStruct.Data.DVarNum(ii_d) * N_nodes, CompStruct.Data.DVarNum(ii_d) * N_nodes))
    ElMatrices.B2tCB1Matrix = np.zeros((CompStruct.Data.DVarNum(ii_d) * N_nodes, CompStruct.Data.DVarNum(ii_d) * N_nodes))
    ElMatrices.B2tCB2Matrix = np.zeros((CompStruct.Data.DVarNum(ii_d) * N_nodes, CompStruct.Data.DVarNum(ii_d) * N_nodes))

    ElMatrices.NtRho2LambdaNMatrix = np.zeros((CompStruct.Data.DVarNum(ii_d) * N_nodes, CompStruct.Data.DVarNum(ii_d) * N_nodes))
    # ===============================================================================
    # Compute the elements of K_i and M matrices according to the above presented formulas
    # ===============================================================================

    # Prepare various properties
    Rho = ElPhysProps.RhoVec
    Rho2_Lambda = ElPhysProps.Rho2LambdaVec
    dxL = TriProps.dxL
    dyL = TriProps.dyL

    # process all of the elements of the stiffness and mass matrices
    # expansion coefficients (N_nodes x N_nodes x ... x ... x ... )
    Rho2_Lambda_large = np.zeros(np.shape(BasicMatrices.NNNConvMatrixInt))
    Rho_large = np.zeros(np.shape(BasicMatrices.NNNConvMatrixInt))
    Rho_dxL_dxL_large = np.zeros(np.shape(BasicMatrices.dNNdNConvMatrixInt))

    for kk in range(0,N_nodes):
        Rho2_Lambda_large[:, kk,:] = Rho2_Lambda[kk]
        Rho_large[:, kk,:] = Rho[kk]
        for ii in range (0,3):
                for jj in range (0,3):
                    Rho_dxL_dxL_large[ii, jj,:, kk,:]  = Rho[kk] * (dxL[ii] * dxL[jj] + dyL[ii] * dyL[jj])


    NtRho2LambdaNMatrix_unsummed = Rho2_Lambda_large * BasicMatrices.NNNConvMatrixInt
    ElMatrices.NtRho2LambdaNMatrix = np.squeeze(np.sum(NtRho2LambdaNMatrix_unsummed, axis = 1))

    B2tCB2Matrix_unsummed = Rho_large * BasicMatrices.NNNConvMatrixInt
    ElMatrices.B2tCB2Matrix = np.squeeze(np.sum(B2tCB2Matrix_unsummed, axis = 1))
    #                 ElMatrices.B1tCB2Matrix(aa,bb,:,:,:) = 0
    #                 ElMatrices.B2tCB1Matrix(aa,bb,:,:,:) = 0
    B1tCB1Matrix_unsummed = Rho_dxL_dxL_large * BasicMatrices.dNNdNConvMatrixInt
    ElMatrices.B1tCB1Matrix = np.squeeze(np.sum(np.sum(np.sum(B1tCB1Matrix_unsummed, axis = 0), axis = 1), axis = 3))

    return ElMatrices
