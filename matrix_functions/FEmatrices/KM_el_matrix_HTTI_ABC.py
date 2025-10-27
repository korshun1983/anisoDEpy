import numpy as np
from matrix_functions.matrix_definitions import ElMatrices
from structures import BigCompStruct

def KM_el_matrix_HTTI_ABC(BasicMatrices, CompStruct: BigCompStruct, FEMatrices, ii_d, ii_el, ElPhysProps, TriProps):
    
    # NB!!!!!!! WE HAVE ABC LAYER
    
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
    #  N_m = (N1*E_N, ..., Nn*E_N) = np.kron(N_vector,E_N) (3 x 3Nn matrix)
    #  B2_m = Lz_m * N_m = Lz_m * np.kron(N_vector,E_N) (6 x 3Nn matrix)
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
    # E_matrix = np.eye(3)
    N_nodes = CompStruct.Advanced.N_nodes
    
    # Allocate memory for the matrices.
    # The implementation is for cubic elements, hence expansion is up to the 3rd power in L_i
    B1Matrix = np.zeros(6, CompStruct.Data.DVarNum[ii_d] * N_nodes, 4, 4, 4)
    B2Matrix = np.zeros(6, CompStruct.Data.DVarNum[ii_d] * N_nodes, 4, 4, 4)
    NMatrix = np.zeros(3, CompStruct.Data.DVarNum[ii_d] * N_nodes, 4, 4, 4)
    
    ElMatrices.B1tCB1Matrix = np.zeros(CompStruct.Data.DVarNum[ii_d] * N_nodes, CompStruct.Data.DVarNum[ii_d] * N_nodes)
    ElMatrices.B1tCB2Matrix = np.zeros(CompStruct.Data.DVarNum[ii_d] * N_nodes, CompStruct.Data.DVarNum[ii_d] * N_nodes)
    ElMatrices.B2tCB1Matrix = np.zeros(CompStruct.Data.DVarNum[ii_d] * N_nodes, CompStruct.Data.DVarNum[ii_d] * N_nodes)
    ElMatrices.B2tCB2Matrix = np.zeros(CompStruct.Data.DVarNum[ii_d] * N_nodes, CompStruct.Data.DVarNum[ii_d] * N_nodes)
    
    NtRhoNMatrix = np.zeros(CompStruct.Data.DVarNum[ii_d] * N_nodes, CompStruct.Data.DVarNum[ii_d] * N_nodes)
    
    # ===============================================================================
    # Compute the elements of K_i and M matrices according to the above presented formulas
    # ===============================================================================
    # Prepare various properties
    Rho = ElPhysProps.RhoVec
    CijMatrix = ElPhysProps.CijMatrix
    dxL = TriProps.dxL
    dyL = TriProps.dyL
    Lx = BasicMatrices.Lx
    Ly = BasicMatrices.Ly
    Lz = BasicMatrices.Lz
    
    OnesMatrix = np.ones(N_nodes)
    OnesVarNum = np.ones(CompStruct.Data.DVarNum[ii_d])
    IdMatrix = np.eye(CompStruct.Data.DVarNum[ii_d])
    Msize = CompStruct.Data.DVarNum[ii_d] * N_nodes

    # process all of the elements of the stiffness and mass matrices
    # expansion coefficients (N_nodes x N_nodes x ... x ... x ... )
    Rho_large = np.zeros((Msize, N_nodes, Msize))
    LxTCijLx_dxL_dxL_large = np.zeros((3, 3, Msize, N_nodes, Msize))
    LxTCijLz_dxL_large = np.zeros((3, Msize, N_nodes, Msize))
    LzTCijLx_dxL_large = np.zeros((3, Msize, N_nodes, Msize))
    LzTCijLz_large = np.zeros((Msize, N_nodes, Msize))

    TriNodes = FEMatrices.DElements[ii_d][1: 10, ii_el]

    # ABC layer, its thickness is always in SH wavelength
    # cij=cij*(1-i*a*((r-r_max)/h)^n), a - abc factor, n - abc degree, h - layer length, r_max of main core
    # => 1/Q = (w/v_SH)*(a/2)*r*(r-r_max)/h^n
    # Thus, a ~ Advanced.ABC_factor*(v_SH/w)       if Advanced.ABC_account_r = 'none'
    # or    a ~ Advanced.ABC_factor*(v_SH/w)*(1/r) if Advanced.ABC_account_r = 'yes'

    f_peak = CompStruct.f_grid[CompStruct.if_grid] * CompStruct.Misc.F_conv  # Hz
    w_peak = 2. * np.pi * f_peak
    cone = complex(0., 1.)

    r_zv = CompStruct.Model.DomainRx[-2]
    L_ext_layer = CompStruct.Model.DomainRx[-1] - r_zv

    for kk in range(0,N_nodes):
        if ii_d == CompStruct.Data.N_domain:
            alpha = CompStruct.Model.ABC_factor
            xy_node = FEMatrices.MeshNodes[0:1, TriNodes[kk]]
            r_node = np.sqrt(xy_node[0] * xy_node[0] + xy_node[2] * xy_node[2])
            sigma = alpha * (np.abs(r_node - r_zv) / L_ext_layer) ** CompStruct.Model.ABC_degree
            if CompStruct.Model.ABC_account_r == 1:
                sigma = sigma / r_node

            CijMatrix = CijMatrix * (1. + cone * sigma)

        #     !!!!!!!!!!!!!!!!!!!
        LxTCijLx_kk = np.multi_dot([np.conj(Lx.T),np.squeeze(CijMatrix[kk,:,:]),Lx])
        LxTCijLy_kk = np.multi_dot([np.conj(Lx.T),np.squeeze(CijMatrix[kk,:,:]),Ly])
        LyTCijLx_kk = np.multi_dot([np.conj(Ly.T),np.squeeze(CijMatrix[kk,:,:]),Lx])
        LyTCijLy_kk = np.multi_dot([np.conj(Ly.T),np.squeeze(CijMatrix[kk,:,:]),Ly])
        LxTCijLz_kk = np.multi_dot([np.conj(Lx.T),np.squeeze(CijMatrix[kk,:,:]),Lz])
        LzTCijLx_kk = np.multi_dot([np.conj(Lz.T),np.squeeze(CijMatrix[kk,:,:]),Lx])
        LyTCijLz_kk = np.multi_dot([np.conj(Ly.T),np.squeeze(CijMatrix[kk,:,:]),Lz])
        LzTCijLy_kk = np.multi_dot([np.conj(Lz.T),np.squeeze(CijMatrix[kk,:,:]),Ly])
        LzTCijLz_kk = np.multi_dot([np.conj(Lz.T),np.squeeze(CijMatrix[kk,:,:]),Lz])


        Rho_large[:, kk,:] = np.reshape(np.kron(OnesMatrix, Rho(kk) * IdMatrix), Msize, 1, Msize)

        LzTCijLz_large[:, kk,:] = np.kron(OnesMatrix, LzTCijLz_kk)
        for ii in range(0,3):
            LxTCijLz_dxL_large[ii,:, kk,:] = np.kron(OnesMatrix, (LxTCijLz_kk * dxL(ii) + LyTCijLz_kk * dyL(ii) ))
            LzTCijLx_dxL_large[ii,:, kk,:] = np.kron(OnesMatrix, (LzTCijLx_kk * dxL(ii) + LzTCijLy_kk * dyL(ii) ))
            for jj in range(0,3):
                LxTCijLx_dxL_dxL_large[ii, jj,:, kk,:] = np.kron(OnesMatrix, (LxTCijLx_kk * dxL(ii) * dxL(jj) +
                                                                              LxTCijLy_kk * dxL(ii) * dyL(jj) +
                                                                              LyTCijLx_kk * dyL(ii) * dxL(jj) +
                                                                              LyTCijLy_kk * dyL(ii) * dyL(jj) ))
    
    
    
    
    NtRhoNMatrix_unsummed = Rho_large * BasicMatrices.NNNConvMatrixIntLarge[ii_d]
    ElMatrices.NtRhoNMatrix = np.squeeze(np.sum(NtRhoNMatrix_unsummed, axis = 1))
    B1tCB1Matrix_unsummed = LxTCijLx_dxL_dxL_large * BasicMatrices.dNNdNConvMatrixIntLarge[ii_d]
    ElMatrices.B1tCB1Matrix = np.squeeze(np.sum(np.sum(np.sum(B1tCB1Matrix_unsummed, axis = 0), axis = 1), axis = 3))
    B1tCB2Matrix_unsummed = LxTCijLz_dxL_large * BasicMatrices.dNNNConvMatrixIntLarge[ii_d]
    ElMatrices.B1tCB2Matrix = np.squeeze(np.sum(np.sum(B1tCB2Matrix_unsummed, axis = 0), axis = 2))
    B2tCB1Matrix_unsummed = LzTCijLx_dxL_large * BasicMatrices.NNdNConvMatrixIntLarge[ii_d]
    ElMatrices.B2tCB1Matrix = np.squeeze(np.sum(np.sum(B2tCB1Matrix_unsummed, axis = 0), axis = 2))
    B2tCB2Matrix_unsummed = LzTCijLz_large * BasicMatrices.NNNConvMatrixIntLarge[ii_d]
    ElMatrices.B2tCB2Matrix = np.squeeze(np.sum(B2tCB2Matrix_unsummed, axis = 1))
    
    return ElMatrices