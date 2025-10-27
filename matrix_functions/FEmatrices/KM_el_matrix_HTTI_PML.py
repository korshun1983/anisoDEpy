import numpy as np
from matrix_functions.matrix_definitions import ElMatrices
from structures import BigCompStruct

def KM_el_matrix_HTTI_PML(BasicMatrices, CompStruct: BigCompStruct, FEMatrices, ii_d, ii_el, ElPhysProps, TriProps):

    # NB!!!!!!! WE HAVE PML LAYER

    #   Part of the toolbox for solving problems of wave propagation
    # in arbitrary anisotropic inhomogeneous waveguides.
    # For details see User manual
    # and comments to the main script gen_aniso.m
    #
    #   KM_el_matrix_HTTI function computes the matrices, which indicates the expansion
    #  and the coefficients of the expansion of the integrands for K_i and M matrices.T elements
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
    Rho_large = np.zeros([Msize, N_nodes, Msize])
    LxTCijLx_dxL_dxL_large = np.zeros([3, 3, Msize, N_nodes, Msize])
    LxTCijLz_dxL_large = np.zeros([3, Msize, N_nodes, Msize])
    LzTCijLx_dxL_large = np.zeros([3, Msize, N_nodes, Msize])
    LzTCijLz_large = np.zeros([Msize, N_nodes, Msize])


    TriNodes = FEMatrices.DElements[ii_d][0: 9, ii_el]

    # PML layer, its thickness is always in SH wavelength
    # gamma(r)=1+sigma(r)/iw
    # sigma(r)=H(r-r_zv)*alpha*(r-r_zv)^2
    # H is Heaviside step function, r_zv is the inner radius of PML layer
    # alpha is the constant: 1) =8*2pi*f_peak/L_pml^2, L_pml is the length of PML layer

    f_peak = CompStruct.f_grid[CompStruct.if_grid] * CompStruct.Misc.F_conv  # Hz
    w_peak = 2. * np.pi * f_peak
    cone = complex(0., 1.)

    r_zv = CompStruct.Model.DomainRx[-2]
    L_ext_layer = CompStruct.Model.DomainRx[-1] - r_zv

    for kk in range(0,N_nodes):
        Lx = BasicMatrices.Lx
        Ly = BasicMatrices.Ly
        var_df_xy = 1.

        if ii_d == CompStruct.Data.N_domain:

            xy_node = FEMatrices.MeshNodes[0:1, TriNodes[kk]]

            if CompStruct.Model.PML_method == 1:  # Circle PML

                r_node = np.sqrt(xy_node[0] ** 2 + xy_node[1] ** 2)

                if CompStruct.Model.AddDomainLoc.lower() == 'ext':
                    r_zv = min(CompStruct.Model.DomainRx[-2], CompStruct.Model.DomainRy[-2])
                    Lr_ext_layer = min(CompStruct.Model.DomainRx[-1], CompStruct.Model.DomainRy[-1]) - r_zv

                if CompStruct.Model.AddDomainLoc.lower() == 'int':
                    Lr_ext_layer = CompStruct.Model.AddDomainL_m
                    r_zv = min(CompStruct.Model.DomainRx[-1], CompStruct.Model.DomainRy[-1]) - Lr_ext_layer

                if r_node > r_zv:
                    ivar = CompStruct.Model.PML_degree

                    sigma_r = ((r_node - r_zv) / Lr_ext_layer) ** ivar
                    gamma_r = cone * CompStruct.Model.PML_factor * sigma_r
                    r_tilde = r_node - cone * (CompStruct.Model.PML_factor / (ivar + 1)) * ((r_node - r_zv) / Lr_ext_layer) ** (ivar + 1)

                    var_xy = xy_node[0] ** 2 / (gamma_r * r_node ^ 2) + xy_node[1] ^ 2 / (r_tilde * r_node)
                    var_yx = xy_node[1] ** 2 / (gamma_r * r_node ^ 2) + xy_node[0] ^ 2 / (r_tilde * r_node)
                    var_all = (1. / (gamma_r * r_node ^ 2) - 1. / (r_tilde * r_node)) * xy_node[0] * xy_node[1]

                    Lx = var_xy * BasicMatrices.Lx + var_all * BasicMatrices.Ly
                    Ly = var_all * BasicMatrices.Lx + var_yx * BasicMatrices.Ly

                    var_df_xy = gamma_r * r_tilde / r_node

            if CompStruct.Model.PML_method == 2:  # Rectange PML
                ivar = CompStruct.Model.PML_degree

                if CompStruct.Model.AddDomainLoc.lower() == 'ext':
                    x_zv = CompStruct.Model.DomainRx[-2]
                    Lx_ext_layer = CompStruct.Model.DomainRx[-1] - x_zv
                    y_zv = CompStruct.Model.DomainRy[-2]
                    Ly_ext_layer = CompStruct.Model.DomainRy[-1] - y_zv

                if CompStruct.Model.AddDomainLoc.lower() == 'int':
                    Lx_ext_layer = CompStruct.Model.AddDomainL_m
                    x_zv = CompStruct.Model.DomainRx[-1] - Lx_ext_layer
                    Ly_ext_layer = CompStruct.Model.AddDomainL_m
                    y_zv = CompStruct.Model.DomainRx[-1] - Ly_ext_layer

                gamma_x = 1.
                if abs(xy_node[1]) > x_zv:
                    sigma_x = ((abs(xy_node[0]) - x_zv) / Lx_ext_layer) ** ivar
                    gamma_x = 1. - cone * CompStruct.Model.PML_factor * sigma_x

                    LL_xx = 1. / gamma_x
                    Lx = LL_xx * BasicMatrices.Lx

                gamma_y = 1.
                if abs(xy_node[1]) > y_zv:
                    sigma_y = ((abs(xy_node[1]) - y_zv) / Ly_ext_layer) ** ivar
                    gamma_y = 1. - cone * CompStruct.Model.PML_factor * sigma_y

                    LL_yy = 1. / gamma_y
                    Ly = LL_yy * BasicMatrices.Ly

                var_df_xy = gamma_x * gamma_y

    #     !!!!!!!!!!!!!!!!!!!
    LxTCijLx_kk = np.multi_dot([np.conj(Lx.T),np.squeeze(CijMatrix[kk,:,:]),Lx])
    LxTCijLy_kk = np.conj(Lx.T)*np.squeeze(CijMatrix[kk,:,:])*Ly
    LyTCijLx_kk = np.conj(Ly.T)*np.squeeze(CijMatrix[kk,:,:])*Lx
    LyTCijLy_kk = np.conj(Ly.T)*np.squeeze(CijMatrix[kk,:,:])*Ly
    LxTCijLz_kk = np.conj(Lx.T)*np.squeeze(CijMatrix[kk,:,:])*Lz
    LzTCijLx_kk = np.conj(Lz.T)*np.squeeze(CijMatrix[kk,:,:])*Lx
    LyTCijLz_kk = np.conj(Ly.T)*np.squeeze(CijMatrix[kk,:,:])*Lz
    LzTCijLy_kk = np.conj(Lz.T)*np.squeeze(CijMatrix[kk,:,:])*Ly
    LzTCijLz_kk = np.conj(Lz.T)*np.squeeze(CijMatrix[kk,:,:])*Lz



    LxTCijLx_kk = LxTCijLx_kk * var_df_xy
    LxTCijLy_kk = LxTCijLy_kk * var_df_xy
    LyTCijLx_kk = LyTCijLx_kk * var_df_xy
    LyTCijLy_kk = LyTCijLy_kk * var_df_xy
    LxTCijLz_kk = LxTCijLz_kk * var_df_xy
    LzTCijLx_kk = LzTCijLx_kk * var_df_xy
    LyTCijLz_kk = LyTCijLz_kk * var_df_xy
    LzTCijLy_kk = LzTCijLy_kk * var_df_xy
    LzTCijLz_kk = LzTCijLz_kk * var_df_xy

    Rho_large[:, kk,:] = np.reshape(np.kron(OnesMatrix, Rho[kk] * IdMatrix), Msize, 1, Msize)
    Rho_large[:, kk,:] = Rho_large[:, kk,:]*var_df_xy

    LzTCijLz_large[:, kk,:] = np.kron(OnesMatrix, LzTCijLz_kk)
    for ii in range(0,3):
        LxTCijLz_dxL_large[ii,:, kk,:] = np.kron(OnesMatrix, (LxTCijLz_kk * dxL[ii] +
                                                           LyTCijLz_kk * dyL[ii] ))
        LzTCijLx_dxL_large[ii,:, kk,:] = np.kron(OnesMatrix, (LzTCijLx_kk * dxL[ii] +
                                                           LzTCijLy_kk * dyL[ii] ))
    for jj in range(0,3):
        LxTCijLx_dxL_dxL_large[ii, jj,:, kk,:] = np.kron(OnesMatrix, (LxTCijLx_kk * dxL(ii) * dxL(jj) +
                                                               LxTCijLy_kk * dxL(ii) * dyL(jj) +
                                                               LyTCijLx_kk * dyL(ii) * dxL(jj) +
                                                               LyTCijLy_kk * dyL(ii) * dyL(jj) ))




    NtRhoNMatrix_unsummed = Rho_large * BasicMatrices.NNNConvMatrixIntLarge[ii_d]
    ElMatrices.NtRhoNMatrix = np.squeeze(sum(NtRhoNMatrix_unsummed, axis = 1))
    B1tCB1Matrix_unsummed = LxTCijLx_dxL_dxL_large * BasicMatrices.dNNdNConvMatrixIntLarge[ii_d]
    ElMatrices.B1tCB1Matrix = np.squeeze(sum(sum(sum(B1tCB1Matrix_unsummed, axis = 0), axis = 1), axis = 3))
    B1tCB2Matrix_unsummed = LxTCijLz_dxL_large * BasicMatrices.dNNNConvMatrixIntLarge[ii_d]
    ElMatrices.B1tCB2Matrix = np.squeeze(sum(sum(B1tCB2Matrix_unsummed, axis = 0), axis = 2))
    B2tCB1Matrix_unsummed = LzTCijLx_dxL_large * BasicMatrices.NNdNConvMatrixIntLarge[ii_d]
    ElMatrices.B2tCB1Matrix = np.squeeze(sum(sum(B2tCB1Matrix_unsummed, axis = 0), axis = 2))
    B2tCB2Matrix_unsummed = LzTCijLz_large * BasicMatrices.NNNConvMatrixIntLarge[ii_d]
    ElMatrices.B2tCB2Matrix = np.squeeze(sum(B2tCB2Matrix_unsummed, axis = 1))

    return ElMatrices