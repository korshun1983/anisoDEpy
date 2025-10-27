import numpy as np
import scipy.io
import os
from structures import BigCompStruct
from matrix_functions import basicMatrices


def ConvolveMatrices(CompStruct: BigCompStruct, BasicMatrices: basicMatrices):
    """
    Part of the toolbox for solving problems of wave propagation
    in arbitrary anisotropic inhomogeneous waveguides.
    For details see User manual
    and comments to the main script gen_aniso.m

    ConvolveMatrices function computes the matrices, which indicate the expansion
    and the coefficients of the expansion of the double convolutions of the
    types conv(N,conv(N,N)), conv(dN,conv(N,N)), conv(N,conv(N,dN)), conv(dN,conv(N,dN)),
    into the intepolating functions L_j (L1, L2, L3).
    These convolutions enter into the expressions for the elements
    of the integrands for K_i and M matrices' elements.
    For HTTI solid.

    Current implementation is for cubic elements.
    NB!!! This implementation is specific for FE computations

    [T.Zharnikov, SMR v0.3_08.2014]

    function [NNNConvMatrix,dNNNConvMatrix,NNdNConvMatrix,dNNdNConvMatrix]...
                = ConvolveMatrices(BasicMatrices)

    Inputs -

        CompStruct - structure containing parameters of the model

    Outputs -

        Pos - structure array, indicating positions of the blocks corresponding
                to the various hierarchy levels (layers, harmonics, variables)
                inside the full matrix representation matrix.

    M-files required-

    Last Modified by Timur Zharnikov SMR v0.3_08.2014
    """

    ################################################################################
    #
    #   Code for ConvolveMatrices
    #
    ################################################################################
    # ===============================================================================
    # Initialization
    # ===============================================================================
    fname = CompStruct['Config']['root_path'] + 'output/ConvMatrices.mat'
    # fname=''

    if os.path.exists(fname):
        mat_data = scipy.io.loadmat(fname)

        BasicMatrices['NNNConvMatrixInt'] = mat_data['NNNConvMatrixInt']
        BasicMatrices['dNNNConvMatrixInt'] = mat_data['dNNNConvMatrixInt']
        BasicMatrices['NNdNConvMatrixInt'] = mat_data['NNdNConvMatrixInt']
        BasicMatrices['dNNdNConvMatrixInt'] = mat_data['dNNdNConvMatrixInt']

        #     NNNConvMatrix_size = [ N_nodes,N_nodes,N_nodes,10,10,10 ]
        #     dNNNConvMatrix_size = [ 3,N_nodes,N_nodes,N_nodes,10,10,10 ]
        #     NNdNConvMatrix_size = [ 3,N_nodes,N_nodes,N_nodes,10,10,10 ]
        #     dNNdNConvMatrix_size = [ 3,3,N_nodes,N_nodes,N_nodes,10,10,10 ]

        #     NNNConvMatrixInt = reshape(full(NNNConvMatrixInt_sparse),NNNConvMatrix_size)
        #     dNNNConvMatrixInt = reshape(full(dNNNConvMatrixInt_sparse),dNNNConvMatrix_size)
        #     NNdNConvMatrixInt = reshape(full(NNdNConvMatrixInt_sparse),NNdNConvMatrix_size)
        #     dNNdNConvMatrixInt = reshape(full(dNNdNConvMatrixInt_sparse),dNNdNConvMatrix_size)

    else:
        # Identify number of nodes
        N_nodes = CompStruct['Advanced']['N_nodes']

        # Allocate memory for the matrices.
        # The implementation is for cubic elements, hence expansion is up to the
        # 3rd power in L_i

        NNNConvMatrix = np.zeros((N_nodes, N_nodes, N_nodes, 10, 10, 10))
        dNNNConvMatrix = np.zeros((3, N_nodes, N_nodes, N_nodes, 10, 10, 10))
        NNdNConvMatrix = np.zeros((3, N_nodes, N_nodes, N_nodes, 10, 10, 10))
        dNNdNConvMatrix = np.zeros((3, 3, N_nodes, N_nodes, N_nodes, 10, 10, 10))

        LIntMatrix = BasicMatrices['LIntMatrix9'][0:10, 0:10, 0:10]
        LIntMatrixNNN = np.zeros_like(NNNConvMatrix)
        LIntMatrixNNdN = np.zeros_like(NNdNConvMatrix)
        LIntMatrixdNNN = np.zeros_like(NNdNConvMatrix)
        LIntMatrixdNNdN = np.zeros_like(dNNdNConvMatrix)

        # ===============================================================================
        # Compute the elements of all of the convolution matrices
        # and prepare large LIntMatrices for the integration
        # ===============================================================================

        # process all of the elements of the
        # NNNConvMatrix, dNNNConvMatrix, NNdNConvMatrix, dNNdNConvMatrix matrices
        print('conv')
        # tic;
        for bb in range(N_nodes):
            for cc in range(N_nodes):
                Nb = BasicMatrices['NLMatrix'][bb, :, :, :]
                Nc = BasicMatrices['NLMatrix'][cc, :, :, :]
                convn_Nb_Nc = np.convolve(Nb.flatten(), Nc.flatten()).reshape((13, 13, 13))[0:10, 0:10, 0:10]

                for aa in range(N_nodes):
                    Na = BasicMatrices['NLMatrix'][aa, :, :, :]
                    NNNConvMatrix[aa, bb, cc, :, :, :] = np.convolve(Na.flatten(), convn_Nb_Nc.flatten()).reshape(
                        (19, 19, 19))[0:10, 0:10, 0:10]

                    LIntMatrixNNN[aa, bb, cc, :, :, :] = LIntMatrix

                    for ii in range(3):
                        dNa_i = BasicMatrices['dNLMatrix'][ii, aa, :, :, :]
                        dNNNConvMatrix[ii, aa, bb, cc, :, :, :] = np.convolve(dNa_i.flatten(),
                                                                              convn_Nb_Nc.flatten()).reshape(
                            (13, 13, 13))[0:10, 0:10, 0:10]

                        LIntMatrixdNNN[ii, aa, bb, cc, :, :, :] = LIntMatrix

                for jj in range(3):
                    dNc_j = BasicMatrices['dNLMatrix'][jj, cc, :, :, :]
                    convn_Nb_dNc_j = np.convolve(Nb.flatten(), dNc_j.flatten()).reshape((13, 13, 13))[0:10, 0:10, 0:10]

                    for aa in range(N_nodes):
                        Na = BasicMatrices['NLMatrix'][aa, :, :, :]
                        NNdNConvMatrix[jj, aa, bb, cc, :, :, :] = np.convolve(Na.flatten(),
                                                                              convn_Nb_dNc_j.flatten()).reshape(
                            (13, 13, 13))[0:10, 0:10, 0:10]

                        LIntMatrixNNdN[jj, aa, bb, cc, :, :, :] = LIntMatrix

                        for ii in range(3):
                            dNa_i = BasicMatrices['dNLMatrix'][ii, aa, :, :, :]
                            dNNdNConvMatrix[ii, jj, aa, bb, cc, :, :, :] = np.convolve(dNa_i.flatten(),
                                                                                       convn_Nb_dNc_j.flatten()).reshape(
                                (13, 13, 13))[0:10, 0:10, 0:10]

                            LIntMatrixdNNdN[ii, jj, aa, bb, cc, :, :, :] = LIntMatrix
        # toc;
        # ===============================================================================
        # Compute the integrals over the standard triangle of the monoms of the form
        # N_a*N_b*N_c, dN_a*N_b*N_c, N_a*N_b*dN_c, dN_a*N_b*dN_c
        # by multiplying elementwise convolution matrices NNN, etc. on the large
        # LIntMatrices and summing over the L1^i*L2^j*L3^k contributions
        # ===============================================================================
        print('conv_int')
        # tic;
        NNNConvMatrix_unsummed = NNNConvMatrix * LIntMatrixNNN
        BasicMatrices['NNNConvMatrixInt'] = np.squeeze(
            np.sum(np.sum(np.sum(NNNConvMatrix_unsummed, axis=5), axis=4), axis=3))
        dNNNConvMatrix_unsummed = dNNNConvMatrix * LIntMatrixdNNN
        BasicMatrices['dNNNConvMatrixInt'] = np.squeeze(
            np.sum(np.sum(np.sum(dNNNConvMatrix_unsummed, axis=6), axis=5), axis=4))
        NNdNConvMatrix_unsummed = NNdNConvMatrix * LIntMatrixNNdN
        BasicMatrices['NNdNConvMatrixInt'] = np.squeeze(
            np.sum(np.sum(np.sum(NNdNConvMatrix_unsummed, axis=6), axis=5), axis=4))
        dNNdNConvMatrix_unsummed = dNNdNConvMatrix * LIntMatrixdNNdN
        BasicMatrices['dNNdNConvMatrixInt'] = np.squeeze(
            np.sum(np.sum(np.sum(dNNdNConvMatrix_unsummed, axis=7), axis=6), axis=5))

        # Save precomputed NNN integral matrices to the external file
        #     NNNConvMatrix_s1 = N_nodes*N_nodes*N_nodes
        #     dNNNConvMatrix_s1 = 3*N_nodes*N_nodes*N_nodes
        #     NNdNConvMatrix_s1 = 3*N_nodes*N_nodes*N_nodes
        #     dNNdNConvMatrix_s1 = 3*3*N_nodes*N_nodes*N_nodes
        #     NNNConvMatrix_s2 = 10*10*10
        #     dNNNConvMatrix_s2 = 10*10*10
        #     NNdNConvMatrix_s2 = 10*10*10
        #     dNNdNConvMatrix_s2 = 10*10*10

        #     NNNConvMatrixInt_sparse = sparse(reshape(NNNConvMatrixInt,NNNConvMatrix_s1,NNNConvMatrix_s2))
        #     dNNNConvMatrixInt_sparse = sparse(reshape(dNNNConvMatrixInt,dNNNConvMatrix_s1,dNNNConvMatrix_s2))
        #     NNdNConvMatrixInt_sparse = sparse(reshape(NNdNConvMatrixInt,NNdNConvMatrix_s1,NNdNConvMatrix_s2))
        #     dNNdNConvMatrixInt_sparse = sparse(reshape(dNNdNConvMatrixInt,dNNdNConvMatrix_s1,dNNdNConvMatrix_s2))

        scipy.io.savemat(fname, {
            'NNNConvMatrixInt': BasicMatrices['NNNConvMatrixInt'],
            'dNNNConvMatrixInt': BasicMatrices['dNNNConvMatrixInt'],
            'NNdNConvMatrixInt': BasicMatrices['NNdNConvMatrixInt'],
            'dNNdNConvMatrixInt': BasicMatrices['dNNdNConvMatrixInt']
        })
        # toc;

    # ===============================================================================
    # Compute the elements of large convolution matrices for all of the domains
    # ===============================================================================

    N_nodes = CompStruct['Advanced']['N_nodes']
    for ii_d in range(CompStruct['Data']['N_domain']):
        OnesVarNum = np.ones((CompStruct['Data']['DVarNum'][ii_d], CompStruct['Data']['DVarNum'][ii_d]))
        Msize = CompStruct['Data']['DVarNum'][ii_d] * CompStruct['Advanced']['N_nodes']

        BasicMatrices['dNNdNConvMatrixIntLarge'][ii_d] = np.zeros((3, 3, Msize, N_nodes, Msize))
        BasicMatrices['dNNNConvMatrixIntLarge'][ii_d] = np.zeros((3, Msize, N_nodes, Msize))
        BasicMatrices['NNdNConvMatrixIntLarge'][ii_d] = np.zeros((3, Msize, N_nodes, Msize))
        BasicMatrices['NNNConvMatrixIntLarge'][ii_d] = np.zeros((Msize, N_nodes, Msize))

        for ii_k in range(N_nodes):
            BasicMatrices['NNNConvMatrixIntLarge'][ii_d][:, ii_k, :] = np.kron(
                np.squeeze(BasicMatrices['NNNConvMatrixInt'][:, ii_k, :]), OnesVarNum)

            for ii_c in range(3):
                BasicMatrices['NNdNConvMatrixIntLarge'][ii_d][ii_c, :, ii_k, :] = np.kron(
                    np.squeeze(BasicMatrices['NNdNConvMatrixInt'][ii_c, :, ii_k, :]), OnesVarNum)
                BasicMatrices['dNNNConvMatrixIntLarge'][ii_d][ii_c, :, ii_k, :] = np.kron(
                    np.squeeze(BasicMatrices['dNNNConvMatrixInt'][ii_c, :, ii_k, :]), OnesVarNum)

                for ii_c2 in range(3):
                    BasicMatrices['dNNdNConvMatrixIntLarge'][ii_d][ii_c, ii_c2, :, ii_k, :] = np.kron(
                        np.squeeze(BasicMatrices['dNNdNConvMatrixInt'][ii_c, ii_c2, :, ii_k, :]), OnesVarNum)

    return BasicMatrices