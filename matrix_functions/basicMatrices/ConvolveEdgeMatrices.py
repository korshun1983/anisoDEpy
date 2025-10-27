import numpy as np
import os
from structures import BigCompStruct
from matrix_functions import basicMatrices


def ConvolveEdgeMatrices(CompStruct: BigCompStruct, BasicMatrices: basicMatrices):
    #   Part of the toolbox for solving problems of wave propagation
    # in arbitrary anisotropic inhomogeneous waveguides.
    # For details see User manual
    # and comments to the main script gen_aniso.m
    #
    #   ConvolveMatrices function computes the matrices, which indicate the expansion
    #  and the coefficients of the expansion of the double convolutions of the
    #  types conv(N,conv(N,N)), conv(dN,conv(N,N)), conv(N,conv(N,dN)), conv(dN,conv(N,dN)),
    #  into the intepolating functions L_j (L1, L2, L3).
    #  These convolutions enter into the expressions for the elements
    #  of the integrands for K_i and M matrices' elements.
    #  For HTTI solid.
    #
    #  Current implementation is for cubic elements.
    # NB!!! This implementation is specific for FE computations
    #
    #   [T.Zharnikov, SMR v0.3_08.2014]
    #
    # function [NNNConvMatrix,dNNNConvMatrix,NNdNConvMatrix,dNNdNConvMatrix]...
    #             = ConvolveMatrices(BasicMatrices)
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
    #   Code for ConvolveMatrices
    #
    ################################################################################
    # ===============================================================================
    # Initialization
    # ===============================================================================
    fname = os.path.join(CompStruct['Config']['root_path'], 'output', 'ConvEdgeMatrices.npz')

    if os.path.exists(fname):
        loaded_data = np.load(fname)
        BasicMatrices['NENENEConvMatrixInt'] = loaded_data['NENENEConvMatrixInt']
    else:
        # Identify number of nodes
        NEdge_nodes = CompStruct['Advanced']['NEdge_nodes']

        # Allocate memory for the matrices.
        # The implementation is for cubic elements, hence expansion is up to the
        # 3rd power in L_i

        NENENEConvMatrix = np.zeros((NEdge_nodes, NEdge_nodes, NEdge_nodes, 10, 10))

        LIntMatrix = BasicMatrices['LEdgeIntMatrix9'][0:10, 0:10]  # 0:9 вместо 1:10
        LIntMatrixNENENE = np.zeros_like(NENENEConvMatrix)

        # ===============================================================================
        # Compute the elements of all of the convolution matrices
        # and prepare large LIntMatrices for the integration
        # ===============================================================================

        # process all of the elements of the
        # NNNConvMatrix, dNNNConvMatrix, NNdNConvMatrix, dNNdNConvMatrix matrices
        print('conv')

        for bb in range(NEdge_nodes):
            for cc in range(NEdge_nodes):
                NEb = BasicMatrices['NLEdgeMatrix'][bb, :, :]  # 0-based индексация
                NEc = BasicMatrices['NLEdgeMatrix'][cc, :, :]  # 0-based индексация
                convn_NEb_NEc = np.convolve(NEb.flatten(), NEc.flatten()).reshape(7, 7)

                for aa in range(NEdge_nodes):
                    NEa = BasicMatrices['NLEdgeMatrix'][aa, :, :]  # 0-based индексация
                    conv_result = np.convolve(NEa.flatten(), convn_NEb_NEc.flatten()).reshape(10, 10)
                    NENENEConvMatrix[aa, bb, cc, :, :] = conv_result

                    LIntMatrixNENENE[aa, bb, cc, :, :] = LIntMatrix

        # ===============================================================================
        # Compute the integrals over the standard triangle of the monoms of the form
        # N_a*N_b*N_c, dN_a*N_b*N_c, N_a*N_b*dN_c, dN_a*N_b*dN_c
        # by multiplying elementwise convolution matrices NNN, etc. on the large
        # LIntMatrices and summing over the L1^i*L2^j*L3^k contributions
        # ===============================================================================
        print('conv_int')

        NENENEConvMatrix_unsummed = NENENEConvMatrix * LIntMatrixNENENE
        BasicMatrices['NENENEConvMatrixInt'] = np.sum(np.sum(NENENEConvMatrix_unsummed, axis=4), axis=3)

        # Save precomputed NNN integral matrices to the external file
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        np.savez(fname, NENENEConvMatrixInt=BasicMatrices['NENENEConvMatrixInt'])

    return BasicMatrices