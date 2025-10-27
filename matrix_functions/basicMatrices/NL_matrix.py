import numpy as np

def NL_matrix():
    #   Part of the toolbox for solving problems of wave propagation
    # in arbitrary anisotropic inhomogeneous waveguides.
    # For details see User manual
    # and comments to the main script gen_aniso.m
    #
    #   NL_matrix function prepares the matrix, which indicates the expansion
    #  and the coefficients of the expansion of the shape functions N_i
    #  into the intepolating functions L_j (L1, L2, L3).
    #  The formulas for the expansion and the coefficients are described
    #  in the book of Zienkiewicz (8.18), (8.36), (8.37).
    #  Current implementation is for cubic elements.
    # NB!!! This implementation is specific for FE computations
    #
    #   [T.Zharnikov, SMR v0.3_08.2014]
    #
    # function [NLMatrix] = NL_matrix()
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
    #   Code for NL_matrix
    #
    ################################################################################
    #===============================================================================
    # Initialization
    #===============================================================================

    # allocate memory for [NLMatrix] (expansion of shape functions into the
    # interpolating functions

    # for cubic elements, there are 10 shape functions and the expansion is up
    # to the 3rd power
    NLMatrix = np.zeros((10, 4, 4, 4))

    #===============================================================================
    # Set the matrix of node coordinates in internal L-coordinates of the
    # triangle
    #===============================================================================

    NodeLCoord = np.array([
        [1, 0, 0],      #1
        [0, 1, 0],      #2
        [0, 0, 1],      #3
        [2/3, 1/3, 0],  #4
        [1/3, 2/3, 0],  #5
        [0, 2/3, 1/3],  #6
        [0, 1/3, 2/3],  #7
        [2/3, 0, 1/3],  #8
        [1/3, 0, 2/3],  #9
        [1/3, 1/3, 1/3] #10
    ])

    #===============================================================================
    # Compute the elements of the NLMatrix
    # the expansion is of the form: N_i = NLMatrix(i,j,k,l)
    # N_i = sum_{j,k,l = 1 .. N } NLMatrix(i,j,k,l) L1^{j-1} L2^{k-1} L3^{l-1}
    #===============================================================================

    # define NLMatrix for the cubic element
    # there are 10 shape functions

    # corner nodes - 1, 2, 3
    # N_i = 1/2 * L_i*(3*L_i - 1)*(3*L_i - 2 ) = 9/2*L_i^3 - 9/2 * L_i^2 + L_i

    NLMatrix[0, 1, 0, 0] = 1
    NLMatrix[0, 2, 0, 0] = -9/2
    NLMatrix[0, 3, 0, 0] = 9/2

    NLMatrix[1, 0, 1, 0] = 1
    NLMatrix[1, 0, 2, 0] = -9/2
    NLMatrix[1, 0, 3, 0] = 9/2

    NLMatrix[2, 0, 0, 1] = 1
    NLMatrix[2, 0, 0, 2] = -9/2
    NLMatrix[2, 0, 0, 3] = 9/2

    # mid-side (edge) nodes - 4-9

    # N_4 = 9/2 * L_1*(3*L_1 - 1)*L_2 = 27/2*L_1^2*L_2 - 9/2 * L_1*L_2
    NLMatrix[3, 1, 1, 0] = -9/2
    NLMatrix[3, 2, 1, 0] = 27/2

    # N_5 = 9/2 * L_1*(3*L_2 - 1)*L_2 = 27/2*L_2^2*L_1 - 9/2 * L_1*L_2
    NLMatrix[4, 1, 1, 0] = -9/2
    NLMatrix[4, 1, 2, 0] = 27/2

    # N_6 = 9/2 * L_2*(3*L_2 - 1)*L_3 = 27/2*L_2^2*L_3 - 9/2 * L_2*L_3
    NLMatrix[5, 0, 1, 1] = -9/2
    NLMatrix[5, 0, 2, 1] = 27/2

    # N_7 = 9/2 * L_2*(3*L_3 - 1)*L_3 = 27/2*L_3^2*L_2 - 9/2 * L_2*L_3
    NLMatrix[6, 0, 1, 1] = -9/2
    NLMatrix[6, 0, 1, 2] = 27/2

    # N_8 = 9/2 * L_1*(3*L_3 - 1)*L_3 = 27/2*L_3^2*L_1 - 9/2 * L_3*L_1
    NLMatrix[7, 1, 0, 1] = -9/2
    NLMatrix[7, 1, 0, 2] = 27/2

    # N_9 = 9/2 * L_1*(3*L_1 - 1)*L_3 = 27/2*L_1^2*L_3 - 9/2 * L_3*L_1
    NLMatrix[8, 1, 0, 1] = -9/2
    NLMatrix[8, 2, 0, 1] = 27/2

    # internal node - 10
    # N_i = 27 * L_1*L_2*L_3
    NLMatrix[9, 1, 1, 1] = 27

    return NLMatrix, NodeLCoord