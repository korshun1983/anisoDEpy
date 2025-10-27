import numpy as np

def NLEdge_matrix():
    #   Part of the toolbox for solving problems of wave propagation
    # in arbitrary anisotropic inhomogeneous waveguides.
    # For details see User manual
    # and comments to the main script gen_aniso.m
    #
    #   NLEdge_matrix function prepares the matrix, which indicates the expansion
    #  and the coefficients of the expansion of the shape functions N_i
    #  into the intepolating functions L_j (L1, L2).
    #  For the edge of the cubic element.
    #  The formulas for essentially are the constraint of the N_i shape functions
    #  for the element to the element boundary (edge).
    #  I.e. shape functions N1, N2, N3, N4 for the edge are the shape functions
    #  N1, N2, N4, N5 for the element.
    #  Current implementation is for cubic elements.
    # NB!!! This implementation is specific for FE computations
    #
    #   [T.Zharnikov, SMR v0.3_08.2014]
    #
    # function [NLEdgeMatrix] = NLEdge_matrix()
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
    #   Code for NLEdge_matrix
    #
    ################################################################################
    #===============================================================================
    # Initialization
    #===============================================================================

    # allocate memory for [NLEdgeMatrix] (expansion of shape functions into the
    # interpolating functions

    # for the edge of the cubic element, there are 4 shape functions and the expansion is up
    # to the 3rd power
    NLEdgeMatrix = np.zeros((4, 4, 4))

    #===============================================================================
    # Compute the elements of the NLEdgeMatrix
    # the expansion is of the form: N_i = NLEdgeMatrix(i,j,k,l)
    # N_i = sum_{j,k,l = 1 .. N } NLEdgeMatrix(i,j,k,l) L1^{j-1} L2^{k-1}
    #===============================================================================

    # define NLEdgeMatrix for the 1-2 edge of the cubic element
    # there are 4 shape functions

    # corner nodes - 1, 2  (correspond to the corner nodes 1, 2 of the element)
    # N_i = 1/2 * L_i*(3*L_i - 1)*(3*L_i - 2 ) = 9/2*L_i^3 - 9/2 * L_i^2 + L_i

    NLEdgeMatrix[0, 1, 0] = 1
    NLEdgeMatrix[0, 2, 0] = -9/2
    NLEdgeMatrix[0, 3, 0] = 9/2

    NLEdgeMatrix[1, 0, 1] = 1
    NLEdgeMatrix[1, 0, 2] = -9/2
    NLEdgeMatrix[1, 0, 3] = 9/2

    # mid-side (edge) nodes - 3, 4 (correspond to the mid-side nodes 4, 5 of the element)

    # N_4 = 9/2 * L_1*(3*L_1 - 1)*L_2 = 27/2*L_1^2*L_2 - 9/2 * L_1*L_2
    NLEdgeMatrix[2, 1, 1] = -9/2
    NLEdgeMatrix[2, 2, 1] = 27/2

    # N_5 = 9/2 * L_1*(3*L_2 - 1)*L_2 = 27/2*L_2^2*L_1 - 9/2 * L_1*L_2
    NLEdgeMatrix[3, 1, 1] = -9/2
    NLEdgeMatrix[3, 1, 2] = 27/2

    return NLEdgeMatrix