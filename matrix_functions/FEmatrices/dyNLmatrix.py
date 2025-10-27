import numpy as np

def dyNL_matrix(NLMatrix, TriProps):

    #   Part of the toolbox for solving problems of wave propagation
    # in arbitrary anisotropic inhomogeneous waveguides.
    # For details see User manual
    # and comments to the main script gen_aniso.m
    #
    #   dyNL_matrix function prepares the matrix, which indicates the expansion
    #  and the coefficients of the expansion of the derivatives
    #  with respect to y coordinate of the shape functions N_i
    #  into the intepolating functions L_j (L1, L2, L3) (times delta).
    #  The formulas for the expansion and the coefficients are obtained from the
    #  given in the book of Zienkiewicz (8.18), (8.36), (8.37).
    #  Current implementation is for cubic elements.
    # NB!!! This implementation is specific for FE computations
    #
    #   [T.Zharnikov, SMR v0.3_08.2014]
    #
    # function [dyNLMatrix] = dyNL_matrix(NLMatrix,TriProps)
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
    #   Code for dyNL_matrix
    #
    ################################################################################
    # ===============================================================================
    # Initialization
    # ===============================================================================

    # allocate memory for [dyNL_matrix] (expansion of the derivatives
    # of the shape functions into the interpolating functions

    # for cubic elements, there are 10 shape functions and the expansion of their
    # derivatives will be up to the 2nd power
    dyNLMatrix = np.zeros((10, 3, 3, 3))

    # coefficients of the derivatives of the intrepolating functions L_i
    # see the book by Zienkiewicz, see notation there as well
    # L_i = 1/(2*delta) * {a_i + b_i * x + c_i * y}
    # hence
    # d/dy L_i = 1/(2*delta) * c_i
    # the factor 1/delta is omitted for the simplification of computations
    dyL = 1 / 2 * TriProps.c

    # ===============================================================================
    # Compute the elements of the dyNLMatrix
    # the expansion is of the form: d/dy N_i = d/dy NLMatrix(i,j,k,l)
    # d/dy N_i = 1/delta * sum_{j,k,l = 1 .. N } dyNLMatrix(i,j,k,l) L1^{j-1} L2^{k-1} L3^{l-1}
    # 1/delta factor is omitted,
    # since the terms in the N_i expansion are of the form
    # NLMatrix(i,j,k,l) L1^{j-1} L2^{k-1} L3^{l-1}
    # they become upon differentiation ->
    # dyNLMatrix(i,j,k,l) = d/dyL(1)*j*NLMatrix(i,j+1,k,l) +
    # d/dyL(2)*k*NLMatrix(i,j,k+1,l) + d/dyL(3)*j*NLMatrix(i,j,k,l+1)
    # ===============================================================================

    # define dyNLMatrix for the cubic element
    # there are 10 shape functions

    # process the shape functions for all of the nodes
    for nN in range(0,10):
        for ii in range(0,3):
            for jj in range(0,3):
                for kk in range(0,3):
                    dyNLMatrix[nN, ii, jj, kk] = dyL[0] * ii * NLMatrix[nN, (ii + 1), jj, kk] + dyL[1] * jj * NLMatrix[nN, ii, (jj + 1), kk] + dyL[2] * kk * NLMatrix[nN, ii, jj, (kk + 1)]

    return dyNLMatrix