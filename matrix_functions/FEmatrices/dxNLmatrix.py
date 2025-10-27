import numpy as np

def dxNL_matrix(NLMatrix, TriProps):

    #   Part of the toolbox for solving problems of wave propagation
    # in arbitrary anisotropic inhomogeneous waveguides.
    # For details see User manual
    # and comments to the main script gen_aniso.m
    #
    #   dxNL_matrix function prepares the matrix, which indicates the expansion
    #  and the coefficients of the expansion of the derivatives
    #  with respect to x coordinate of the shape functions N_i
    #  into the intepolating functions L_j (L1, L2, L3) (times delta).
    #  The formulas for the expansion and the coefficients are obtained from the
    #  given in the book of Zienkiewicz (8.18), (8.36), (8.37).
    #  Current implementation is for cubic elements.
    # NB!!! This implementation is specific for FE computations
    #
    #   [T.Zharnikov, SMR v0.3_08.2014]
    #
    # function [dxNLMatrix] = dxNL_matrix(NLMatrix,TriProps)
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
    #   Code for dxNL_matrix
    #
    ################################################################################
    # ===============================================================================
    # Initialization
    # ===============================================================================

    # allocate memory for [dxNLMatrix] (expansion of the derivatives
    # of the shape functions into the interpolating functions

    # for cubic elements, there are 10 shape functions and the expansion of their
    # derivatives will be up to the 2nd power
    dxNLMatrix = np.zeros((10, 3, 3, 3))

    # coefficients of the derivatives of the intrepolating functions L_i
    # see the book by Zienkiewicz, see notation there as well
    # L_i = 1/(2*delta) * {a_i + b_i * x + c_i * y}
    # hence
    # d/dx L_i = 1/(2*delta) * b_i
    # the factor 1/delta is omitted for the simplification of computations
    dxL = 1 / 2 * TriProps.b

    # ===============================================================================
    # Compute the elements of the dxNLMatrix
    # the expansion is of the form: d/dx N_i = d/dx NLMatrix(i,j,k,l)
    # d/dx N_i = 1/delta * sum_{j,k,l = 1 .. N } dxNLMatrix(i,j,k,l) L1^{j-1} L2^{k-1} L3^{l-1}
    # 1/delta factor is omitted,
    # since the terms in the N_i expansion are of the form
    # NLMatrix(i,j,k,l) L1^{j-1} L2^{k-1} L3^{l-1}
    # they become upon differentiation ->
    # dxNLMatrix(i,j,k,l) = d/dxL(1)*j*NLMatrix(i,j+1,k,l) +
    # d/dxL(2)*k*NLMatrix(i,j,k+1,l) + d/dxL(3)*j*NLMatrix(i,j,k,l+1)
    # ===============================================================================

    # define dxNLMatrix for the cubic element
    # there are 10 shape functions

    # process the shape functions for all of the nodes
    for nN in range(0,10):
        for ii in range(0,3):
            for jj in range(0,3):
                for kk in range(0,3):
                    dxNLMatrix[nN, ii, jj, kk] = dxL[0] * ii * NLMatrix[nN, (ii + 1), jj, kk] + dxL[1] * jj * NLMatrix[nN, ii, (jj + 1), kk] + dxL[2] * kk * NLMatrix[nN, ii, jj, (kk + 1)]

    return dxNLMatrix