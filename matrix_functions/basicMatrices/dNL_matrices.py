import numpy as np
from matrix_functions import basicMatrices

def dNL_matrices(BasicMatrices: basicMatrices):
    """
    Part of the toolbox for solving problems of wave propagation
    in arbitrary anisotropic inhomogeneous waveguides.
    For details see User manual
    and comments to the main script gen_aniso.m

    dNL_matrices function prepares the matrices, which are necessary to construct
    the coefficients of the expansion of the derivatives
    with respect to x or y coordinate of the shape functions N_i
    into the intepolating functions L_j (L1, L2, L3) (times delta).
    The formulas for the expansion and the coefficients are obtained from the
    given in the book of Zienkiewicz (8.18), (8.36), (8.37).
    Current implementation is for cubic elements.
    NB!!! This implementation is specific for FE computations

    [T.Zharnikov, SMR v0.3_08.2014]

    function [dxNLMatrix] = dxNL_matrix(NLMatrix,TriProps)

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
    #   Code for dNL_matrix
    #
    ################################################################################
    # ===============================================================================
    # Explanation of the dNLMatrices
    # ===============================================================================

    # coefficients of the derivatives of the intrepolating functions L_i
    # see the book by Zienkiewicz, see notation there as well
    # L_i = 1/(2*delta) * {a_i + b_i * x + c_i * y}
    # hence
    # d/dx L_i = 1/(2*delta) * b_i
    # d/dy L_i = 1/(2*delta) * c_i
    # the factor 1/delta is omitted for the simplification of computations
    # dxL = 1/2*TriProps.b;
    # dyL = 1/2*TriProps.c;

    # Compute the elements of the dNLMatrix
    # the expansion is of the form: d/dx N_i = d/dx NLMatrix(i,j,k,l)
    # d/dx N_i = 1/delta * sum_{j,k,l = 1 .. N } dxNLMatrix(i,j,k,l) L1^{j-1} L2^{k-1} L3^{l-1}
    # 1/delta factor is omitted,
    # since the terms in the N_i expansion are of the form
    # NLMatrix(i,j,k,l) L1^{j-1} L2^{k-1} L3^{l-1}
    # they become upon differentiation ->
    # dxNLMatrix(i,j,k,l) = d/dxL(1)*j*NLMatrix(i,j+1,k,l) +
    # d/dxL(2)*k*NLMatrix(i,j,k+1,l) + d/dxL(3)*j*NLMatrix(i,j,k,l+1)
    # dyNLMatrix(i,j,k,l) = d/dyL(1)*j*NLMatrix(i,j+1,k,l) +
    # d/dyL(2)*k*NLMatrix(i,j,k+1,l) + d/dyL(3)*j*NLMatrix(i,j,k,l+1)
    # The analogous reasoning goes for the dyNLMatrix

    # According to the above, dNLMatrix is defined as
    # dxNLMatrix(i,j,k,l) = d/dxL(1)*dNLMatrix(1,i,j,k,l) +
    # d/dxL(2)*dNLMatrix(2,i,j,k,l) + d/dxL(3)*dNLMatrix(3,i,j,k,l)
    # dNLMatrix(1,i,j,k,l) = j*NLMatrix(i,j+1,k,l)
    # dNLMatrix(2,i,j,k,l) = k*NLMatrix(i,j,k+1,l)
    # dNLMatrix(3,i,j,k,l) = l*NLMatrix(i,j,k,l+1)

    # ===============================================================================
    # Initialization
    # ===============================================================================

    # allocate memory for [dNLMatrix] (required to construct the expansion
    # of the derivatives of the shape functions into the interpolating functions)

    # for cubic elements, there are 10 shape functions and the expansion of their
    # derivatives will be up to the 2nd power
    dNLMatrix = np.zeros((3, 10, 4, 4, 4))
    dNLMatrix_val = np.zeros((3, 10, 10))

    # ===============================================================================
    # Compute the elements of the dNLMatrix
    # ===============================================================================

    # define dNLMatrix for the cubic element
    # there are 10 shape functions

    # process the shape functions for all of the nodes
    for nN in range(10):
        dNLMatrix_val[0:3, nN] = 0
        for ii in range(3):
            for jj in range(3):
                for kk in range(3):
                    # compute coefficients of expansion for dNL into the
                    # powers of L1^{ii} L2^{jj} L3^{kk}
                    # Note: индексы сдвинуты на -1 по сравнению с MATLAB
                    dNLMatrix[0, nN, ii, jj, kk] = (ii + 1) * BasicMatrices['NLMatrix'][nN, (ii + 1), jj, kk]
                    dNLMatrix[1, nN, ii, jj, kk] = (jj + 1) * BasicMatrices['NLMatrix'][nN, ii, (jj + 1), kk]
                    dNLMatrix[2, nN, ii, jj, kk] = (kk + 1) * BasicMatrices['NLMatrix'][nN, ii, jj, (kk + 1)]

                    # compute the values of coefficients dNL at the nodes
                    for nN2 in range(10):
                        dNLMatrix_val[0:3, nN, nN2] = dNLMatrix_val[0:3, nN, nN2] + \
                                                      dNLMatrix[0:3, nN, ii, jj, kk] * BasicMatrices['NodeLCoord'][
                                                          nN2, 0] ** ii \
                                                      * BasicMatrices['NodeLCoord'][nN2, 1] ** jj * \
                                                      BasicMatrices['NodeLCoord'][nN2, 2] ** kk

    return dNLMatrix, dNLMatrix_val

# ===============================================================================
# Knowing dxL(i) and dyL(i) vectors -
# dxL = 1/2*TriProps.b;
# dyL = 1/2*TriProps.c;
# It is possible to compute the elements of the dxNLMatrix as
# dxNLMatrix(nN,ii,jj,kk) = d/dxL(1)*dNLMatrix(1,nN,ii,jj,kk) +
# d/dxL(2)*dNLMatrix(2,nN,ii,jj,kk) + d/dxL(3)*dNLMatrix(3,nN,ii,jj,kk)
# ===============================================================================
# ===============================================================================
# The elements of the dxNLMatrix
# the expansion is of the form: d/dx N_i = d/dx NLMatrix(i,j,k,l)
# d/dx N_i = 1/delta * sum_{j,k,l = 1 .. N } dxNLMatrix(i,j,k,l) L1^{j-1} L2^{k-1} L3^{l-1}
# 1/delta factor is omitted,
# since the terms in the N_i expansion are of the form
# NLMatrix(i,j,k,l) L1^{j-1} L2^{k-1} L3^{l-1}
# they become upon differentiation ->
# dxNLMatrix(i,j,k,l) = d/dxL(1)*j*NLMatrix(i,j+1,k,l) +
# d/dxL(2)*k*NLMatrix(i,j,k+1,l) + d/dxL(3)*l*NLMatrix(i,j,k,l+1)
# ===============================================================================

# define dxNLMatrix for the cubic element
# there are 10 shape functions

# process the shape functions for all of the nodes
# for nN in range(10):
#     for ii in range(3):
#         for jj in range(3):
#             for kk in range(3):
#                 dxNLMatrix(nN,ii,jj,kk) = ...
#                     dxL(0)*ii*NLMatrix(nN,(ii + 1),jj,kk) + ...
#                     dxL(1)*jj*NLMatrix(nN,ii,(jj + 1),kk) + ...
#                     dxL(2)*kk*NLMatrix(nN,ii,jj,(kk + 1)) ;