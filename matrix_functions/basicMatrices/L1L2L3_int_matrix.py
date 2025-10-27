import numpy as np

def L1L2L3_int_matrix(N_degree):
    #   Part of the toolbox for solving problems of wave propagation
    # in arbitrary anisotropic inhomogeneous waveguides.
    # For details see User manual
    # and comments to the main script gen_aniso.m
    #
    #   L1L2L3_int_matrix function computes the integrals of the form
    #  int int {over triangle} L1^{a}L2^{b}L3^{c} dx dy using the exact integration
    #  formula from the book of Zienkiewicz (8.38):
    #  (a!b!c!/(a+b+c+2)!)*2*delta, where delta is the total area of the triangle
    #  This function computes the matrix of coefficients 2*(a!b!c!/(a+b+c+2)!)
    # NB!!! This implementation is specific for FE computations
    #
    #   [T.Zharnikov, SMR v0.3_08.2014]
    #
    # function [LIntMatrix] = L1L2L3_int_matrix()
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
    #   Code for L1L2L3_int_matrix
    #
    ################################################################################
    # ===============================================================================
    # Initialization
    # ===============================================================================

    # allocate memory for LIntMatrix (matrix of coefficients to compute
    # intregrals of the form int int {over triangle} L1^{a}L2^{b}L3^{c} )
    LIntMatrix = np.zeros((N_degree, N_degree, N_degree))

    # ===============================================================================
    # Compute the elements of the matrix using the exact formula from the book
    # of Zienkiewicz (8.38): 2*(a!b!c!/(a+b+c+2)!)
    # ===============================================================================

    for aa in range(1, N_degree + 1):
        for bb in range(aa, N_degree + 1):
            for cc in range(bb, N_degree + 1):
                value = (2 * np.math.factorial(aa - 1) * np.math.factorial(bb - 1) /
                         np.prod(np.arange((cc - 1) + 1, (aa - 1) + (bb - 1) + (cc - 1) + 2 + 1)))

                LIntMatrix[aa - 1, bb - 1, cc - 1] = value
                LIntMatrix[aa - 1, cc - 1, bb - 1] = value
                LIntMatrix[bb - 1, aa - 1, cc - 1] = value
                LIntMatrix[bb - 1, cc - 1, aa - 1] = value
                LIntMatrix[cc - 1, aa - 1, bb - 1] = value
                LIntMatrix[cc - 1, bb - 1, aa - 1] = value

    return LIntMatrix