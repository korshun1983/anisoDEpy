import numpy as np

def L1L2_int_matrix(N_degree):
    #   Part of the toolbox for solving problems of wave propagation
    # in arbitrary anisotropic inhomogeneous waveguides.
    # For details see User manual
    # and comments to the main script gen_aniso.m
    #
    #   L1L2_int_matrix function computes the integrals of the form
    #  int {_(x1,y1)^{x2,y2} } L1^{a}L2^{b} ds using the exact integration formula:
    #  int {_(x1,y1)^{x2,y2} } L1^{a}L2^{b} = L* int {_(0)^{1} } ( 1 - xi )^{a} xi^{b}
    #
    #  J_ab = int {_(0)^{1} } ( 1 - xi )^{a} xi^{b}
    #  This integral can be evaluated by using the formulas on pp. 298, 952,
    #  and 964 of the Gradshteyn, Ryzhik reference book on the integrals and
    #  special functions. As the result,
    #  J_ab = a!b!/(a+b+1)!
    #  The factor L (the length of the edge) is omitted.
    #  This function computes the matrix of coefficients a!b!/(a+b+1)!
    # NB!!! This implementation is specific for FE computations
    #
    #   [T.Zharnikov, SMR v0.3_08.2014]
    #
    # function [LEdgeIntMatrix] = L1L2_int_matrix(N_degree)
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
    #   Code for L1L2_int_matrix
    #
    ################################################################################
    # ===============================================================================
    # Initialization
    # ===============================================================================

    # allocate memory for LEdgeIntMatrix (matrix of coefficients to compute
    # intregrals of the form int int {over the edge (element of the boundary)}
    # L1^{a}L2^{b} )
    LEdgeIntMatrix = np.zeros((N_degree, N_degree))

    # ===============================================================================
    # Compute the elements of the matrix using the exact formula given above
    # (in the introductory area of this file): (a!b!/(a+b+1)!)
    # ===============================================================================

    for aa in range(1, N_degree + 1):
        for bb in range(aa, N_degree + 1):
            LEdgeIntMatrix[aa - 1, bb - 1] = (np.math.factorial(aa - 1) /
                                              np.prod(np.arange((bb - 1) + 1, (aa - 1) + (bb - 1) + 1 + 1)))
            LEdgeIntMatrix[bb - 1, aa - 1] = LEdgeIntMatrix[aa - 1, bb - 1]

    return LEdgeIntMatrix