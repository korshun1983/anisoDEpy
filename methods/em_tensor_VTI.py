def em_tensor_VTI(c_VTI):
    import numpy as np
    # This routine constructs elastic moduli tensor both in Voigt
    #   and full the 4th rank tensor notations from the vector of VTI parameters
    #
    # em_tensor_VTI M-file      
    #      em_tensor_VTI, by itself, creates elastic moduli tensor both in
    #      reduced Voigt notation c_ij and in full 4th rank tensor formulation c_ijkl
    #      using c11=c22, c13=c23, c33, c44=55, c66 as an input.
    #
    #      [Timur Zharnikov SMR v0.1 2011]
    #
    # function [c_ij c_ijkl] = em_tensor_VTI(c_VTI)
    #
    #  Inputs -
    #       c_VTI - 1 by 5 array (vector) of VTI parameters, which
    #               are conviniently chosen as [c11 c13 c33 c44 c66]. 
    #               In GPa.
    #
    #  Outputs -
    #       c_ij - 6 by 6 elastic moduli tensor in Voigt notation in GPa
    #       c_ijkl - 3x3x3x3 elastic moduli tensor for full formulation 
    #       (as in elastic theory) in GPa
    #
    #  M-files required-
    #
    # Last Modified by Timur Zharnikov SMR v0.1 2011.10.28

    ################################################################################
    # Begin initialization code - DO NOT EDIT
    # End initialization code - DO NOT EDIT
    ################################################################################

    ################################################################################
    #
    #   Code for em_tensor_VTI
    #
    ################################################################################
    # ===============================================================================
    # Initialization
    # ===============================================================================

    # -------------------------------------------------------------------------------
    # populating elastic moduli tensors
    # -------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------
    # Voigt notation (c_IJ)
    # -------------------------------------------------------------------------------

    c_ij = np.zeros((6, 6), dtype = float)
    c_ij[1, 1] = c_VTI[1]
    c_ij[2, 2] = c_VTI[1]
    c_ij[3, 3] = c_VTI[3]

    c_ij[4, 4] = c_VTI[4]
    c_ij[5, 5] = c_VTI[4]
    c_ij[6, 6] = c_VTI[5]

    c_ij[1, 2] = c_ij(1, 1) - 2 * c_ij(6, 6)
    c_ij[1, 3] = c_VTI(2)
    c_ij[2, 3] = c_VTI(2)

    c_ij[2, 1] = c_ij[1, 2]
    c_ij[3, 1] = c_ij[1, 3]
    c_ij[3, 2] = c_ij[2, 3]

    # -------------------------------------------------------------------------------
    # Full formulation notation (c_ijkl)
    # -------------------------------------------------------------------------------

    c_ijkl = np.zeros((3, 3, 3, 3), dtype = float)
    c_ijkl[1, 1, 1, 1] = c_ij[1, 1]
    c_ijkl[2, 2, 2, 2] = c_ij[2, 2]
    c_ijkl[3, 3, 3, 3] = c_ij[3, 3]

    c_ijkl[1, 1, 2, 2] = c_ij[1, 2]
    c_ijkl[2, 2, 1, 1] = c_ij[2, 1]
    c_ijkl[1, 1, 3, 3] = c_ij[1, 3]
    c_ijkl[3, 3, 1, 1] = c_ij[3, 1]
    c_ijkl[2, 2, 3, 3] = c_ij[2, 3]
    c_ijkl[3, 3, 2, 2] = c_ij[3, 2]

    c_ijkl[3, 2, 3, 2] = c_ij[4, 4]
    c_ijkl[3, 2, 2, 3] = c_ij[4, 4]
    c_ijkl[2, 3, 3, 2] = c_ij[4, 4]
    c_ijkl[2, 3, 2, 3] = c_ij[4, 4]

    c_ijkl[3, 1, 3, 1] = c_ij[5, 5]
    c_ijkl[3, 1, 1, 3] = c_ij[5, 5]
    c_ijkl[1, 3, 3, 1] = c_ij[5, 5]
    c_ijkl[1, 3, 1, 3] = c_ij[5, 5]

    c_ijkl[1, 2, 1, 2] = c_ij[6, 6]
    c_ijkl[1, 2, 2, 1] = c_ij[6, 6]
    c_ijkl[2, 1, 1, 2] = c_ij[6, 6]
    c_ijkl[2, 1, 2, 1] = c_ij[6, 6]

    return c_ij, c_ijkl