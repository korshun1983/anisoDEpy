import numpy as np

def rot_c_ij(c_ij, rot_m):
    # This routine takes elastic moduli tensor in Voigt notation
    #   and rotates it using rotation matrix
    #
    # rot_c_ij M-file
    #      rot_c_ij, by itself, rotates elastic moduli tensor in Voigt notation
    #      into new coordinate system using transformation matrix rot_ij
    #      according to the standard rule (see, e.g. Auld).
    #      It uses c_ij and rot_m as an input.
    #
    #      [Timur Zharnikov SMR v0.1 2011]
    #
    # function [c_ij_rot] = rot_c_ij(c_ij, rot_m)
    #
    #  Inputs -
    #       c_ij - elastic moduli tensor. In GPa.
    #       rot_m - rotation matrix. Non-dim.
    #
    #  Outputs -
    #       c_ij_rot - rotated elastic moduli tensor. In GPa.
    #                   direct rotation
    #           (used to go from reference system (North) to the borehole system).
    #
    #       c_ij_rot_back - rotated elastic moduli tensor. In GPa.
    #                   rotation in the reverse direction.
    #           (used to go from formation system to reference system (North)).
    #
    #  M-files required-
    #
    # Last Modified by Timur Zharnikov SMR v0.1 2011.11.03

    ################################################################################
    # Begin initialization code - DO NOT EDIT
    # End initialization code - DO NOT EDIT
    ################################################################################

    ################################################################################
    #
    #   Code for rot_c_ij
    #
    ################################################################################
    # ===============================================================================
    # Initialization
    # ===============================================================================

    # -------------------------------------------------------------------------------
    # Calculating rotated c_ij_rot
    # -------------------------------------------------------------------------------

    a = rot_m

    Mincl = np.array([a[1, 1] ^ 2, a[1, 2] ^ 2, a[1, 3] ^ 2, 2 * a[1, 2] * a[1, 3], 2 * a[1, 1] * a[1, 3], 2 * a[1, 1] * a[1, 2],
    a[2, 1] ^ 2, a[2, 2] ^ 2, a[2, 3] ^ 2, 2 * a[2, 2] * a[2, 3], 2 * a[2, 1] * a[2, 3], 2 * a[2, 1] * a[2, 2],
             a[3, 1] ^ 2, a[3, 2] ^ 2, a[3, 3] ^ 2, 2 * a[3, 2] * a[3, 3], 2 * a[3, 1] * a[3, 3], 2 * a[3, 1] * a[3, 2],
             a[2, 1] * a[3, 1], a[2, 2] * a[3, 2], a[2, 3] * a[3, 3], a[2, 2] * a[3, 3] + a[2, 3] * a[3, 2],
             a[2, 1] * a[3, 3] + a[2, 3] * a[3, 1], a[2, 1] * a[3, 2] + a[2, 2] * a[3, 1],
             a[1, 1] * a[3, 1], a[1, 2] * a[3, 2], a[1, 3] * a[3, 3], a[1, 2] * a[3, 3] + a[1, 3] * a[3, 2],
             a[1, 1] * a[3, 3] + a[1, 3] * a[3, 1], a[1, 1] * a[3, 2] + a[1, 2] * a[3, 1],
             a[1, 1] * a[2, 1], a[1, 2] * a[2, 2], a[1, 3] * a[2, 3], a[1, 2] * a[2, 3] + a[1, 3] * a[2, 2],
             a[1, 1] * a[2, 3] + a[1, 3] * a[2, 1], a[1, 1] * a[2, 2] + a[1, 2] * a[2, 1]])

    Nincl = np.array([a[1, 1] ^ 2, a[1, 2] ^ 2, a[1, 3] ^ 2, a[1, 2] * a[1, 3], a[1, 1] * a[1, 3], a[1, 1] * a[1, 2],
    a[2, 1] ^ 2, a[2, 2] ^ 2, a[2, 3] ^ 2, a[2, 2] * a[2, 3], a[2, 1] * a[2, 3], a[2, 1] * a[2, 2],
             a[3, 1] ^ 2, a[3, 2] ^ 2, a[3, 3] ^ 2, a[3, 2] * a[3, 3], a[3, 1] * a[3, 3], a[3, 1] * a[3, 2],
             2 * a[2, 1] * a[3, 1], 2 * a[2, 2] * a[3, 2], 2 * a[2, 3] * a[3, 3], a[2, 2] * a[3, 3] + a[2, 3] * a[3, 2],
             a[2, 1] * a[3, 3] + a[2, 3] * a[3, 1], a[2, 1] * a[3, 2] + a[2, 2] * a[3, 1],
             2 * a[1, 1] * a[3, 1], 2 * a[1, 2] * a[3, 2], 2 * a[1, 3] * a[3, 3], a[1, 2] * a[3, 3] + a[1, 3] * a[3, 2],
             a[1, 1] * a[3, 3] + a[1, 3] * a[3, 1], a[1, 1] * a[3, 2] + a[1, 2] * a[3, 1],
             2 * a[1, 1] * a[2, 1], 2 * a[1, 2] * a[2, 2], 2 * a[1, 3] * a[2, 3], a[1, 2] * a[2, 3] + a[1, 3] * a[2, 2],
             a[1, 1] * a[2, 3] + a[1, 3] * a[2, 1], a[1, 1] * a[2, 2] + a[1, 2] * a[2, 1]])

    c_ij_rot = np.dot(np.dot(Mincl, c_ij), Mincl.transpose())
    c_ij_rot_back = ((Nincl.transpose())*c_ij)*Nincl

    return c_ij_rot, c_ij_rot_back