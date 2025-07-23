import math

def rot_matrix(theta, phi):
    # This routine prepares rotation matrix
    #
    # rot_matrix M-file
    #      rot_matrix, by itself, computes rotation matrix in standard notation
    #      into new coordinate system according to the standard rule (see, e.g. Auld).
    #      It uses theta and phi as an input.
    #
    #      [Timur Zharnikov SMR v0.1 2011]
    #
    #function [a_rot] = rot_matrix(theta, phi)
    #
    #  Inputs -
    #       theta - inclination (relative dip) angle theta. In radians.
    #       phi   - rotation angle phi. In radians.
    #
    #  Outputs -
    #       a_rot - rotation matrix. Non-dim.
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
    #   Code for rot_matrix
    #
    ################################################################################
    #===============================================================================
    # Initialization
    #===============================================================================

    # if nargin < 1
    #     theta = 0;
    #     phi = 0;
    # end
    #
    # if nargin == 1
    #     phi = 0;
    # end

    #-------------------------------------------------------------------------------
    # Calculating rotation matrix a_rot
    # This computation uses Euler angles, and is the same as used by B.Sinha.
    # It assumes that first the system is rotated to angle phi
    # and then inclined to angle theta
    #
    # This formulation corresponds to TTI symmetry plane at theta = pi/2
    #-------------------------------------------------------------------------------

    a_rot = [ [math.cos(phi),                math.sin(phi),                0],
              [-math.cos(theta)*math.sin(phi),    math.cos(theta)*math.cos(phi),     math.sin(theta)],
              [math.sin(theta)*math.sin(phi),     -math.sin(theta)*math.cos(phi),    math.cos(theta)]]

    #-------------------------------------------------------------------------------
    # Calculating rotation matrix a_rot
    # This computation is different from B.Sinha.
    # It assumes that first the system is inclined to angle theta
    # and then rotated to angle phi
    #
    # This formulation corresponds to TTI symmetry plane at theta set by user
    # (gives the ability to rotate symmetry plane) - need to double check...
    #-------------------------------------------------------------------------------

    # a_rot = [ cos(phi)                cos(theta)*sin(phi)     sin(theta)*sin(phi);...
    #          -sin(phi)               cos(theta)*cos(phi)     sin(theta)*cos(phi);...
    #          0                       -sin(theta)             cos(theta)];

    #-------------------------------------------------------------------------------
    # Calculating rotation matrix a_rot
    # This computation is according to Vershinin's formulation (phi <-> beta, theta <-> alpha).
    # It assumes that first the system is inclined to angle theta
    # and then rotated to angle phi
    #-------------------------------------------------------------------------------

    # a_rot = [ cos(theta)*cos(phi)    -sin(phi)     -sin(theta)*cos(phi);...
    #           cos(theta)*sin(phi)     cos(phi)     -sin(theta)*sin(phi);...
    #           sin(theta)              0             cos(theta)];
    return a_rot