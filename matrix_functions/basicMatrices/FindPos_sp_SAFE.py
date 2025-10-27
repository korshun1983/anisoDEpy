from scipy.sparse import csr_matrix, save_npz, load_npz
import os
from matrix_functions.matrix_definitions import ElMatrices
from matrix_functions import basicMatrices


# Файл: FindPos_sp_SAFE.py
def FindPos_sp_SAFE(CompStruct, FEMatrices, BasicMatrices: basicMatrices):
    #   Part of the toolbox for solving problems of wave propagation
    # in arbitrary anisotropic inhomogeneous waveguides.
    # For details see User manual
    # and comments to the main script gen_aniso.m
    #
    #   FindPos_sp_SAFE function computes positions of blocks
    # corresponding to the domains, etc.
    # in the full 2D SAFE matrix
    # (before reductions due to the interface and the boundary conditions).
    # Knowledge of these positions is required to accurately
    # put the blocks from cell arrays and various boundary and interface conditions.
    # NB!!! This implementation is specific for spectrum computation with SAFE.
    #
    #   [T.Zharnikov, SMR v0.3_09.2014]
    #
    # function [Pos] = FindPos_sp_SAFE(CompStruct,BasicMatrices)
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
    # Last Modified by Timur Zharnikov SMR v0.3_09.2014

    ################################################################################
    #
    #   Code for FindPos_sp_SAFE
    #
    ################################################################################
    # ===============================================================================
    # Initialization
    # ===============================================================================

    # ===============================================================================
    # Calculating indices, which describe start positions of various blocks of matrices
    # ===============================================================================

    # Start position for blocks in large 2D matrix
    Pos_end = 0
    NodePos_end = 0

    # index for the domain number
    for ii_d in range(CompStruct['Data']['N_domain']):
        Pos_beg = Pos_end + 1
        NodePos_beg = NodePos_end + 1
        NumNodes = FEMatrices['DNodes'][ii_d].shape
        Pos_end = Pos_beg + CompStruct['Data']['DVarNum'][ii_d] * NumNodes[1] - 1
        NodePos_end = NodePos_beg + NumNodes[1] - 1
        BasicMatrices['Pos'][ii_d] = [Pos_beg, Pos_end]
        BasicMatrices['NodePos'][ii_d] = [NodePos_beg, NodePos_end]

    return BasicMatrices