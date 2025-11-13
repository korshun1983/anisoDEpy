# File 4: find_pos_sp_safe.py
import numpy as np


def find_pos_sp_safe(comp_struct, fe_matrices, basic_matrices):
    """
    Find positions of blocks in the full 2D SAFE matrix

    Inputs:
        comp_struct : dict
            Dictionary containing model parameters
        fe_matrices : dict
            Dictionary containing FE matrices
        basic_matrices : dict
            Dictionary containing basic matrices

    Outputs:
        basic_matrices : dict
            Updated with position information
    """
    # Initialize position arrays if they don't exist
    basic_matrices.setdefault('Pos', [None] * comp_struct['Data']['N_domain'])
    basic_matrices.setdefault('NodePos', [None] * comp_struct['Data']['N_domain'])

    # Start positions (0-indexed for Python)
    pos_end = -1
    node_pos_end = -1

    # Loop over domains
    for ii_d in range(comp_struct['Data']['N_domain']):
        pos_beg = pos_end + 1
        node_pos_beg = node_pos_end + 1

        # Number of nodes in this domain
        num_nodes = len(fe_matrices['DNodes'][ii_d])

        # Calculate end positions (inclusive)
        pos_end = pos_beg + comp_struct['Data']['DVarNum'][ii_d] * num_nodes - 1
        node_pos_end = node_pos_beg + num_nodes - 1

        # Store positions
        basic_matrices['Pos'][ii_d] = [pos_beg, pos_end]
        basic_matrices['NodePos'][ii_d] = [node_pos_beg, node_pos_end]

    return basic_matrices