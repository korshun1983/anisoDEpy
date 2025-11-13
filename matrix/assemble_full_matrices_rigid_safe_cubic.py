# File: assemble_full_matrices_rigid_safe_cubic.py
import numpy as np


def assemble_full_matrices_rigid_safe_cubic(comp_struct, basic_matrices, fem_matrices, full_matrices, ii_int, ii_d1,
                                            ii_d2):
    """
    Assemble full matrices for rigid boundary conditions (cubic elements).

    Part of the toolbox for solving problems of wave propagation
    in arbitrary anisotropic inhomogeneous waveguides.

    Rigid means that the corresponding displacements are zero (implies solid medium)

    Last Modified by Timur Zharnikov SMR v0.3_09.2014
    Converted to Python 2025

    Args:
        comp_struct: Structure containing model parameters (dict)
        basic_matrices: Structure containing basic matrices (dict)
        fem_matrices: FEM matrices structure (dict)
        full_matrices: Full matrices structure (dict)
        ii_int: Interface index
        ii_d1: First domain index (solid)
        ii_d2: Second domain index (not used for rigid boundary)

    Returns:
        fem_matrices: Updated FEM matrices structure (dict)
        full_matrices: Updated full matrices structure (dict)
    """

    # Find all nodes belonging to the outer boundary that should be removed
    # Rigid boundary: displacements are zero

    # Boundary elements in 2D formulation are edges
    boundary_edges_mask = np.isin(fem_matrices['BoundaryEdges'][2, :], ii_int)
    belements = fem_matrices['BoundaryEdges'][:2, boundary_edges_mask]
    nb_elements = belements.shape[1]

    b_nodes_full = np.array([], dtype=int)

    for ii_ed in range(nb_elements):
        # Find start and end nodes of the edge
        edge_nodes = np.array([
            belements[0, ii_ed],  # First node of boundary element (edge)
            belements[1, ii_ed]  # Second node of boundary element (edge)
        ])

        # Find elements containing these nodes
        fnodes_mask = np.isin(fem_matrices['DElements'][ii_d1][:10, :], edge_nodes[:2])
        nnodes = np.sum(fnodes_mask, axis=0)
        d1_edge_el = np.argmax(nnodes)

        # Read nodes belonging to the adjacent element
        tri_nodes = fem_matrices['DElements'][ii_d1][:10, d1_edge_el]

        # Find mid-side nodes for cubic interpolation
        edge_nodes_pos_d1 = np.zeros(4, dtype=int)
        edge_nodes_pos_d1[0] = np.where(np.isin(tri_nodes, edge_nodes[0]))[0][0]
        edge_nodes_pos_d1[1] = np.where(np.isin(tri_nodes, edge_nodes[1]))[0][0]

        # Find interior nodes of the edge element
        if edge_nodes_pos_d1[1] > (edge_nodes_pos_d1[0] % 3):
            edge_nodes_pos_d1[2] = (edge_nodes_pos_d1[0] + 1) * 2
            edge_nodes_pos_d1[3] = (edge_nodes_pos_d1[0] + 1) * 2 + 1
        else:
            edge_nodes_pos_d1[2] = (edge_nodes_pos_d1[1] + 1) * 2 + 1
            edge_nodes_pos_d1[3] = (edge_nodes_pos_d1[1] + 1) * 2

        # Add the mid-side nodes
        edge_nodes = np.concatenate([
            edge_nodes,
            [tri_nodes[edge_nodes_pos_d1[2]], tri_nodes[edge_nodes_pos_d1[3]]]
        ])

        b_nodes_full = np.concatenate([b_nodes_full, edge_nodes])

    # Identify nodes to remove from computation
    fem_matrices['BNodesFull'][ii_int] = np.unique(b_nodes_full)

    # Initialize DNodesRem if it doesn't exist
    if 'DNodesRem' not in fem_matrices:
        fem_matrices['DNodesRem'] = [np.array([]) for _ in range(10)]

    # Extend DNodesRem if needed
    if ii_d1 >= len(fem_matrices['DNodesRem']):
        fem_matrices['DNodesRem'].extend([np.array([]) for _ in range(ii_d1 + 1 - len(fem_matrices['DNodesRem']))])

    fem_matrices['DNodesRem'][ii_d1] = np.unique(np.concatenate([
        fem_matrices['DNodesRem'][ii_d1],
        b_nodes_full
    ]))

    remove_nodes_pos = np.isin(fem_matrices['DNodes'][ii_d1], fem_matrices['DNodesRem'][ii_d1])
    fem_matrices['DNodesComp'][ii_d1] = fem_matrices['DNodes'][ii_d1][~remove_nodes_pos]

    # Find positions of rows corresponding to nodes that will be removed
    # NB! Nodes in DNodes and BNodesFull are ordered!
    # This is an important condition for this script to work.

    b_nodes_full_size = len(fem_matrices['BNodesFull'][ii_int])
    remove_var_pos = []

    for ii_n in range(b_nodes_full_size):
        d1_node_pos = np.where(fem_matrices['DNodes'][ii_d1] == fem_matrices['BNodesFull'][ii_int][ii_n])[0]

        if len(d1_node_pos) > 0:
            d1_node_pos = d1_node_pos[0]
            dvar_num_1 = comp_struct['Data']['DVarNum'][ii_d1]

            # Calculate variable positions (0-indexed)
            start_idx = basic_matrices['Pos'][ii_d1][0] - 1  # Convert to 0-indexed
            var_range = np.arange(dvar_num_1 * d1_node_pos, dvar_num_1 * (d1_node_pos + 1))
            remove_var_pos.extend((start_idx + var_range).tolist())

    # Initialize DZeroVarPos if it doesn't exist
    if 'DZeroVarPos' not in fem_matrices:
        fem_matrices['DZeroVarPos'] = [np.array([]) for _ in range(10)]

    # Extend DZeroVarPos if needed
    if ii_d1 >= len(fem_matrices['DZeroVarPos']):
        fem_matrices['DZeroVarPos'].extend([np.array([]) for _ in range(ii_d1 + 1 - len(fem_matrices['DZeroVarPos']))])

    fem_matrices['DZeroVarPos'][ii_d1] = np.unique(np.concatenate([
        fem_matrices['DZeroVarPos'][ii_d1],
        np.array(remove_var_pos)
    ]))

    return fem_matrices, full_matrices