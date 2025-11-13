import numpy as np


def ic_matrices_ff_ss_safe_cubic(comp_struct, basic_matrices, fem_matrices, ii_int, ii_d1, ii_d2):
    """
    Part of the toolbox for solving problems of wave propagation
    in arbitrary anisotropic inhomogeneous waveguides.

    Prepares interface matrices for fluid-solid boundary (ff_ss).
    [T.Zharnikov SMR v0.3_09.2014]
    Converted to Python 2025
    """

    # Get boundary edges for this interface
    boundary_edges_mask = np.isin(fem_matrices['BoundaryEdges'][2, :], ii_int)
    belements = fem_matrices['BoundaryEdges'][:2, boundary_edges_mask]
    nb_elements = belements.shape[1]

    b_nodes_full = np.array([], dtype=int)

    # Process all edges belonging to the boundary
    for ii_ed in range(nb_elements):
        # Find start and end nodes of the edge
        edge_nodes = np.array([
            belements[0, ii_ed],
            belements[1, ii_ed]
        ])

        # Find elements containing these nodes
        fnodes = np.isin(fem_matrices['DElements'][ii_d1][:10, :], edge_nodes[:2])
        nnodes = np.sum(fnodes, axis=0)
        d1_edge_el = np.argmax(nnodes)

        # Read nodes belonging to adjacent element
        tri_nodes = fem_matrices['DElements'][ii_d1][:10, d1_edge_el]

        # Find mid-side nodes for cubic interpolation
        edge_nodes_pos_d1 = np.zeros(4, dtype=int)
        edge_nodes_pos_d1[0] = np.where(np.isin(tri_nodes, edge_nodes[0]))[0][0]
        edge_nodes_pos_d1[1] = np.where(np.isin(tri_nodes, edge_nodes[1]))[0][0]

        # Find interior nodes of edge element
        if edge_nodes_pos_d1[1] > (edge_nodes_pos_d1[0] % 3):
            edge_nodes_pos_d1[2] = (edge_nodes_pos_d1[0] + 1) * 2
            edge_nodes_pos_d1[3] = (edge_nodes_pos_d1[0] + 1) * 2 + 1
        else:
            edge_nodes_pos_d1[2] = (edge_nodes_pos_d1[1] + 1) * 2 + 1
            edge_nodes_pos_d1[3] = (edge_nodes_pos_d1[1] + 1) * 2

        edge_nodes = np.concatenate([
            edge_nodes,
            [tri_nodes[edge_nodes_pos_d1[2]], tri_nodes[edge_nodes_pos_d1[3]]]
        ])

        b_nodes_full = np.concatenate([b_nodes_full, edge_nodes])

    # Assemble full list of nodes belonging to interface
    fem_matrices['BNodesFull'][ii_int] = np.unique(b_nodes_full)

    return fem_matrices