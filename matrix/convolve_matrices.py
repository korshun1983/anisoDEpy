import numpy as np
import scipy.io
import scipy.signal
import os


def convolve_matrices(comp_struct, basic_matrices):
    """
    ConvolveMatrices function computes matrices for wave propagation in anisotropic inhomogeneous waveguides.

    This function computes expansion matrices and coefficients for double convolutions:
    conv(N,conv(N,N)), conv(dN,conv(N,N)), conv(N,conv(N,dN)), conv(dN,conv(N,dN))
    into interpolating functions L_j (L1, L2, L3).

    Args:
        comp_struct (dict): Structure containing model parameters with keys:
            - Config.root_path: Root directory path
            - Advanced.N_nodes: Number of nodes
            - Data.N_domain: Number of domains
            - Data.DVarNum: List/array of domain variable numbers
        basic_matrices (dict): Structure containing basic matrices with keys:
            - LIntMatrix9: Integration matrix (9x9x9)
            - NLMatrix: Node shape function matrix
            - dNLMatrix: Derivative of node shape function matrix

    Returns:
        dict: Modified basic_matrices dictionary with added convolution matrices

    Note:
        Current implementation is for cubic elements and is specific for FE computations.
        Original MATLAB version by T.Zharnikov, SMR v0.3_08.2014
    """

    # ===============================================================================
    # Initialization
    # ===============================================================================

    # Construct file path for precomputed matrices
    fname = os.path.join(comp_struct['Config']['root_path'], 'output', 'ConvMatrices.mat')

    # Check if precomputed matrices exist
    if os.path.exists(fname):
        # Load precomputed matrices from file
        mat_data = scipy.io.loadmat(fname)

        basic_matrices['NNNConvMatrixInt'] = mat_data['NNNConvMatrixInt']
        basic_matrices['dNNNConvMatrixInt'] = mat_data['dNNNConvMatrixInt']
        basic_matrices['NNdNConvMatrixInt'] = mat_data['NNdNConvMatrixInt']
        basic_matrices['dNNdNConvMatrixInt'] = mat_data['dNNdNConvMatrixInt']

    else:
        # ===============================================================================
        # Compute matrices from scratch
        # ===============================================================================

        # Identify number of nodes
        n_nodes = comp_struct['Advanced']['N_nodes']

        # Allocate memory for the matrices
        # The implementation is for cubic elements, hence expansion is up to the 3rd power in L_i

        nnn_conv_matrix = np.zeros((n_nodes, n_nodes, n_nodes, 10, 10, 10))
        dnnn_conv_matrix = np.zeros((3, n_nodes, n_nodes, n_nodes, 10, 10, 10))
        nn_dn_conv_matrix = np.zeros((3, n_nodes, n_nodes, n_nodes, 10, 10, 10))
        dnn_dn_conv_matrix = np.zeros((3, 3, n_nodes, n_nodes, n_nodes, 10, 10, 10))

        l_int_matrix = basic_matrices['LIntMatrix9'][:10, :10, :10]
        l_int_matrix_nnn = np.zeros_like(nnn_conv_matrix)
        l_int_matrix_nndn = np.zeros_like(nn_dn_conv_matrix)
        l_int_matrix_dnnn = np.zeros_like(nn_dn_conv_matrix)
        l_int_matrix_dnndn = np.zeros_like(dnn_dn_conv_matrix)

        # ===============================================================================
        # Compute the elements of all convolution matrices
        # and prepare large LIntMatrices for the integration
        # ===============================================================================

        print('Computing convolutions...')  # Status indicator

        for bb in range(n_nodes):
            for cc in range(n_nodes):
                # Extract Nb and Nc shape functions
                Nb = np.squeeze(basic_matrices['NLMatrix'][bb, :, :, :])
                Nc = np.squeeze(basic_matrices['NLMatrix'][cc, :, :, :])
                conv_nb_nc = scipy.signal.convolve(Nb, Nc, mode='full')

                for aa in range(n_nodes):
                    Na = np.squeeze(basic_matrices['NLMatrix'][aa, :, :, :])
                    nnn_conv_matrix[aa, bb, cc, :, :, :] = scipy.signal.convolve(Na, conv_nb_nc, mode='full')
                    l_int_matrix_nnn[aa, bb, cc, :, :, :] = l_int_matrix

                    for ii in range(3):
                        dna_i = np.squeeze(basic_matrices['dNLMatrix'][ii, aa, :, :, :])
                        dnnn_conv_matrix[ii, aa, bb, cc, :, :, :] = scipy.signal.convolve(dna_i, conv_nb_nc,
                                                                                          mode='full')
                        l_int_matrix_dnnn[ii, aa, bb, cc, :, :, :] = l_int_matrix

                for jj in range(3):
                    dnc_j = np.squeeze(basic_matrices['dNLMatrix'][jj, cc, :, :, :])
                    conv_nb_dnc_j = scipy.signal.convolve(Nb, dnc_j, mode='full')

                    for aa in range(n_nodes):
                        Na = np.squeeze(basic_matrices['NLMatrix'][aa, :, :, :])
                        nn_dn_conv_matrix[jj, aa, bb, cc, :, :, :] = scipy.signal.convolve(Na, conv_nb_dnc_j,
                                                                                           mode='full')
                        l_int_matrix_nndn[jj, aa, bb, cc, :, :, :] = l_int_matrix

                        for ii in range(3):
                            dna_i = np.squeeze(basic_matrices['dNLMatrix'][ii, aa, :, :, :])
                            dnn_dn_conv_matrix[ii, jj, aa, bb, cc, :, :, :] = scipy.signal.convolve(dna_i,
                                                                                                    conv_nb_dnc_j,
                                                                                                    mode='full')
                            l_int_matrix_dnndn[ii, jj, aa, bb, cc, :, :, :] = l_int_matrix

        # ===============================================================================
        # Compute the integrals over the standard triangle of the monoms
        # by multiplying elementwise convolution matrices on the large LIntMatrices
        # and summing over L1^i*L2^j*L3^k contributions
        # ===============================================================================

        print('Computing integrals...')  # Status indicator

        # NNN convolution integral: sum over last three dimensions
        nnn_conv_matrix_unsummed = nnn_conv_matrix * l_int_matrix_nnn
        basic_matrices['NNNConvMatrixInt'] = np.squeeze(
            np.sum(nnn_conv_matrix_unsummed, axis=(3, 4, 5))
        )

        # dNNN convolution integral
        dnnn_conv_matrix_unsummed = dnnn_conv_matrix * l_int_matrix_dnnn
        basic_matrices['dNNNConvMatrixInt'] = np.squeeze(
            np.sum(dnnn_conv_matrix_unsummed, axis=(4, 5, 6))
        )

        # NNdN convolution integral
        nn_dn_conv_matrix_unsummed = nn_dn_conv_matrix * l_int_matrix_nndn
        basic_matrices['NNdNConvMatrixInt'] = np.squeeze(
            np.sum(nn_dn_conv_matrix_unsummed, axis=(4, 5, 6))
        )

        # dNNdN convolution integral
        dnn_dn_conv_matrix_unsummed = dnn_dn_conv_matrix * l_int_matrix_dnndn
        basic_matrices['dNNdNConvMatrixInt'] = np.squeeze(
            np.sum(dnn_dn_conv_matrix_unsummed, axis=(5, 6, 7))
        )

        # ===============================================================================
        # Save precomputed integral matrices to external file
        # ===============================================================================

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(fname), exist_ok=True)

        # Save only the integral matrices
        save_dict = {
            'NNNConvMatrixInt': basic_matrices['NNNConvMatrixInt'],
            'dNNNConvMatrixInt': basic_matrices['dNNNConvMatrixInt'],
            'NNdNConvMatrixInt': basic_matrices['NNdNConvMatrixInt'],
            'dNNdNConvMatrixInt': basic_matrices['dNNdNConvMatrixInt']
        }
        scipy.io.savemat(fname, save_dict)

    # ===============================================================================
    # Compute the elements of large convolution matrices for all domains
    # ===============================================================================

    n_nodes = comp_struct['Advanced']['N_nodes']
    n_domain = comp_struct['Data']['N_domain']

    # Initialize cell arrays (lists in Python)
    basic_matrices['dNNdNConvMatrixIntLarge'] = [None] * n_domain
    basic_matrices['dNNNConvMatrixIntLarge'] = [None] * n_domain
    basic_matrices['NNdNConvMatrixIntLarge'] = [None] * n_domain
    basic_matrices['NNNConvMatrixIntLarge'] = [None] * n_domain

    for ii_d in range(n_domain):
        ones_var_num = np.ones(comp_struct['Data']['DVarNum'][ii_d])
        msize = comp_struct['Data']['DVarNum'][ii_d] * n_nodes

        # Initialize arrays for this domain
        basic_matrices['dNNdNConvMatrixIntLarge'][ii_d] = np.zeros((3, 3, msize, n_nodes, msize))
        basic_matrices['dNNNConvMatrixIntLarge'][ii_d] = np.zeros((3, msize, n_nodes, msize))
        basic_matrices['NNdNConvMatrixIntLarge'][ii_d] = np.zeros((3, msize, n_nodes, msize))
        basic_matrices['NNNConvMatrixIntLarge'][ii_d] = np.zeros((msize, n_nodes, msize))

        for ii_k in range(n_nodes):
            # Process NNNConvMatrixInt: shape (N_nodes, N_nodes, N_nodes)
            slice_nnn = np.squeeze(basic_matrices['NNNConvMatrixInt'][:, ii_k, :])
            basic_matrices['NNNConvMatrixIntLarge'][ii_d][:, ii_k, :] = np.kron(slice_nnn, ones_var_num)

            for ii_c in range(3):
                # Process NNdNConvMatrixInt: shape (3, N_nodes, N_nodes, N_nodes)
                slice_nndn = np.squeeze(basic_matrices['NNdNConvMatrixInt'][ii_c, :, ii_k, :])
                basic_matrices['NNdNConvMatrixIntLarge'][ii_d][ii_c, :, ii_k, :] = np.kron(slice_nndn, ones_var_num)

                # Process dNNNConvMatrixInt: shape (3, N_nodes, N_nodes, N_nodes)
                slice_dnnn = np.squeeze(basic_matrices['dNNNConvMatrixInt'][ii_c, :, ii_k, :])
                basic_matrices['dNNNConvMatrixIntLarge'][ii_d][ii_c, :, ii_k, :] = np.kron(slice_dnnn, ones_var_num)

                for ii_c2 in range(3):
                    # Process dNNdNConvMatrixInt: shape (3, 3, N_nodes, N_nodes, N_nodes)
                    slice_dnndn = np.squeeze(basic_matrices['dNNdNConvMatrixInt'][ii_c, ii_c2, :, ii_k, :])
                    basic_matrices['dNNdNConvMatrixIntLarge'][ii_d][ii_c, ii_c2, :, ii_k, :] = np.kron(slice_dnndn,
                                                                                                       ones_var_num)

    return basic_matrices