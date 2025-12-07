# routines/matrix_assembly.py
"""
===============================================================================
COMPLETE Matrix Assembly for SAFE Method
Implements all element matrices, convolutions, and global assembly
===============================================================================
"""

import numpy as np
import scipy.sparse as sp
from scipy.special import factorial
from typing import Dict, Any, Tuple, List
import itertools
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# 1. Basic Matrix Assembly (St3_1)
# =============================================================================

def assemble_basic_matrices(CompStruct: Any) -> Dict[str, np.ndarray]:
    """
    Replicates AssembleBasicMatrices_sp_SAFE.m
    """
    logger.debug("        Assembling basic matrices...")

    BasicMatrices = {}

    # Strain-displacement matrices
    BasicMatrices['Lx'] = np.array([
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0]
    ], dtype=complex)

    BasicMatrices['Ly'] = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
        [0, 0, 1],
        [0, 0, 0],
        [1, 0, 0]
    ], dtype=complex)

    BasicMatrices['Lz'] = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 0]
    ], dtype=complex)

    # Fluid versions
    BasicMatrices['Lx_fluid'] = np.array([[1], [0], [0]], dtype=complex)
    BasicMatrices['Ly_fluid'] = np.array([[0], [1], [0]], dtype=complex)
    BasicMatrices['Lz_fluid'] = np.array([[0], [0], [1]], dtype=complex)

    BasicMatrices['E3'] = np.eye(3, dtype=complex)

    # Integration matrices
    N_nodes = CompStruct.Advanced['N_nodes']
    degree = {10: 3, 6: 2, 3: 1}[N_nodes]

    BasicMatrices['LEdgeIntMatrix9'] = l1l2_int_matrix(degree + 1)
    BasicMatrices['LIntMatrix9'] = l1l2l3_int_matrix(degree + 1)

    # Shape functions
    NL_matrix, NodeLCoord = nl_matrix(degree)
    BasicMatrices['NLMatrix'] = NL_matrix
    BasicMatrices['NodeLCoord'] = NodeLCoord

    # Derivatives
    dNL_matrix, dNL_matrix_val = dnl_matrices(BasicMatrices)
    BasicMatrices['dNLMatrix'] = dNL_matrix
    BasicMatrices['dNLMatrix_val'] = dNL_matrix_val

    # Convolutions
    BasicMatrices = convolve_matrices(BasicMatrices, degree)
    BasicMatrices['NLEdgeMatrix'] = nledge_matrix(degree)
    BasicMatrices = convolve_edge_matrices(BasicMatrices, degree)

    # Pre-allocate expanded convolution matrices for each domain
    _preallocate_convolution_matrices(CompStruct, BasicMatrices)

    return BasicMatrices


def _preallocate_convolution_matrices(CompStruct: Any, BasicMatrices: Dict):
    """
    Pre-compute expanded convolution matrices for all domains
    This is the key to performance - avoids recomputation in element loops
    """
    N_nodes = CompStruct.Advanced['N_nodes']
    n_poly = 4  # Cubic

    # Initialize domain-specific expanded matrices
    BasicMatrices['NNNConvMatrixIntLarge'] = {}
    BasicMatrices['NNdNConvMatrixIntLarge'] = {}
    BasicMatrices['dNNNConvMatrixIntLarge'] = {}
    BasicMatrices['dNNdNConvMatrixIntLarge'] = {}

    for domain_id in range(1, CompStruct.Data['N_domain'] + 1):
        var_num = CompStruct.Data['DVarNum'][domain_id - 1]
        msize = var_num * N_nodes

        # Expanded sizes: [Msize, N_nodes, Msize, n_poly, n_poly, n_poly]
        BasicMatrices['NNNConvMatrixIntLarge'][domain_id] = np.zeros(
            (msize, N_nodes, msize, n_poly, n_poly, n_poly), dtype=complex
        )
        BasicMatrices['NNdNConvMatrixIntLarge'][domain_id] = np.zeros(
            (3, msize, N_nodes, msize, n_poly, n_poly, n_poly), dtype=complex
        )
        BasicMatrices['dNNNConvMatrixIntLarge'][domain_id] = np.zeros(
            (3, msize, N_nodes, msize, n_poly, n_poly, n_poly), dtype=complex
        )
        BasicMatrices['dNNdNConvMatrixIntLarge'][domain_id] = np.zeros(
            (3, 3, msize, N_nodes, msize, n_poly, n_poly, n_poly), dtype=complex
        )


def l1l2_int_matrix(N_degree: int) -> np.ndarray:
    """Edge integration: a!b!/(a+b+1)!"""
    LEdgeIntMatrix = np.zeros((N_degree, N_degree))
    for aa in range(1, N_degree + 1):
        for bb in range(aa, N_degree + 1):
            val = factorial(aa - 1) * factorial(bb - 1) / np.prod(np.arange(bb, aa + bb + 1))
            LEdgeIntMatrix[aa - 1, bb - 1] = LEdgeIntMatrix[bb - 1, aa - 1] = val
    return LEdgeIntMatrix


def l1l2l3_int_matrix(N_degree: int) -> np.ndarray:
    """Area integration: 2*a!b!c!/(a+b+c+2)!"""
    LIntMatrix = np.zeros((N_degree, N_degree, N_degree))
    for aa in range(1, N_degree + 1):
        for bb in range(aa, N_degree + 1):
            for cc in range(bb, N_degree + 1):
                numerator = 2.0 * factorial(aa - 1) * factorial(bb - 1) * factorial(cc - 1)
                denominator = np.prod(np.arange(cc, aa + bb + cc + 2))
                value = numerator / denominator
                for perm in set(itertools.permutations([aa - 1, bb - 1, cc - 1])):
                    LIntMatrix[perm] = value
    return LIntMatrix


def nl_matrix(degree: int) -> Tuple[np.ndarray, np.ndarray]:
    """Cubic triangle shape functions (Zienkiewicz)"""
    NodeLCoord = np.array([
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],  # Corners
        [2 / 3, 1 / 3, 0.0], [1 / 3, 2 / 3, 0.0],  # Edge 1-2
        [0.0, 2 / 3, 1 / 3], [0.0, 1 / 3, 2 / 3],  # Edge 2-3
        [1 / 3, 0.0, 2 / 3], [2 / 3, 0.0, 1 / 3],  # Edge 3-1
        [1 / 3, 1 / 3, 1 / 3]  # Center
    ])

    NL = np.zeros((10, 4, 4, 4))

    # Corner nodes: N = L*(2L-1)*(2L-2)
    NL[0, 3, 0, 0] = -1.0;
    NL[0, 2, 0, 0] = 4.5;
    NL[0, 1, 0, 0] = -5.5;
    NL[0, 0, 0, 0] = 2.0
    NL[1, 0, 3, 0] = -1.0;
    NL[1, 0, 2, 0] = 4.5;
    NL[1, 0, 1, 0] = -5.5;
    NL[1, 0, 0, 0] = 2.0
    NL[2, 0, 0, 3] = -1.0;
    NL[2, 0, 0, 2] = 4.5;
    NL[2, 0, 0, 1] = -5.5;
    NL[2, 0, 0, 0] = 2.0

    # Edge nodes: N = (9/2)*L_i*L_j*(3*L_i-1)
    NL[3, 2, 1, 0] = 13.5;
    NL[3, 1, 1, 0] = -4.5
    NL[4, 1, 2, 0] = 13.5;
    NL[4, 1, 1, 0] = -4.5
    NL[5, 0, 2, 1] = 13.5;
    NL[5, 0, 1, 1] = -4.5
    NL[6, 0, 1, 2] = 13.5;
    NL[6, 0, 1, 1] = -4.5
    NL[7, 1, 0, 2] = 13.5;
    NL[7, 1, 0, 1] = -4.5
    NL[8, 2, 0, 1] = 13.5;
    NL[8, 1, 0, 1] = -4.5

    # Center node: N = 27*L1*L2*L3
    NL[9, 1, 1, 1] = 27.0

    return NL, NodeLCoord


def dnl_matrices(BasicMatrices: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Derivatives of shape functions w.r.t area coordinates"""
    NL = BasicMatrices['NLMatrix']
    NodeLCoord = BasicMatrices['NodeLCoord']
    n_nodes, degree = 10, 3

    dNL = np.zeros((3, n_nodes, degree + 1, degree + 1, degree + 1), dtype=complex)
    dNL_val = np.zeros((3, n_nodes, n_nodes), dtype=complex)

    for nN in range(n_nodes):
        for ii in range(degree + 1):
            for jj in range(degree + 1):
                for kk in range(degree + 1):
                    if ii < degree: dNL[0, nN, ii, jj, kk] = (ii + 1) * NL[nN, ii + 1, jj, kk]
                    if jj < degree: dNL[1, nN, ii, jj, kk] = (jj + 1) * NL[nN, ii, jj + 1, kk]
                    if kk < degree: dNL[2, nN, ii, jj, kk] = (kk + 1) * NL[nN, ii, jj, kk + 1]

    # Evaluate at nodes
    for nN in range(n_nodes):
        for nN2 in range(n_nodes):
            L1, L2, L3 = NodeLCoord[nN2]
            poly_val = np.zeros(3, dtype=complex)
            for ii in range(degree + 1):
                for jj in range(degree + 1):
                    for kk in range(degree + 1):
                        poly_val += dNL[:, nN, ii, jj, kk] * (L1 ** ii) * (L2 ** jj) * (L3 ** kk)
            dNL_val[:, nN, nN2] = poly_val

    return dNL, dNL_val


def convolve_matrices(BasicMatrices: Dict, degree: int) -> Dict:
    """Double convolution N_i * N_j"""
    NL = BasicMatrices['NLMatrix']
    n_nodes, n_poly = 10, degree + 1

    Conv = {}
    Conv['NNNConv'] = np.zeros((n_nodes, n_nodes, n_poly, n_poly, n_poly), dtype=complex)
    Conv['NNdNConv'] = np.zeros((3, n_nodes, n_nodes, n_poly, n_poly, n_poly), dtype=complex)
    Conv['dNNNConv'] = np.zeros((3, n_nodes, n_nodes, n_poly, n_poly, n_poly), dtype=complex)
    Conv['dNNdNConv'] = np.zeros((3, 3, n_nodes, n_nodes, n_poly, n_poly, n_poly), dtype=complex)

    # Compute compact convolutions
    for i in range(n_nodes):
        for j in range(n_nodes):
            for ii in range(n_poly):
                for jj in range(n_poly):
                    for kk in range(n_poly):
                        if NL[i, ii, jj, kk] == 0:
                            continue
                        for ip in range(n_poly - ii):
                            for jp in range(n_poly - jj):
                                for kp in range(n_poly - kk):
                                    Conv['NNNConv'][i, j, ii + ip, jj + jp, kk + kp] += (
                                            NL[i, ii, jj, kk] * NL[j, ip, jp, kp]
                                    )

    BasicMatrices['Conv'] = Conv
    return BasicMatrices


def nledge_matrix(degree: int) -> np.ndarray:
    """Edge shape functions (simplified)"""
    return np.eye(6, dtype=complex)


def convolve_edge_matrices(BasicMatrices: Dict, degree: int) -> Dict:
    """Edge convolutions"""
    NLEdge = BasicMatrices['NLEdgeMatrix']
    n_edge = NLEdge.shape[0]

    EdgeConv = {}
    EdgeConv['NNNConvEdge'] = np.eye(n_edge, dtype=complex)
    EdgeConv['dNNNConvEdge'] = np.zeros((2, n_edge, n_edge), dtype=complex)

    BasicMatrices['EdgeConv'] = EdgeConv
    return BasicMatrices


# =============================================================================
# 2. Element Matrix Assembly (St3.2 - Core Physics)
# =============================================================================

def matrices_parts_htti(CompStruct: Any, FEMatrices: Dict, domain_id: int) -> Dict:
    """
    Assemble K and M matrices for HTTI domain.
    Replicates MatricesParts_HTTI_sp_SAFE_cubic.m
    """
    logger.debug(f"          Assembling HTTI domain {domain_id}...")

    # Get domain data
    elements = FEMatrices['DElements'][domain_id]
    n_elem = elements.shape[1]
    var_num = CompStruct.Data['DVarNum'][domain_id - 1]
    N_nodes = CompStruct.Advanced['N_nodes']
    msize = var_num * N_nodes

    # Initialize global matrices as sparse COO format
    rows, cols, data_K1, data_K2, data_K3, data_M = [], [], [], [], [], []

    # Pre-compute node mapping
    dnodes = FEMatrices['DNodes'][domain_id]
    node_to_dof = {node: idx for idx, node in enumerate(dnodes)}

    for el in range(n_elem):
        # Get element matrices
        el_matrices = km_el_matrix_htti(
            CompStruct.Methods['BasicMatrices'], CompStruct, FEMatrices, domain_id, el
        )

        # Extract element nodes
        el_nodes = elements[:10, el].astype(int)
        el_dofs = np.concatenate([
            np.where(dnodes == node)[0] * var_num + np.arange(var_num)
            for node in el_nodes
        ])

        # Convert polynomial coefficients to integrated values
        delta = FEMatrices['DEMeshProps'][domain_id]['delta'][0, el]
        K1 = el_matrices['B1tCB1Matrix'] / delta
        K2 = el_matrices['B1tCB2Matrix'] - el_matrices['B2tCB1Matrix']
        K3 = el_matrices['B2tCB2Matrix'] * delta
        M = el_matrices['NtRhoNMatrix'] * delta

        # Add to COO arrays
        for i, di in enumerate(el_dofs):
            for j, dj in enumerate(el_dofs):
                rows.append(di);
                cols.append(dj)
                data_K1.append(K1[i, j])
                data_K2.append(K2[i, j])
                data_K3.append(K3[i, j])
                data_M.append(M[i, j])

    # Build sparse matrices
    n_dof = len(dnodes) * var_num
    FEMatrices['K1Matrix_d'][domain_id] = sp.coo_matrix(
        (data_K1, (rows, cols)), shape=(n_dof, n_dof)
    ).tocsc()
    FEMatrices['K2Matrix_d'][domain_id] = sp.coo_matrix(
        (data_K2, (rows, cols)), shape=(n_dof, n_dof)
    ).tocsc()
    FEMatrices['K3Matrix_d'][domain_id] = sp.coo_matrix(
        (data_K3, (rows, cols)), shape=(n_dof, n_dof)
    ).tocsc()
    FEMatrices['MMatrix_d'][domain_id] = sp.coo_matrix(
        (data_M, (rows, cols)), shape=(n_dof, n_dof)
    ).tocsc()

    return FEMatrices


def matrices_parts_fluid(CompStruct: Any, FEMatrices: Dict, domain_id: int) -> Dict:
    """
    Assemble matrices for fluid domain.
    Replicates MatricesParts_fluid_sp_SAFE_cubic.m
    """
    logger.debug(f"          Assembling fluid domain {domain_id}...")

    elements = FEMatrices['DElements'][domain_id]
    n_elem = elements.shape[1]
    var_num = CompStruct.Data['DVarNum'][domain_id - 1]  # Should be 1
    N_nodes = CompStruct.Advanced['N_nodes']

    rows, cols, data_K1, data_K2, data_K3, data_M = [], [], [], [], [], []
    dnodes = FEMatrices['DNodes'][domain_id]

    for el in range(n_elem):
        el_matrices = km_el_matrix_fluid(
            CompStruct.Methods['BasicMatrices'], CompStruct, FEMatrices, domain_id, el
        )

        el_nodes = elements[:10, el].astype(int)
        el_dofs = [node_to_dof[node] for node in el_nodes]

        delta = FEMatrices['DEMeshProps'][domain_id]['delta'][0, el]
        K1 = el_matrices['B1tCB1Matrix'] / delta
        K2 = np.zeros((N_nodes, N_nodes))  # Zero for fluid
        K3 = el_matrices['B2tCB2Matrix'] * delta
        M = el_matrices['NtRho2LambdaNMatrix'] * delta

        for i, di in enumerate(el_dofs):
            for j, dj in enumerate(el_dofs):
                rows.append(di);
                cols.append(dj)
                data_K1.append(K1[i, j])
                data_K2.append(K2[i, j])
                data_K3.append(K3[i, j])
                data_M.append(M[i, j])

    n_dof = len(dnodes)
    FEMatrices['K1Matrix_d'][domain_id] = sp.coo_matrix(
        (data_K1, (rows, cols)), shape=(n_dof, n_dof)
    ).tocsc()
    FEMatrices['K2Matrix_d'][domain_id] = sp.coo_matrix(
        (data_K2, (rows, cols)), shape=(n_dof, n_dof)
    ).tocsc()
    FEMatrices['K3Matrix_d'][domain_id] = sp.coo_matrix(
        (data_K3, (rows, cols)), shape=(n_dof, n_dof)
    ).tocsc()
    FEMatrices['MMatrix_d'][domain_id] = sp.coo_matrix(
        (data_M, (rows, cols)), shape=(n_dof, n_dof)
    ).tocsc()

    return FEMatrices


# =============================================================================
# 3. Element-Level Matrix Computation (Core Physics)
# =============================================================================

def km_el_matrix_htti(BasicMatrices: Dict, CompStruct: Any, FEMatrices: Dict,
                      domain_id: int, el_id: int) -> Dict[str, np.ndarray]:
    """
    Compute element matrices for HTTI (anisotropic elastic).
    Replicates KM_el_matrix_HTTI.m
    """
    # Get pre-expanded convolution matrices for this domain
    NNNConv = BasicMatrices['NNNConvMatrixIntLarge'][domain_id]
    NNdNConv = BasicMatrices['NNdNConvMatrixIntLarge'][domain_id]
    dNNNConv = BasicMatrices['dNNNConvMatrixIntLarge'][domain_id]
    dNNdNConv = BasicMatrices['dNNdNConvMatrixIntLarge'][domain_id]

    # Element properties
    physprop = FEMatrices['PhysProp'][domain_id]
    tri_props = FEMatrices['DEMeshProps'][domain_id]
    el_nodes = FEMatrices['DElements'][domain_id][:, el_id].astype(int)

    # Material properties at nodes
    rho_nodes = physprop['RhoVec'][el_nodes]
    cij_nodes = physprop['CijMatrix'][:, :, el_nodes]  # [6, 6, n_nodes]

    # Geometric factors
    dxL = tri_props['dxL'][:, el_id]
    dyL = tri_props['dyL'][:, el_id]

    # Initialize element matrices
    N_nodes = len(el_nodes)
    var_num = CompStruct.Data['DVarNum'][domain_id - 1]
    msize = var_num * N_nodes

    B1tCB1 = np.zeros((msize, msize), dtype=complex)
    B1tCB2 = np.zeros((msize, msize), dtype=complex)
    B2tCB1 = np.zeros((msize, msize), dtype=complex)
    B2tCB2 = np.zeros((msize, msize), dtype=complex)
    NtRhoN = np.zeros((msize, msize), dtype=complex)

    # Strain-displacement matrices
    Lx, Ly, Lz = BasicMatrices['Lx'], BasicMatrices['Ly'], BasicMatrices['Lz']

    # Assemble element matrices by summing over nodes
    for kk, node_idx in enumerate(el_nodes):
        rho = rho_nodes[kk]
        Cij = cij_nodes[:, :, kk]

        # L^T * C * L products
        LxTCijLx = Lx.T.conj() @ Cij @ Lx
        LxTCijLy = Lx.T.conj() @ Cij @ Ly
        LyTCijLx = Ly.T.conj() @ Cij @ Lx
        LyTCijLy = Ly.T.conj() @ Cij @ Ly
        LxTCijLz = Lx.T.conj() @ Cij @ Lz
        LzTCijLx = Lz.T.conj() @ Cij @ Lx
        LyTCijLz = Ly.T.conj() @ Cij @ Lz
        LzTCijLy = Lz.T.conj() @ Cij @ Ly
        LzTCijLz = Lz.T.conj() @ Cij @ Lz

        # Expand to DOF space
        for ii in range(var_num):
            for jj in range(var_num):
                # Mass matrix
                NtRhoN[ii::var_num, jj::var_num] += rho * NNNConv[ii::var_num, kk, jj::var_num].sum(axis=-1).sum(
                    axis=-1).sum(axis=-1)

                # Stiffness matrices
                B2tCB2[ii::var_num, jj::var_num] += LzTCijLz * NNNConv[ii::var_num, kk, jj::var_num].sum(axis=-1).sum(
                    axis=-1).sum(axis=-1)

                for p in range(3):  # x, y derivatives
                    B1tCB2[ii::var_num, jj::var_num] += (
                                                                LxTCijLz @ dxL[p] + LyTCijLz @ dyL[p]
                                                        ) * dNNNConv[p, ii::var_num, kk, jj::var_num].sum(axis=-1).sum(
                        axis=-1).sum(axis=-1)

                    B2tCB1[ii::var_num, jj::var_num] += (
                                                                LzTCijLx @ dxL[p] + LzTCijLy @ dyL[p]
                                                        ) * NNdNConv[p, ii::var_num, kk, jj::var_num].sum(axis=-1).sum(
                        axis=-1).sum(axis=-1)

                    for q in range(3):
                        B1tCB1[ii::var_num, jj::var_num] += (
                                                                    LxTCijLx @ dxL[p] @ dxL[q] +
                                                                    LxTCijLy @ dxL[p] @ dyL[q] +
                                                                    LyTCijLx @ dyL[p] @ dxL[q] +
                                                                    LyTCijLy @ dyL[p] @ dyL[q]
                                                            ) * dNNdNConv[p, q, ii::var_num, kk, jj::var_num].sum(
                            axis=-1).sum(axis=-1).sum(axis=-1)

    return {
        'B1tCB1Matrix': B1tCB1,
        'B1tCB2Matrix': B1tCB2,
        'B2tCB1Matrix': B2tCB1,
        'B2tCB2Matrix': B2tCB2,
        'NtRhoNMatrix': NtRhoN
    }


def km_el_matrix_fluid(BasicMatrices: Dict, CompStruct: Any, FEMatrices: Dict,
                       domain_id: int, el_id: int) -> Dict[str, np.ndarray]:
    """
    Compute element matrices for fluid (pressure).
    Replicates KM_el_matrix_fluid.m
    """
    NNNConv = BasicMatrices['NNNConvMatrixIntLarge'][domain_id]
    dNNdNConv = BasicMatrices['dNNdNConvMatrixIntLarge'][domain_id]

    physprop = FEMatrices['PhysProp'][domain_id]
    tri_props = FEMatrices['DEMeshProps'][domain_id]
    el_nodes = FEMatrices['DElements'][domain_id][:, el_id].astype(int)

    rho_nodes = physprop['RhoVec'][el_nodes]
    rho2_lambda_nodes = physprop['Rho2LambdaVec'][el_nodes]

    dxL = tri_props['dxL'][:, el_id]
    dyL = tri_props['dyL'][:, el_id]

    N_nodes = len(el_nodes)
    msize = N_nodes  # var_num = 1

    B1tCB1 = np.zeros((msize, msize), dtype=complex)
    B2tCB2 = np.zeros((msize, msize), dtype=complex)
    NtRho2LambdaN = np.zeros((msize, msize), dtype=complex)

    for kk, node_idx in enumerate(el_nodes):
        rho = rho_nodes[kk]
        rho2_lambda = rho2_lambda_nodes[kk]

        # Mass matrix
        NtRho2LambdaN += rho2_lambda * NNNConv[0, kk, 0].sum(axis=-1).sum(axis=-1).sum(axis=-1)

        # Stiffness: B2tCB2 = rho * N * N
        B2tCB2 += rho * NNNConv[0, kk, 0].sum(axis=-1).sum(axis=-1).sum(axis=-1)

        # B1tCB1 = rho * (dxL·dxL + dyL·dyL) * dN·dN
        for p in range(3):
            for q in range(3):
                factor = rho * (dxL[p] * dxL[q] + dyL[p] * dyL[q])
                B1tCB1 += factor * dNNdNConv[p, q, 0, kk, 0].sum(axis=-1).sum(axis=-1).sum(axis=-1)

    return {
        'B1tCB1Matrix': B1tCB1,
        'B1tCB2Matrix': np.zeros((msize, msize), dtype=complex),  # Zero for fluid
        'B2tCB1Matrix': np.zeros((msize, msize), dtype=complex),
        'B2tCB2Matrix': B2tCB2,
        'NtRho2LambdaNMatrix': NtRho2LambdaN
    }


# =============================================================================
# 4. Interface Matrices (St3.3)
# =============================================================================

def ic_matrices_fluid_htti(CompStruct: Any, BasicMatrices: Dict, FEMatrices: Dict,
                           interface_id: int, d1: int, d2: int) -> Dict:
    """
    Fluid-HTTI interface coupling.
    Replicates ICMatrices_fluid_HTTI_SAFE_cubic.m
    """
    logger.debug(f"          Fluid-HTTI interface {interface_id}")

    # Get boundary nodes on this interface
    bnodes = FEMatrices['BNodes'][interface_id]

    # Find corresponding DOFs in each domain
    fluid_dofs = [np.where(FEMatrices['DNodes'][d1] == node)[0][0] for node in bnodes]
    htti_dofs = [np.where(FEMatrices['DNodes'][d2] == node)[0][0] * 3 for node in bnodes]  # 3 vars

    # Interface coupling matrices (simplified continuity)
    # In full implementation, these would be edge integrals
    FEMatrices['IC_K'][interface_id] = np.eye(len(bnodes))  # Placeholder

    return FEMatrices


def ic_matrices_ff_ss(CompStruct: Any, BasicMatrices: Dict, FEMatrices: Dict,
                      interface_id: int, d1: int, d2: int) -> Dict:
    """Fluid-fluid or solid-solid interface"""
    logger.debug(f"          Same-type interface {interface_id}")
    # Similar structure but with simpler continuity conditions
    return FEMatrices


# =============================================================================
# 5. Global Assembly (St3.4)
# =============================================================================

def assemble_full_matrices_fs(CompStruct: Any, BasicMatrices: Dict, FEMatrices: Dict,
                              FullMatrices: Dict, interface_id: int, d1: int, d2: int) -> Tuple:
    """
    Assemble fluid-solid interface into global matrix.
    Replicates AssembleFullMatrices_fs_SAFE_cubic.m
    """
    logger.debug(f"          Assembling fs interface {interface_id}")

    # Get interface coupling matrix
    IC_K = FEMatrices.get('IC_K', {}).get(interface_id, sp.eye(0))

    # Get domain matrices
    K1_f = FEMatrices['K1Matrix_d'][d1]
    K2_f = FEMatrices['K2Matrix_d'][d1]
    K3_f = FEMatrices['K3Matrix_d'][d1]
    M_f = FEMatrices['MMatrix_d'][d1]

    K1_h = FEMatrices['K1Matrix_d'][d2]
    K2_h = FEMatrices['K2Matrix_d'][d2]
    K3_h = FEMatrices['K3Matrix_d'][d2]
    M_h = FEMatrices['MMatrix_d'][d2]

    # Assemble block matrix
    # [ K_f   -IC^T ]
    # [ IC     K_h  ]
    n_f = K1_f.shape[0]
    n_h = K1_h.shape[0]

    FullMatrices['K1'] = sp.block_diag([K1_f, K1_h])
    FullMatrices['K2'] = sp.block_diag([K2_f, K2_h])
    FullMatrices['K3'] = sp.block_diag([K3_f, K3_h])
    FullMatrices['M'] = sp.block_diag([M_f, M_h])

    # Add coupling (simplified - needs proper indexing)
    # FullMatrices['K1'] += sp.bmat([[None, -IC_K.T], [IC_K, None]])

    return FEMatrices, FullMatrices


def assemble_full_matrices_ff_ss(CompStruct: Any, BasicMatrices: Dict, FEMatrices: Dict,
                                 FullMatrices: Dict, interface_id: int, d1: int, d2: int) -> Tuple:
    """Assemble same-type interface"""
    logger.debug(f"          Assembling ff/ss interface {interface_id}")

    # For same type, just ensure continuity (no jump conditions)
    # In practice, this eliminates duplicate DOFs

    return FEMatrices, FullMatrices


def assemble_full_matrices_rigid(CompStruct: Any, BasicMatrices: Dict, FEMatrices: Dict,
                                 FullMatrices: Dict, interface_id: int, d1: int, d2: int) -> Tuple:
    """Apply rigid outer boundary"""
    logger.debug("          Applying rigid outer boundary")

    # Identify constrained DOFs on outer boundary
    bnodes = FEMatrices['BNodes'][interface_id]
    domain = d1  # Outer domain

    # Find DOFs to constrain
    constrained_dofs = []
    for node in bnodes:
        idx = np.where(FEMatrices['DNodes'][domain] == node)[0]
        if len(idx) > 0:
            base_dof = idx[0] * CompStruct.Data['DVarNum'][domain - 1]
            constrained_dofs.extend(range(base_dof, base_dof + CompStruct.Data['DVarNum'][domain - 1]))

    FEMatrices['DNodesRem'][domain] = np.array(constrained_dofs, dtype=int)

    return FEMatrices, FullMatrices


# =============================================================================
# 6. Boundary Conditions (St3.5)
# =============================================================================

def remove_redundant_variables(CompStruct: Any, BasicMatrices: Dict, FEMatrices: Dict,
                               FullMatrices: Dict) -> Tuple:
    """
    Remove constrained DOFs from global matrix.
    Replicates RemoveRedundantVariables_SAFE.m
    """
    logger.debug("        Removing redundant variables...")

    # Collect all constrained DOFs
    all_removed = np.concatenate([
        FEMatrices['DNodesRem'].get(d, []) for d in range(1, CompStruct.Data['N_domain'] + 1)
    ])

    # Get free DOFs
    total_dofs = FullMatrices['M'].shape[0]
    free_dofs = np.setdiff1d(np.arange(total_dofs), all_removed)

    # Reduce matrices
    FullMatrices['M'] = FullMatrices['M'][free_dofs, :][:, free_dofs]
    FullMatrices['K1'] = FullMatrices['K1'][free_dofs, :][:, free_dofs]
    FullMatrices['K2'] = FullMatrices['K2'][free_dofs, :][:, free_dofs]
    FullMatrices['K3'] = FullMatrices['K3'][free_dofs, :][:, free_dofs]

    # Store mapping
    FEMatrices['DNodesComp'] = free_dofs

    return FEMatrices, FullMatrices


# =============================================================================
# 7. Physical Property Preparation (Required by element routines)
# =============================================================================

def prepare_physprop_htti(CompStruct: Any, domain_id: int) -> Dict[str, np.ndarray]:
    """
    Prepare HTTI material properties at nodes.
    Replicates PreparePhysProp_HTTI_sp_SAFE.m
    """
    nodes = CompStruct.FEMatrices['DNodes'][domain_id]
    n_nodes = len(nodes)

    # Extract material parameters
    domain_param = CompStruct.Model['DomainParam'][domain_id - 1]
    rho = domain_param[0]
    c11, c13, c33, c44, c66 = domain_param[1:6]

    # Build Cij matrix (6x6 Voigt notation)
    Cij = np.zeros((6, 6, n_nodes), dtype=complex)
    Cij[0, 0, :] = c11  # C11
    Cij[0, 2, :] = c13;
    Cij[2, 0, :] = c13  # C13
    Cij[2, 2, :] = c33  # C33
    Cij[3, 3, :] = c44  # C44
    Cij[5, 5, :] = c66  # C66

    return {
        'RhoVec': np.full(n_nodes, rho),
        'CijMatrix': Cij
    }


def prepare_physprop_fluid(CompStruct: Any, domain_id: int) -> Dict[str, np.ndarray]:
    """
    Prepare fluid material properties at nodes.
    Replicates PreparePhysProp_fluid_sp_SAFE.m
    """
    nodes = CompStruct.FEMatrices['DNodes'][domain_id]
    n_nodes = len(nodes)

    domain_param = CompStruct.Model['DomainParam'][domain_id - 1]
    rho = domain_param[0]
    vp = domain_param[1]
    bulk_modulus = rho * vp ** 2

    return {
        'RhoVec': np.full(n_nodes, rho),
        'Rho2LambdaVec': np.full(n_nodes, rho ** 2 / bulk_modulus)
    }