# fem_matrices.py
# Complex SAFE stiffness and mass matrices with PML/ABC damping
# Reads damping parameters directly from JSON model
from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from typing import Dict, Any, Tuple


# ------------------------------------------------------------------
# 1.  Tri6 element – shape functions & quadrature
# ------------------------------------------------------------------
class Tri6Element:
    """Quadratic 6-node triangle – shape functions and derivatives."""
    qp = np.array([
        [0.470142064105115, 0.470142064105115],
        [0.470142064105115, 0.059715871789770],
        [0.059715871789770, 0.470142064105115],
        [0.101286507323456, 0.101286507323456],
        [0.101286507323456, 0.797426985353087],
        [0.797426985353087, 0.101286507323456],
        [0.333333333333333, 0.333333333333333]
    ])
    w = np.array([
        0.066197076394253, 0.066197076394253, 0.066197076394253,
        0.062969590272413, 0.062969590272413, 0.062969590272413,
        0.1125
    ])

    @staticmethod
    def shape(xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray]:
        z = 1.0 - xi - eta
        N = np.array([
            z * (2.0 * z - 1.0),
            xi * (2.0 * xi - 1.0),
            eta * (2.0 * eta - 1.0),
            4.0 * xi * z,
            4.0 * xi * eta,
            4.0 * eta * z
        ])
        dN = np.array([
            [4.0 * (xi + eta) - 3.0, 4.0 * xi - 1.0, 0.0, 4.0 * (1.0 - 2.0 * xi - eta), 4.0 * eta, -4.0 * eta],
            [4.0 * (xi + eta) - 3.0, 0.0, 4.0 * eta - 1.0, -4.0 * xi, 4.0 * xi, 4.0 * (1.0 - xi - 2.0 * eta)]
        ])
        return N, dN


# ------------------------------------------------------------------
# 2.  Bond transform (plane-strain)
# ------------------------------------------------------------------
def rot_cij(cij: np.ndarray, theta: float) -> np.ndarray:
    """Rotate 6×6 elastic matrix by angle theta (degrees)."""
    if isinstance(cij, list):
        cij = np.array(cij, dtype=complex)
    c = cij.reshape(6, 6) if cij.size == 36 else cij
    rad = np.deg2rad(theta)
    cth, sth = np.cos(rad), np.sin(rad)
    c2, s2, cs = cth * cth, sth * sth, cth * sth

    R = np.zeros((6, 6), dtype=complex)
    R[0, 0] = c2;           R[0, 1] = s2;           R[0, 3] = 2.0 * cs;
    R[1, 0] = s2;           R[1, 1] = c2;           R[1, 3] = -2.0 * cs;
    R[2, 2] = 1.0;
    R[3, 0] = -cs;          R[3, 1] = cs;           R[3, 3] = c2 - s2;
    R[4, 4] = cth;          R[4, 5] = sth;
    R[5, 4] = -sth;         R[5, 5] = cth;
    return R @ c @ R.T


# ------------------------------------------------------------------
# 3.  Complex damping profile (PML/ABC)
# ------------------------------------------------------------------
def _damping_profile(r: np.ndarray, r_max: float, factor: float, degree: float) -> np.ndarray:
    """gamma(r) = factor * ((r - r_min)/r_max)^degree."""
    r_norm = np.clip(r / r_max, 0.0, 1.0)
    return factor * (r_norm ** degree)


def _complex_damping(nodes: np.ndarray, thickness: float, factor: float, degree: float) -> np.ndarray:
    """Complex multiplier 1 - 1j*gamma(r) for each node inside layer."""
    r = np.linalg.norm(nodes, axis=1)          # radial distance (circle model)
    gamma = _damping_profile(r, thickness, factor, degree)
    return 1.0 - 1j * gamma


# ------------------------------------------------------------------
# 4.  Global matrix assembly (complex)
# ------------------------------------------------------------------
def build_global_matrices(mesh, model: Dict[str, Any], omega: float):
    """
    Build complex stiffness (K) and mass (M) matrices with PML/ABC damping.
    Reads PML/ABC parameters directly from JSON model.
    Returns: K_complex, M_complex, active_dof
    """
    nodes = mesh.coord
    tri6  = mesh.tri6
    nnode = nodes.shape[0]
    ndof  = 3 * nnode                         # ux, uy, uz per node
    dom   = model['Model']                    # JSON layer data
    layers = len(dom['DomainType'])

    # COO buffers
    I_K, J_K, V_K = [], [], []
    I_M, J_M, V_M = [], [], []

    for el in tri6:
        xyz = nodes[el]                       # (6,2)
        ctr = xyz.mean(axis=0)
        r_ctr = np.hypot(ctr[0], ctr[1])

        # ---- identify layer by radius ----
        rx = np.asarray(dom['DomainRx'])
        layer = None
        for k in range(layers):
            if k == 0:
                if r_ctr <= rx[0]:
                    layer = k
                    break
            else:
                if rx[k-1] <= r_ctr <= rx[k]:
                    layer = k
                    break
        if layer is None:
            layer = layers - 1

        # ---- material matrix ----
        params = dom['DomainParam'][layer]
        dtype  = dom['DomainType'][layer].lower()

        if dtype == 'fluid':
            rho, vel = params[0], params[1]
            lam = rho * vel**2
            C = np.zeros((3, 3), dtype=complex)
            C[0, 0] = C[1, 1] = lam
            C[0, 1] = C[1, 0] = lam
            C[2, 2] = 0.0          # fluid shear = 0
        else:                       # solid (HTTI, etc.)
            rho = params[0]
            c11, c13, c33, c44, c66 = params[1:6]
            theta = params[6] if len(params) > 6 else 0.0

            C_3d = np.array([
                [c11, c13, c13, 0.0, 0.0, 0.0],
                [c13, c33, c13, 0.0, 0.0, 0.0],
                [c13, c13, c33, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, c44, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, c44, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, c66]
            ], dtype=complex)

            if abs(theta) > 1e-10:
                C_3d = rot_cij(C_3d, theta)

            # plane-strain reduction
            C = np.zeros((3, 3), dtype=complex)
            C[0, 0] = C_3d[0, 0]; C[0, 1] = C_3d[0, 1]; C[1, 0] = C_3d[1, 0]; C[1, 1] = C_3d[1, 1]; C[2, 2] = C_3d[5, 5]

        # ---- PML/ABC damping (read from JSON) ----
        r_max = rx[-1]
        pml_thick = float(dom.get('PML_thickness', 0.0))
        pml_factor = float(dom.get('PML_factor', 10.0))
        pml_degree = float(dom.get('PML_degree', 2.0))

        abc_thick = float(dom.get('ABC_thickness', 0.0))
        abc_factor = float(dom.get('ABC_factor', 0.1))
        abc_degree = float(dom.get('ABC_degree', 1.0))

        r = np.linalg.norm(xyz, axis=1)
        inside_pml = (r > (r_max - pml_thick)) & (pml_thick > 0.0)
        inside_abc = (r > (r_max - abc_thick)) & (abc_thick > 0.0) & (~inside_pml)

        if np.any(inside_pml):
            gamma_pml = _complex_damping(xyz, pml_thick, pml_factor, pml_degree)
            C = C * gamma_pml.mean()      # mean factor

        if np.any(inside_abc):
            gamma_abc = _complex_damping(xyz, abc_thick, abc_factor, abc_degree)
            C = C * gamma_abc.mean()

        # ---- elementary matrices ----
        Ke, Me = elem_mat(xyz, C, rho, omega)

        # ---- assembly into COO ----
        gdof = np.repeat(3 * el, 3) + np.tile([0, 1, 2], 6)
        for a in range(18):
            for b in range(18):
                I_K.append(gdof[a]); J_K.append(gdof[b]); V_K.append(Ke[a, b])
                I_M.append(gdof[a]); J_M.append(gdof[b]); V_M.append(Me[a, b])

    # ---- final complex matrices ----
    K = coo_matrix((V_K, (I_K, J_K)), shape=(ndof, ndof)).tocsr().astype(complex)
    M = coo_matrix((V_M, (I_M, J_M)), shape=(ndof, ndof)).tocsr().astype(complex)
    return K, M, np.arange(ndof)


# ------------------------------------------------------------------
# 5.  Element 18×18 matrices (Tri6)
# ------------------------------------------------------------------
def elem_mat(xyz: np.ndarray, C: np.ndarray, rho: float, omega: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return complex Ke, Me for single Tri6 element."""
    qp, w = Tri6Element.qp, Tri6Element.w
    Ke = np.zeros((18, 18), dtype=complex)
    Me = np.zeros((18, 18), dtype=complex)

    for xi, eta, wt in zip(qp[:, 0], qp[:, 1], w):
        N, dN = Tri6Element.shape(xi, eta)
        J = dN @ xyz                                # 2×2 Jacobian
        detJ = np.linalg.det(J)
        if abs(detJ) < 1e-12:
            continue

        invJ = np.linalg.inv(J)
        dNxy = invJ @ dN                            # 2×6 derivatives w.r.t x,y

        # strain-displacement B (3×18)
        B = np.zeros((3, 18), dtype=complex)
        for i in range(6):
            B[0, 3 * i] = dNxy[0, i]          # ε_xx
            B[1, 3 * i + 1] = dNxy[1, i]      # ε_yy
            B[2, 3 * i] = dNxy[1, i]          # γ_xy
            B[2, 3 * i + 1] = dNxy[0, i]

        Ke += wt * B.T @ C @ B * detJ

        # consistent mass
        for i in range(6):
            for j in range(6):
                Mij = rho * wt * N[i] * N[j] * detJ
                for d in range(3):
                    ii, jj = 3 * i + d, 3 * j + d
                    Me[ii, jj] += Mij

    return Ke, Me