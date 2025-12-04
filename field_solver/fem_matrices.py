# fem_matrices.py
# Complex SAFE stiffness and mass matrices with PML/ABC damping
# Applied per-Gauss-point, gamma ≤ 1.0 to avoid singular matrices
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
# 3.  Damping profile (PML/ABC) – per Gauss point
# ------------------------------------------------------------------
def _damping_profile(r: float, r_max: float, factor: float, degree: float) -> float:
    """gamma(r) = factor * ((r - r_min)/r_max)^degree, clipped ≤ 1.0."""
    r_norm = max(min((r - (r_max - r_max)) / r_max, 1.0), 0.0)
    return min(factor * (r_norm ** degree), 1.0)


def _complex_multiplier(r: float, thickness: float, factor: float, degree: float) -> complex:
    """1 - 1j*gamma(r), clipped to avoid singular matrices."""
    gamma = _damping_profile(r, thickness, factor, degree)
    return 1.0 - 1j * gamma


# ------------------------------------------------------------------
# 4.  Global matrix assembly (complex, per-Gauss-point)
# ------------------------------------------------------------------
def build_global_matrices(mesh, model: Dict[str, Any], omega: float):
    """
    Build complex stiffness (K) and mass (M) matrices with PML/ABC damping.
    Damping is applied **per Gauss point**, gamma ≤ 1.0.
    Returns: K_complex, M_complex, active_dof
    """
    nodes = mesh.coord
    tri6  = mesh.tri6
    nnode = nodes.shape[0]
    ndof  = 3 * nnode
    dom   = model['Model']
    layers = len(dom['DomainType'])

    # COO buffers
    I_K, J_K, V_K = [], [], []
    I_M, J_M, V_M = [], [], []

    # outer radius for PML/ABC
    rx = np.asarray(dom['DomainRx'])
    r_max = rx[-1]

    # read damping parameters from JSON
    pml_thick = float(dom.get('PML_thickness', 0.0))
    pml_factor = float(dom.get('PML_factor', 10.0))
    pml_degree = float(dom.get('PML_degree', 2.0))
    abc_thick = float(dom.get('ABC_thickness', 0.0))
    abc_factor = float(dom.get('ABC_factor', 0.1))
    abc_degree = float(dom.get('ABC_degree', 1.0))

    for el in tri6:
        xyz = nodes[el]                      # (6,2)
        ctr = xyz.mean(axis=0)
        r_ctr = np.hypot(ctr[0], ctr[1])

        # ---- identify layer by radius ----
        rx = np.asarray(dom['DomainRx'])
        layer = None
        for k in range(len(dom['DomainType'])):
            if k == 0:
                if r_ctr <= rx[0]:
                    layer = k
                    break
            else:
                if rx[k-1] <= r_ctr <= rx[k]:
                    layer = k
                    break
        if layer is None:
            layer = len(dom['DomainType']) - 1

        # ---- material matrix ----
        params = dom['DomainParam'][layer]
        dtype  = dom['DomainType'][layer].lower()
        rho = params[0]

        if dtype == 'fluid':
            vel = params[1]
            lam = rho * vel**2
            C = np.zeros((3, 3), dtype=complex)
            C[0, 0] = C[1, 1] = lam
            C[0, 1] = C[1, 0] = lam
            C[2, 2] = 0.0
        else:  # solid
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
            C = np.zeros((3, 3), dtype=complex)
            C[0, 0] = C_3d[0, 0]; C[0, 1] = C_3d[0, 1]; C[1, 0] = C_3d[1, 0]; C[1, 1] = C_3d[1, 1]; C[2, 2] = C_3d[5, 5]

        # ---- PML/ABC damping – per Gauss point ----
        qp, w = Tri6Element.qp, Tri6Element.w
        for xi, eta, wt in zip(qp[:, 0], qp[:, 1], w):
            N, dN = Tri6Element.shape(xi, eta)
            J = dN @ xyz
            detJ = np.linalg.det(J)
            if abs(detJ) < 1e-12:
                continue
            invJ = np.linalg.inv(J)
            dNxy = invJ @ dN

            # physical coordinates of Gauss point
            xy_gp = N @ xyz
            r_gp  = np.hypot(xy_gp[0], xy_gp[1])

            # PML/ABC masks and gamma for **this Gauss point**
            inside_pml = (r_gp > (r_max - pml_thick)) & (pml_thick > 0.0)
            inside_abc = (r_gp > (r_max - abc_thick)) & (abc_thick > 0.0) & (~inside_pml)

            gamma_pml = 0.0
            if inside_pml:
                sigma = min(((r_gp - (r_max - pml_thick)) / pml_thick) ** pml_degree, 1.0)
                gamma_pml = min(pml_factor * sigma, 1.0)        # ⩽ 1.0

            gamma_abc = 0.0
            if inside_abc:
                sigma = min(((r_gp - (r_max - abc_thick)) / abc_thick) ** abc_degree, 1.0)
                gamma_abc = min(abc_factor * sigma, 1.0)        # ⩽ 1.0

            # complex multiplier for **this Gauss point**
            damp_gp = 1.0 - 1j * (gamma_pml + gamma_abc)

            # elementary matrices with damping at this point
            Ke_gp, Me_gp = _elem_mat_gp(xyz, C, rho, omega, dNxy, detJ, wt, damp_gp)

            # ---- assembly into COO ----
            gdof = np.repeat(3 * el, 3) + np.tile([0, 1, 2], 6)
            for a in range(18):
                for b in range(18):
                    I_K.append(gdof[a]); J_K.append(gdof[b]); V_K.append(Ke_gp[a, b])
                    I_M.append(gdof[a]); J_M.append(gdof[b]); V_M.append(Me_gp[a, b])

    # ---- final complex matrices ----
    K = coo_matrix((V_K, (I_K, J_K)), shape=(ndof, ndof)).tocsr().astype(complex)
    M = coo_matrix((V_M, (I_M, J_M)), shape=(ndof, ndof)).tocsr().astype(complex)
    return K, M, np.arange(ndof)


# ------------------------------------------------------------------
# 5.  Element matrices per Gauss point (with damping)
# ------------------------------------------------------------------
def _elem_mat_gp(xyz: np.ndarray, C: np.ndarray, rho: float, omega: float,
                 dNxy: np.ndarray, detJ: float, wt: float, damp_gp: complex) -> Tuple[np.ndarray, np.ndarray]:
    """Return Ke, Me for single Tri6 GP with complex damping multiplier."""
    # strain-displacement B (3×18)
    B = np.zeros((3, 18), dtype=complex)
    for i in range(6):
        B[0, 3 * i] = dNxy[0, i]          # ε_xx
        B[1, 3 * i + 1] = dNxy[1, i]      # ε_yy
        B[2, 3 * i] = dNxy[1, i]          # γ_xy
        B[2, 3 * i + 1] = dNxy[0, i]

    # stiffness with damping
    Ke = wt * detJ * B.T @ (damp_gp * C) @ B

    # consistent mass (damping multiplier on density)
    Me = np.zeros((18, 18), dtype=complex)
    N = Tri6Element.shape(0.333, 0.333)[0]  # reuse shape at centroid for speed
    for i in range(6):
        for j in range(6):
            Mij = rho * wt * detJ * N[i] * N[j] * damp_gp
            for d in range(3):
                ii, jj = 3 * i + d, 3 * j + d
                Me[ii, jj] += Mij

    return Ke, Me