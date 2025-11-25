import numpy as np
from scipy.sparse import csr_matrix, coo_matrix


# ------------------------------------------------------------------
# 6-node triangle (Tri6) – shape functions & quadrature
# ------------------------------------------------------------------
class Tri6Element:
    """Quadratic 6-node triangle – shape functions and derivatives."""
    "qp and w are the quadrature points and weights used for numerical integration over the triangle (Tri6)."
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
    def shape(xi, eta):
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
            [-3.0 + 4.0 * z,  4.0 * xi - 1.0,  0.0,  4.0 * (z - xi),   4.0 * eta,        -4.0 * eta],
            [-3.0 + 4.0 * z,  0.0,             4.0 * eta - 1.0,  -4.0 * xi,  4.0 * xi,  4.0 * (z - eta)]
        ])
        return N, dN


# ------------------------------------------------------------------
# Rotation of elastic matrix (Bond transform) – plane-strain
# ------------------------------------------------------------------
def rot_cij(cij, theta):
    """Rotate 6×6 elastic matrix by angle theta (rad)."""
    c = np.array(cij).reshape(6, 6)
    R = np.zeros((6, 6))
    c, s = np.cos(theta), np.sin(theta)
    c2, s2, cs = c * c, s * s, c * s

    R[0, 0] = R[1, 1] = c2
    R[0, 1] = s2
    R[1, 0] = s2
    R[0, 2] = R[2, 0] = cs
    R[1, 2] = R[2, 1] = -cs
    R[2, 2] = c2 - s2
    R[3, 3] = R[4, 4] = c
    R[3, 4] = s
    R[4, 3] = -s
    R[5, 5] = 1.0
    return R @ c @ R.T


# ------------------------------------------------------------------
# Global matrix assembly
# ------------------------------------------------------------------
def build_global_matrices(mesh, model, omega):
    """
    Assemble stiffness (K) and mass (M) matrices for all layers
    (anisotropic solids + fluid, with PML stretching).
    Returns sparse CSR matrices.
    """
    nodes = mesh.coord          # (nnod, 2)
    tri6  = mesh.tri6           # (nelem, 6)
    nnode = nodes.shape[0]
    ndof  = 3 * nnode           # ux, uy, uz per node
    m     = model['Model']
    layers = len(m['DomainType'])

    # COO buffers
    I_K, J_K, V_K = [], [], []
    I_M, J_M, V_M = [], [], []

    for el in tri6:
        xyz = nodes[el]                      # element coordinates (6, 2)
        ctr = xyz.mean(axis=0)
        r_ctr = np.hypot(ctr[0], ctr[1])

        # ---- identify layer by radius ----
        rx = m['DomainRx']
        layer = None
        for k in range(layers):
            if rx[k] <= r_ctr <= rx[k + 1]:
                layer = k
                break
        if layer is None:
            layer = layers - 1

        # ---- material matrix ----
        if m['DomainType'][layer] == 'fluid':
            rho, lam = m['DomainParam'][layer]
            C = np.diag([lam, lam, 0.0, 0.0, 0.0, 0.0])
        else:  # HTTI
            par = m['DomainParam'][layer]
            rho, c11, c13, c33, c44, c66, theta = par[0], *par[1:7]
            C = np.array([
                [c11, c13, 0.0, 0.0, 0.0, 0.0],
                [c13, c33, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, c66, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, c44, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, c44, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, c66]
            ])
            C = rot_cij(C, theta)

        # ---- PML stretching (radial) ----
        s = 1.0
        if 'PML_factor' in m and r_ctr > rx[-2]:
            dr = r_ctr - rx[-2]
            L  = rx[-1] - rx[-2]
            s  = 1.0 + 1j * m['PML_factor'] * (dr / L) ** m['PML_degree']
        C = C / s

        # ---- elementary matrices ----
        Ke, Me = elem_mat(xyz, C, rho, omega)

        # ---- assembly into COO ----
        gdof = np.repeat(3 * el, 3) + np.tile([0, 1, 2], 6)  # 18 global DOFs
        for a in range(18):
            for b in range(18):
                I_K.append(gdof[a])
                J_K.append(gdof[b])
                V_K.append(Ke[a, b])
                I_M.append(gdof[a])
                J_M.append(gdof[b])
                V_M.append(Me[a, b])

    K = coo_matrix((V_K, (I_K, J_K)), shape=(ndof, ndof)).tocsr()
    M = coo_matrix((V_M, (I_M, J_M)), shape=(ndof, ndof)).tocsr()
    return K, M, np.arange(ndof)


# ------------------------------------------------------------------
# Element stiffness & mass (18 × 18)
# ------------------------------------------------------------------
def elem_mat(xyz, C, rho, omega):
    """Return Ke, Me for a single Tri6 element."""
    qp, w = Tri6Element.qp, Tri6Element.w
    Ke = np.zeros((18, 18))
    Me = np.zeros((18, 18))

    for xi_eta, wt in zip(qp, w):
        N, dN = Tri6Element.shape(xi_eta[0], xi_eta[1])
        J = dN @ xyz                                # 2×2 Jacobian
        detJ = np.linalg.det(J)
        invJ = np.linalg.inv(J)
        dNxy = invJ @ dN                            # 2×6 derivatives w.r.t x,y

        # strain-displacement B matrix (3×18)
        B = np.zeros((3, 18))
        for i in range(6):
            B[0, 3 * i]     = dNxy[0, i]   # ε_xx
            B[1, 3 * i + 1] = dNxy[1, i]   # ε_yy
            B[2, 3 * i]     = dNxy[1, i]   # γ_xy
            B[2, 3 * i + 1] = dNxy[0, i]

        Ke += wt * B.T @ C @ B * detJ

        # consistent mass lumped for simplicity
        mass = rho * wt * detJ
        for i in range(6):
            idx = [3 * i, 3 * i + 1, 3 * i + 2]
            Me[np.ix_(idx, idx)] += mass * np.outer(N[i], N[i])

    return Ke, Me