import numpy as np
from scipy.sparse import csr_matrix, coo_matrix


# ------------------------------------------------------------------
# 6-node triangle (Tri6) – shape functions & quadrature
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
            [4.0 * (xi + eta) - 3.0, 4.0 * xi - 1.0, 0.0, 4.0 * (1.0 - 2.0 * xi - eta), 4.0 * eta, -4.0 * eta],
            [4.0 * (xi + eta) - 3.0, 0.0, 4.0 * eta - 1.0, -4.0 * xi, 4.0 * xi, 4.0 * (1.0 - xi - 2.0 * eta)]
        ])
        return N, dN


# ------------------------------------------------------------------
# Rotation of elastic matrix (Bond transform) – plane-strain
# ------------------------------------------------------------------
def rot_cij(cij, theta):
    """Rotate 6×6 elastic matrix by angle theta (rad)."""
    # Ensure cij is a 6x6 matrix
    if isinstance(cij, list):
        cij = np.array(cij)
    c = cij.reshape(6, 6) if cij.size == 36 else cij

    # Rotation matrix for 6x6 stiffness matrix (Voigt notation)
    c_rad = np.radians(theta)  # Convert to radians
    c_theta, s_theta = np.cos(c_rad), np.sin(c_rad)
    c2, s2, cs = c_theta * c_theta, s_theta * s_theta, c_theta * s_theta

    R = np.zeros((6, 6))
    R[0, 0] = c2
    R[0, 1] = s2
    R[0, 3] = 2.0 * cs
    R[1, 0] = s2
    R[1, 1] = c2
    R[1, 3] = -2.0 * cs
    R[2, 2] = 1.0
    R[3, 0] = -cs
    R[3, 1] = cs
    R[3, 3] = c2 - s2
    R[4, 4] = c_theta
    R[4, 5] = s_theta
    R[5, 4] = -s_theta
    R[5, 5] = c_theta

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
    nodes = mesh.coord  # (nnod, 2)
    tri6 = mesh.tri6  # (nelem, 6)
    nnode = nodes.shape[0]
    ndof = 3 * nnode  # ux, uy, uz per node
    m = model['Model']
    layers = len(m['DomainType'])

    # COO buffers
    I_K, J_K, V_K = [], [], []
    I_M, J_M, V_M = [], [], []

    for el in tri6:
        xyz = nodes[el]  # element coordinates (6, 2)
        ctr = xyz.mean(axis=0)
        r_ctr = np.hypot(ctr[0], ctr[1])

        # ---- identify layer by radius ----
        rx = m['DomainRx']
        layer = None
        for k in range(layers):
            # Check if center is within this layer's radius
            if k == 0:
                if r_ctr <= rx[0]:
                    layer = k
                    break
            else:
                if rx[k - 1] <= r_ctr <= rx[k]:
                    layer = k
                    break
        if layer is None:
            layer = layers - 1

        # ---- material matrix ----
        domain_params = m['DomainParam'][layer]
        domain_type = m['DomainType'][layer]

        if domain_type.lower() == 'fluid':
            # Fluid parameters: [density, velocity]
            rho, vel = domain_params
            lam = rho * vel ** 2  # Lambda parameter for fluid

            # Create proper fluid stiffness matrix (3x3 for plane strain)
            C = np.zeros((3, 3), dtype=complex)
            C[0, 0] = lam
            C[1, 1] = lam
            C[0, 1] = lam
            C[1, 0] = lam
            # No rotation for fluid
        else:  # HTTI or other solid
            # Solid parameters: [density, c11, c13, c33, c44, c66, theta, ...]
            rho = domain_params[0]
            c11, c13, c33, c44, c66 = domain_params[1:6]
            theta = domain_params[6] if len(domain_params) > 6 else 0.0

            # Create stiffness matrix for transverse isotropy (6x6 in 3D)
            C_3d = np.array([
                [c11, c13, c13, 0.0, 0.0, 0.0],
                [c13, c33, c13, 0.0, 0.0, 0.0],
                [c13, c13, c33, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, c44, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, c44, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, c66]
            ], dtype=complex)

            # Apply rotation if needed
            if abs(theta) > 1e-10:
                C_3d = rot_cij(C_3d, theta)

            # Reduce to 3x3 for plane strain (assuming z-direction is axis of symmetry)
            C = np.zeros((3, 3), dtype=complex)
            C[0, 0] = C_3d[0, 0]  # c11
            C[0, 1] = C_3d[0, 1]  # c13
            C[1, 0] = C_3d[1, 0]  # c13
            C[1, 1] = C_3d[1, 1]  # c33
            C[2, 2] = C_3d[5, 5]  # c66

        # ---- PML stretching (radial) ----
        if 'PML_factor' in m and r_ctr > rx[-2]:
            dr = r_ctr - rx[-2]
            L = rx[-1] - rx[-2]
            s = 1.0 + 1j * m['PML_factor'] * (dr / L) ** m['PML_degree']
            C = C / s  # Apply PML stretching to stiffness matrix

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
    Ke = np.zeros((18, 18), dtype=complex)
    Me = np.zeros((18, 18), dtype=complex)

    for xi_eta, wt in zip(qp, w):
        N, dN = Tri6Element.shape(xi_eta[0], xi_eta[1])
        J = dN @ xyz  # 2×2 Jacobian
        detJ = np.linalg.det(J)

        if abs(detJ) < 1e-12:
            continue  # Skip degenerate elements

        invJ = np.linalg.inv(J)
        dNxy = invJ @ dN  # 2×6 derivatives w.r.t x,y

        # strain-displacement B matrix (3×18)
        B = np.zeros((3, 18))
        for i in range(6):
            B[0, 3 * i] = dNxy[0, i]  # ε_xx
            B[1, 3 * i + 1] = dNxy[1, i]  # ε_yy
            B[2, 3 * i] = dNxy[1, i]  # γ_xy
            B[2, 3 * i + 1] = dNxy[0, i]

        Ke += wt * B.T @ C @ B * detJ

        # consistent mass matrix
        for i in range(6):
            for j in range(6):
                M_ij = rho * wt * N[i] * N[j] * detJ
                for d in range(3):  # ux, uy, uz
                    idx_i = 3 * i + d
                    idx_j = 3 * j + d
                    Me[idx_i, idx_j] += M_ij

    return Ke, Me