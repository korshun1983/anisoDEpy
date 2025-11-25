import numpy as np
from scipy.sparse import csr_matrix


def apply_boundary(K, M, dofmap, mesh, model):
    """
    Apply boundary conditions and return modified K, M, dofmap.
    Current implementations:
      1) rigid (clamped) outer edge (physical tag 1002)
      2) optional fluid-dof condensation (placeholder)
      3) optional symmetry/anti-symmetry (placeholder)
    """
    # 1) clamp outer edge ----------------------------------------------
    outer_nodes = _nodes_on_phys_line(mesh, tag=1002)
    clamped_dofs = np.repeat(3 * outer_nodes, 3) + np.tile([0, 1, 2], outer_nodes.size)

    K = K.tocsr()
    M = M.tocsr()
    for dof in clamped_dofs:
        K[dof, :] = 0.0
        K[:, dof] = 0.0
        K[dof, dof] = 1.0
        M[dof, :] = 0.0
        M[:, dof] = 0.0
        M[dof, dof] = 1.0

    # 2) fluid condensation (example) ----------------------------------
    # if layer 0 is fluid, keep pressure only and drop uy, uz
    # placeholder:  K, M, dofmap = _condense_fluid(K, M, dofmap, mesh, model)

    # 3) symmetry / anti-symmetry --------------------------------------
    # placeholder:  K, M, dofmap = _apply_symmetry(K, M, dofmap, mesh, model)

    return K.tocsr(), M.tocsr(), dofmap


# ------------------------------------------------------------------
# helper: nodes lying on a given physical line (emulated)
# ------------------------------------------------------------------
def _nodes_on_phys_line(mesh, tag=1002):
    """
    Return node indices that lie on physical line `tag`.
    Emulated here: pick nodes whose radius ≈ outermost value.
    """
    rx = mesh.coord[:, 0]
    ry = mesh.coord[:, 1]
    r = np.hypot(rx, ry)
    r_max = r.max()
    tol = 1e-3 * (r_max - r.min())
    return np.where(np.abs(r - r_max) < tol)[0]