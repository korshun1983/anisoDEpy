import numpy as np
def compute_radial_energy(v, mesh, model, omega, r_grid):
    """
    Compute radial kinetic-energy density TE(r) on the user-supplied 1-D grid.
    Parameters
    ----------
    v : ndarray
        Eigen-vector (3*nnod,) – already in physical ordering.
    mesh : MeshContainer
    model : dict
        Full JSON model.
    omega : float
        Angular frequency [rad/s].
    r_grid : ndarray
        Monotonic 1-D array where TE(r) will be returned.

    Returns
    -------
    TE : ndarray
        Same length as r_grid [J·m⁻¹].
    """
    nodes = mesh.coord  # (nnod, 2)
    tri6 = mesh.tri6  # (nelem, 6)
    layers = len(model['Model']['DomainType'])

    # ---------- build layer map (rx boundaries) ----------
    rx = model['Model']['DomainRx']

    # ---------- helpers ----------
    w = omega
    TE = np.zeros_like(r_grid)

    for el in tri6:
        xyz = nodes[el]  # (6, 2)
        ctr = xyz.mean(axis=0)
        r_ctr = np.hypot(ctr[0], ctr[1])

        # ---- find layer by radius ----
        layer = None
        for k in range(layers):
            if r_ctr >= rx[k] and r_ctr <= rx[k + 1]:
                layer = k
                break
        if layer is None:  # fallback
            layer = layers - 1

        # ---- material of this layer ----
        if model['Model']['DomainType'][layer] == 'fluid':
            rho = model['Model']['DomainParam'][layer][0] * 1000
        else:  # HTTI
            rho = model['Model']['DomainParam'][layer][0] * 1000

        # ---- element area (exact for curved Tri6) ----
        # cheap approximation: use corner triangle
        n0, n1, n2 = xyz[0], xyz[1], xyz[2]
        area = 0.5 * abs(np.cross(n1 - n0, n2 - n0))

        # ---- kinetic energy of element ----
        gdof = np.repeat(3 * el, 3) + np.tile([0, 1, 2], 6)  # 18×1
        ve = v[gdof].reshape(-1, 3)  # (6,3)
        ke_elem = 0.5 * rho * w ** 2 * np.sum(np.abs(ve) ** 2) * area  # [J]

        # ---- distribute ke_elem to r_grid linearly ----
        r_nodes = np.hypot(xyz[:, 0], xyz[:, 1])
        r_min, r_max = r_nodes.min(), r_nodes.max()
        if r_max <= r_grid[0] or r_min >= r_grid[-1]:
            continue

        # find affected bins
        left = np.searchsorted(r_grid, r_min, side='left')
        right = np.searchsorted(r_grid, r_max, side='right')
        if left >= right:  # fully inside one bin
            TE[left] += ke_elem / (r_grid[left + 1] - r_grid[left])
        else:
            # linear kernel (trapezoidal)
            for i in range(left, right):
                w1 = max(0, r_grid[i + 1] - r_min) / (r_max - r_min)
                w2 = max(0, r_max - r_grid[i]) / (r_max - r_min)
                w_lin = 0.5 * (w1 + w2)
                TE[i] += ke_elem * w_lin / (r_grid[i + 1] - r_grid[i])

    return TE