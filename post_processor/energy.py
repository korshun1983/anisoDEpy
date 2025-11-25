import numpy as np


def compute_radial_energy(v, mesh, model, omega, r_grid):
    """
    Project eigen-vector on Tri6, integrate kinetic-energy density
    over a user-supplied 1-D radial grid -> TE(r) [J·m⁻¹].
    """
    nodes = mesh.coord
    tri6  = mesh.tri6
    m     = model['Model']
    layers = len(m['DomainType'])
    rx    = m['DomainRx']          # radial layer bounds

    w  = omega
    TE = np.zeros_like(r_grid)

    for el in tri6:
        xyz   = nodes[el]
        ctr   = xyz.mean(axis=0)
        r_ctr = np.hypot(ctr[0], ctr[1])

        # ---- find layer that owns this element ----
        layer = None
        for k in range(layers):
            if rx[k] <= r_ctr <= rx[k + 1]:
                layer = k
                break
        if layer is None:
            layer = layers - 1

        # ---- density of CURRENT layer ----
        if m['DomainType'][layer] == 'fluid':
            rho = m['DomainParam'][layer][0] * 1000
        else:
            rho = m['DomainParam'][layer][0] * 1000

        # ---- element area (corner triangle) ----
        n0, n1, n2 = xyz[0], xyz[1], xyz[2]
        area = 0.5 * abs(np.cross(n1 - n0, n2 - n0))

        # ---- kinetic energy of element ----
        gdof = np.repeat(3 * el, 3) + np.tile([0, 1, 2], 6)
        ve   = v[gdof].reshape(-1, 3)
        ke_elem = 0.5 * rho * w**2 * np.sum(np.abs(ve)**2) * area

        # ---- distribute to r_grid (linear kernel) ----
        r_nodes = np.hypot(xyz[:, 0], xyz[:, 1])
        r_min, r_max = r_nodes.min(), r_nodes.max()
        if r_max <= r_grid[0] or r_min >= r_grid[-1]:
            continue

        left  = np.searchsorted(r_grid, r_min, side='left')
        right = np.searchsorted(r_grid, r_max, side='right')
        for i in range(left, right):
            w_lin = (min(r_grid[i + 1], r_max) - max(r_grid[i], r_min)) / (r_max - r_min)
            TE[i] += ke_elem * w_lin / (r_grid[i + 1] - r_grid[i])

    return TE