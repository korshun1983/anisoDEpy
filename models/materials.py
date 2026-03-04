import numpy as np

class Fluid:
    def __init__(self, rho, c):
        self.rho = rho
        self.c = c
        self.lambda_ = rho * c**2   # модуль объёмного сжатия

    def __repr__(self):
        return f"Fluid(rho={self.rho}, c={self.c})"


class IsotropicSolid:
    def __init__(self, rho, E, nu):
        self.rho = rho
        self.E = E
        self.nu = nu
        self.mu = E / (2 * (1 + nu))
        self.lambda_ = E * nu / ((1 + nu) * (1 - 2 * nu))

    def __repr__(self):
        return f"IsotropicSolid(rho={self.rho}, E={self.E}, nu={self.nu})"


class HTTI:
    def __init__(self, rho, c11, c13, c33, c44, c66, dip=0.0, azimuth=0.0):
        self.rho = rho
        self.params_vti = np.array([c11, c13, c33, c44, c66])
        self.dip = dip
        self.azimuth = azimuth
        self.c_ij_vti = self._vti_tensor(c11, c13, c33, c44, c66)
        self.c_ij_global = self._rotate_to_global(self.c_ij_vti, dip, azimuth)

    def _vti_tensor(self, c11, c13, c33, c44, c66):
        factor = 1e9
        c = np.zeros((6, 6))
        c[0,0] = c11 * factor
        c[1,1] = c11 * factor
        c[2,2] = c33 * factor
        c[3,3] = c44 * factor
        c[4,4] = c44 * factor
        c[5,5] = c66 * factor
        c[0,1] = (c11 - 2*c66) * factor
        c[1,0] = c[0,1]
        c[0,2] = c13 * factor
        c[2,0] = c[0,2]
        c[1,2] = c13 * factor
        c[2,1] = c[1,2]
        return c

    def _rotation_matrix(self, dip, azimuth):
        ca, sa = np.cos(azimuth), np.sin(azimuth)
        cd, sd = np.cos(dip), np.sin(dip)
        return np.array([
            [ ca,           sa,           0],
            [-cd*sa,        cd*ca,        sd],
            [ sd*sa,       -sd*ca,        cd]
        ])

    def _rotate_stiffness(self, C, a):
        M = np.zeros((6,6))
        # строки 1-3
        M[0,0] = a[0,0]**2; M[0,1] = a[0,1]**2; M[0,2] = a[0,2]**2
        M[0,3] = 2*a[0,1]*a[0,2]; M[0,4] = 2*a[0,0]*a[0,2]; M[0,5] = 2*a[0,0]*a[0,1]
        M[1,0] = a[1,0]**2; M[1,1] = a[1,1]**2; M[1,2] = a[1,2]**2
        M[1,3] = 2*a[1,1]*a[1,2]; M[1,4] = 2*a[1,0]*a[1,2]; M[1,5] = 2*a[1,0]*a[1,1]
        M[2,0] = a[2,0]**2; M[2,1] = a[2,1]**2; M[2,2] = a[2,2]**2
        M[2,3] = 2*a[2,1]*a[2,2]; M[2,4] = 2*a[2,0]*a[2,2]; M[2,5] = 2*a[2,0]*a[2,1]
        # строка 4
        M[3,0] = a[1,0]*a[2,0]; M[3,1] = a[1,1]*a[2,1]; M[3,2] = a[1,2]*a[2,2]
        M[3,3] = a[1,1]*a[2,2] + a[1,2]*a[2,1]
        M[3,4] = a[1,0]*a[2,2] + a[1,2]*a[2,0]
        M[3,5] = a[1,0]*a[2,1] + a[1,1]*a[2,0]
        # строка 5
        M[4,0] = a[0,0]*a[2,0]; M[4,1] = a[0,1]*a[2,1]; M[4,2] = a[0,2]*a[2,2]
        M[4,3] = a[0,1]*a[2,2] + a[0,2]*a[2,1]
        M[4,4] = a[0,0]*a[2,2] + a[0,2]*a[2,0]
        M[4,5] = a[0,0]*a[2,1] + a[0,1]*a[2,0]
        # строка 6
        M[5,0] = a[0,0]*a[1,0]; M[5,1] = a[0,1]*a[1,1]; M[5,2] = a[0,2]*a[1,2]
        M[5,3] = a[0,1]*a[1,2] + a[0,2]*a[1,1]
        M[5,4] = a[0,0]*a[1,2] + a[0,2]*a[1,0]
        M[5,5] = a[0,0]*a[1,1] + a[0,1]*a[1,0]
        return M @ C @ M.T

    def _rotate_to_global(self, C_vti, dip, azimuth):
        a = self._rotation_matrix(dip, azimuth)
        return self._rotate_stiffness(C_vti, a)

    def get_stiffness(self):
        return self.c_ij_global

    def __repr__(self):
        return (f"HTTI(rho={self.rho}, c11={self.params_vti[0]}, c13={self.params_vti[1]}, "
                f"c33={self.params_vti[2]}, c44={self.params_vti[3]}, c66={self.params_vti[4]}, "
                f"dip={self.dip}, azimuth={self.azimuth})")