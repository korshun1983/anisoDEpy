"""
Stage-2 post-processing (analogue of MATLAB proc_aniso_TE.m)
Input  – already loaded arrays, output – dict with results.
"""
from __future__ import annotations
import numpy as np
from scipy.interpolate import griddata
from typing import Dict, Tuple

__all__ = ["TEProcessor"]

class TEProcessor:
    """
    1. Frequency / slowness curves
    2. Interpolate onto a cylinder -> ur, uphi, uz
    3. Kinetic energy + azimuthal Fourier transform
    """
    def __init__(self,
                 nodes: np.ndarray,        # (N,2)  x,y
                 eigen_vals: np.ndarray,   # (K,)   ω
                 eigen_vecs: np.ndarray    # (N,K)  complex
                 ):
        self.nodes = nodes
        self.eigen_vals = eigen_vals
        self.eigen_vecs = eigen_vecs

    # -------------------------------------------------
    # 1. Frequency & slowness curves
    # -------------------------------------------------
    def _freq_slowness(self) -> Dict[str, np.ndarray]:
        # optional: interpolate to a user-defined frequency grid
        return {"freq": self.eigen_vals,
                "slowness": 1 / self.eigen_vals}   # s/m

    # -------------------------------------------------
    # 2. Cylindrical grid and field components
    # -------------------------------------------------
    def _cylinder_field(self,
                       r_lim: Tuple[float, float],
                       nr: int,
                       nphi: int) -> Dict[str, np.ndarray]:
        r = np.linspace(*r_lim, nr)
        phi = np.linspace(0, 2*np.pi, nphi, endpoint=False)
        R, Phi = np.meshgrid(r, phi, indexing='ij')   # (nr,nphi)
        X, Y = R*np.cos(Phi), R*np.sin(Phi)

        K = self.eigen_vecs.shape[1]
        ur  = np.zeros((nr, nphi, K), dtype=complex)
        uphi = np.zeros_like(ur)
        uz   = np.zeros_like(ur)

        for k in range(K):
            u = self.eigen_vecs[:, k]
            ur_real = griddata(self.nodes, u.real, (X, Y),
                              method='linear', fill_value=0.)
            ur_imag = griddata(self.nodes, u.imag, (X, Y),
                              method='linear', fill_value=0.)
            ur[:, :, k] = ur_real + 1j*ur_imag
            # uphi, uz – stubs; add similarly if data available
        return {"R": R, "Phi": Phi, "X": X, "Y": Y,
                "ur": ur, "uphi": uphi, "uz": uz}

    # -------------------------------------------------
    # 3. Energy + Fourier harmonics
    # -------------------------------------------------
    def _energy_fourier(self,
                       cyl: Dict[str, np.ndarray],
                       mode: int = 0,
                       r_idx: int = 10) -> Dict[str, np.ndarray]:
        ur = cyl["ur"]                      # (nr,nphi,K)
        rho = 1.0                           # may be replaced by a density profile
        ke = 0.5*rho*np.abs(ur[:, :, mode])**2   # (nr,nphi)

        u_phi = ur[r_idx, :, mode]          # azimuthal slice
        coeffs = np.fft.fft(u_phi)
        harmonics = np.fft.fftfreq(u_phi.size, d=1.0)
        return {"KE": ke,
                "fourier_coeffs": coeffs,
                "fourier_harmonics": harmonics}

    # -------------------------------------------------
    # Single public entry point
    # -------------------------------------------------
    def run(self,
           r_lim: Tuple[float, float] = (0.05, 2.0),
           nr: int = 50,
           nphi: int = 64,
           mode: int = 0,
           r_idx: int = 10) -> Dict[str, np.ndarray]:
        stage1 = self._freq_slowness()
        stage2 = self._cylinder_field(r_lim, nr, nphi)
        stage3 = self._energy_fourier(stage2, mode, r_idx)
        return {**stage1, **stage2, **stage3}