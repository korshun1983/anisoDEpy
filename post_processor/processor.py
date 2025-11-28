# post_processor/processor.py
"""
Stage-2 post-processing (analogue of MATLAB proc_aniso_TE.m)
Input  – already loaded arrays, output – dict + optional Results-N.npz
"""
from __future__ import annotations
import numpy as np
from scipy.interpolate import griddata
from pathlib import Path
from typing import Dict, Tuple, Optional

__all__ = ["TEProcessor"]


class TEProcessor:
    """
    1. Frequency / slowness curves  (Hz and s/m)
    2. Interpolate onto cylinder -> ur, uphi, uz
    3. Kinetic-energy density
    4. Azimuthal Fourier decomposition
    5. Save / load Results-N.npz
    """

    def __init__(self,
                 nodes: np.ndarray,        # (N,2)  full node list
                 eigen_vals: np.ndarray,   # (K,)
                 eigen_vecs: np.ndarray,   # (Ndof,K)  active DOFs only
                 active_dof: Optional[np.ndarray] = None
                 ):
        self.nodes = nodes
        self.eigen_vals = eigen_vals
        self.eigen_vecs = eigen_vecs
        # if no mask given – assume all nodes are active
        self.active_dof = active_dof if active_dof is not None else np.arange(nodes.shape[0])
        self.active_coords = nodes[self.active_dof]          # (Ndof,2)

    # -------------------------------------------------
    # 1. Frequency & slowness  (Hz, s/m)
    # -------------------------------------------------
    def _freq_slowness(self) -> Dict[str, np.ndarray]:
        freq_hz = self.eigen_vals / (2 * np.pi)      # rad/s -> Hz
        return {"freq": freq_hz,
                "slowness": 1 / self.eigen_vals}     # s/m

    # -------------------------------------------------
    # 2. Cylindrical grid and field components
    # -------------------------------------------------
    def _cylinder_field(self,
                       r_lim: Tuple[float, float],
                       nr: int,
                       nphi: int) -> Dict[str, np.ndarray]:
        r = np.linspace(*r_lim, nr)
        phi = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
        R, Phi = np.meshgrid(r, phi, indexing='ij')   # (nr,nphi)
        X, Y = R * np.cos(Phi), R * np.sin(Phi)

        K = self.eigen_vecs.shape[1]
        ur   = np.zeros((nr, nphi, K), dtype=complex)
        uphi = np.zeros_like(ur)
        uz   = np.zeros_like(ur)

        for k in range(K):
            u = self.eigen_vecs[:, k]                   # (Ndof,)
            ur_real = griddata(self.active_coords, u.real, (X, Y),
                               method='linear', fill_value=0.)
            ur_imag = griddata(self.active_coords, u.imag, (X, Y),
                               method='linear', fill_value=0.)
            ur[:, :, k] = ur_real + 1j * ur_imag
            # uphi, uz – stubs for future vector field
        return {"R": R, "Phi": Phi, "X": X, "Y": Y,
                "ur": ur, "uphi": uphi, "uz": uz}

    # -------------------------------------------------
    # 3. Kinetic-energy density
    # -------------------------------------------------
    def _energy_fourier(self,
                        cyl: Dict[str, np.ndarray],
                        mode: int = 0,
                        r_idx: int = 10) -> Dict[str, np.ndarray]:
        ur = cyl["ur"]  # (nr,nphi,K)
        rho = 1.0
        ke = 0.5 * rho * np.abs(ur) ** 2  # (nr,nphi,K)  <-- full cube
        coeffs_list = []
        harmonics_list = []
        for k in range(ur.shape[2]):  # loop over modes
            u_phi = ur[r_idx, :, k]
            coeffs = np.fft.fft(u_phi)
            coeffs_list.append(coeffs)
            harmonics_list.append(np.fft.fftfreq(u_phi.size, d=1.0))

        return {"KE": ke,  # 3-D
                "fourier_coeffs": np.stack(coeffs_list, axis=1),
                "fourier_harmonics": np.array(harmonics_list[0])}

    # -------------------------------------------------
    # 4. Main public API
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

    # -------------------------------------------------
    # 5. Save / load  (analogue of Results-N.mat)
    # -------------------------------------------------
    def save_npz(self, file_path: Path) -> None:
        """Save full results dict into Results-N.npz."""
        data = self.run()
        np.savez(file_path, **data)
        print(f"[TEProcessor] Saved {file_path}")

    @staticmethod
    def load_npz(file_path: Path) -> Dict[str, np.ndarray]:
        """Load previously saved Results-N.npz."""
        with np.load(file_path, allow_pickle=True) as src:
            return dict(src)