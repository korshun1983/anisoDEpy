# post_processor/interpreter.py
"""
Stage-3 interpretation (analogue of MATLAB intr_aniso_TE.m)
Input  – list[dict] produced by TEProcessor.run() for each frequency
Output – figures + dict with classified curves
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

__all__ = ["TEInterpreter"]


class TEInterpreter:
    """
    1. Frequency / slowness curves
    2. Mode classification (Stoneley, fast, slow, etc.)
    3. Polarisation & symmetry checks
    4. Plotting helpers
    """

    def __init__(self, results: List[Dict[str, np.ndarray]]) -> None:
        """
        results[i] – dict returned by TEProcessor.run() for i-th frequency
        """
        self.results = results
        self.freqs = np.array([r["freq"][0] for r in results])  # kHz
        self.nmodes = results[0]["freq"].size

    # -------------------------------------------------
    # 1. Build continuous frequency-slowness arrays
    # -------------------------------------------------
    def _assemble_dispersion(self) -> Dict[str, np.ndarray]:
        """Return (freq, slowness) matrices (n_freq, n_modes)."""
        slowness = np.stack([r["slowness"] for r in self.results], axis=0)
        return {"freq": self.freqs, "slowness": slowness}

    # -------------------------------------------------
    # 2. Very simple mode classifier
    # -------------------------------------------------
    def _classify_modes(self, dispersion: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Labels: 0 = Stoneley, 1 = fast, 2 = slow, 3 = other
        Heuristic: lowest slowness -> Stoneley, then 33/66 % quantiles
        """
        slo = dispersion["slowness"]  # (n_freq, n_modes)
        labels = np.zeros_like(slo, dtype=int)

        for f_idx in range(slo.shape[0]):
            s = slo[f_idx, :]
            stoneley_thresh = np.min(s) * 1.05
            q33, q66 = np.quantile(s, [0.33, 0.66])
            labels[f_idx, :] = np.where(s < stoneley_thresh, 0,
                                       np.where(s < q33, 1,
                                               np.where(s < q66, 2, 3)))
        return labels

    # -------------------------------------------------
    # 3. Plot slowness curves
    # -------------------------------------------------
    def plot_slowness(self, ax: plt.Axes | None = None,
                     show: bool = True) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4))

        dispersion = self._assemble_dispersion()
        labels = self._classify_modes(dispersion)

        colors = ["k", "r", "b", "g"]
        names = ["Stoneley", "Fast", "Slow", "Other"]

        for m in range(self.nmodes):
            ax.plot(dispersion["freq"],
                   dispersion["slowness"][:, m] * 1e3,  # s/km
                   color=colors[labels[0, m]],
                   lw=1.5, label=names[labels[0, m]] if m == 0 else "")

        ax.set_xlabel("Frequency (kHz)")
        ax.set_ylabel("Slowness (s/km)")
        ax.set_title("Dispersion curves")
        ax.legend()
        ax.grid(alpha=0.3)
        if show:
            plt.show()
        return ax

    # -------------------------------------------------
    # 4. Kinetic-energy map for chosen mode & freq
    # -------------------------------------------------
    def plot_energy_map(self, freq_idx: int = 0, mode: int = 0,
                       ax: plt.Axes | None = None,
                       show: bool = True) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots(subplot_kw=dict(projection='polar'))

        r = self.results[freq_idx]["R"][:, 0]
        phi = self.results[freq_idx]["Phi"][0, :]
        ke = self.results[freq_idx]["KE"][:, :, mode]

        pcol = ax.pcolormesh(phi, r, ke, shading='auto')
        ax.set_title(f"KE density  f={self.freqs[freq_idx]:.1f} kHz  mode={mode}")
        plt.colorbar(pcol, ax=ax)
        if show:
            plt.show()
        return ax

    # -------------------------------------------------
    # 5. High-level API: run all & return summary
    # -------------------------------------------------
    def run(self) -> Dict[str, np.ndarray]:
        dispersion = self._assemble_dispersion()
        labels = self._classify_modes(dispersion)
        self.plot_slowness(show=False)   # figure created internally
        return {"dispersion": dispersion, "labels": labels}