# post_processor/interpreter.py
"""
Stage-3 interpretation (analogue of MATLAB intr_aniso_TE.m)
Input  – list[dict] produced by TEProcessor.run() for each frequency
Output – classified dispersion curves + optional figures / files
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Optional

__all__ = ["TEInterpreter"]


class TEInterpreter:
    """
    1. Frequency / slowness curves
    2. Mode classification (Stoneley, fast, slow, other)
    3. Symmetry & polarisation criteria (MATLAB options ported)
    4. Automatic mode swapping (1 ↔ 2) when branches cross
    5. Export figures / npz
    """

    def __init__(self,
                 results: List[Dict[str, np.ndarray]],
                 nodes: np.ndarray,
                 pml_mask: Optional[np.ndarray] = None,
                 adj_mask: Optional[np.ndarray] = None) -> None:
        """
        results   – list of dicts from TEProcessor.run()
        pml_mask  – boolean 1-D array: True for nodes inside PML region
        adj_mask  – boolean 1-D array: True for nodes in layer adjacent to PML
        """
        self.results = results
        self.freqs = np.array([r["freq"][0] for r in results])  # Hz
        self.nmodes = results[0]["freq"].size
        self.nodes = nodes
        self.pml_mask = pml_mask
        self.adj_mask = adj_mask

        # Classification parameters (MATLAB defaults)
        self.max_harm_limit = 0.30
        self.te_pml_total_limit = 0.15
        self.use_class_te_nt = False
        self.use_class_abcpml_htti = True

    # -------------------------------------------------
    # 1. Assemble frequency-slowness matrix
    # -------------------------------------------------
    def _assemble_dispersion(self) -> Dict[str, np.ndarray]:
        slowness = np.stack([r["slowness"] for r in self.results], axis=0)  # (n_freq, n_modes)
        velocity = 1 / slowness
        return {"freq": self.freqs, "slowness": slowness, "velocity": velocity}

    # -------------------------------------------------
    # 2. Energy ratio criteria (MATLAB options)
    # -------------------------------------------------
    def _energy_ratios(self, freq_idx: int, mode: int) -> Tuple[float, float]:
        """Return (TE_PML/TE_total, max(TE_PML)/max(TE_adj))"""
        if self.pml_mask is None or self.adj_mask is None:
            return 0.0, 0.0

        ke = self.results[freq_idx]["KE"][:, :, mode]  # (nr, nphi)
        ke_flat = ke.ravel()

        # radial grid masks
        r = self.results[freq_idx]["R"][:, 0]
        phi = self.results[freq_idx]["Phi"][0, :]
        X, Y = self.results[freq_idx]["X"], self.results[freq_idx]["Y"]

        # interpolate masks to the same grid
        pml_grid = np.zeros_like(X, dtype=bool)
        adj_grid = np.zeros_like(X, dtype=bool)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                # nearest-node lookup (fast enough)
                dist = (self.nodes[:, 0] - X[i, j])**2 + (self.nodes[:, 1] - Y[i, j])**2
                nn = int(np.argmin(dist))
                pml_grid[i, j] = self.pml_mask[nn]
                adj_grid[i, j] = self.adj_mask[nn]

        te_total = np.sum(ke_flat)
        te_pml = np.sum(ke_flat[pml_grid.ravel()])
        ratio1 = (te_pml / te_total) if te_total > 0 else 0.0

        te_adj = ke_flat[adj_grid.ravel()]
        ratio2 = (np.max(ke_flat[pml_grid.ravel()]) / np.max(te_adj)) if te_adj.size else 0.0
        return ratio1, ratio2

    # -------------------------------------------------
    # 3. Symmetry classification via Fourier harmonics
    # -------------------------------------------------
    def _symmetry_label(self, freq_idx: int, mode: int) -> int:
        coeffs = self.results[freq_idx]["fourier_coeffs"][:, mode]  # already stored
        harmonics = self.results[freq_idx]["fourier_harmonics"]
        # dominant harmonic (excluding DC)
        power = np.abs(coeffs)
        power[0] = 0
        max_harm = np.max(power)
        total_power = np.sum(power)
        ratio = max_harm / total_power if total_power > 0 else 0
        return 0 if ratio > self.max_harm_limit else 1  # 0=symmetric, 1=asymmetric

    # -------------------------------------------------
    # 4. Full mode classifier (MATLAB logic)
    # -------------------------------------------------
    def _classify_modes(self, dispersion: Dict[str, np.ndarray]) -> np.ndarray:
        slo = dispersion["slowness"]  # (n_freq, n_modes)
        labels = np.zeros_like(slo, dtype=int)

        for f_idx in range(slo.shape[0]):
            for m in range(slo.shape[1]):
                # 1. Energy ratio criteria
                r1, r2 = self._energy_ratios(f_idx, m)
                if self.use_class_te_nt and r1 > self.te_pml_total_limit:
                    labels[f_idx, m] = 3  # reject
                    continue
                if self.use_class_abcpml_htti and r2 > 1.0:
                    labels[f_idx, m] = 3
                    continue

                # 2. Stoneley / fast / slow via slowness percentiles
                s = slo[f_idx, :]
                stoneley_thresh = np.min(s) * 1.05
                q33, q66 = np.quantile(s, [0.33, 0.66])
                if s[m] < stoneley_thresh:
                    labels[f_idx, m] = 0
                elif s[m] < q33:
                    labels[f_idx, m] = 1
                elif s[m] < q66:
                    labels[f_idx, m] = 2
                else:
                    labels[f_idx, m] = 3
        return labels

    # -------------------------------------------------
    # 5. Automatic mode swapping (1 ↔ 2) when branches cross
    # -------------------------------------------------
    def _swap_crossing(self, dispersion: Dict[str, np.ndarray],
                      labels: np.ndarray) -> np.ndarray:
        slo = dispersion["slowness"]
        n_freq, n_modes = slo.shape
        new_labels = labels.copy()

        for f_idx in range(1, n_freq):
            for m in range(n_modes - 1):
                # if current mode overtakes next mode – swap labels
                if slo[f_idx, m] > slo[f_idx, m + 1]:
                    new_labels[f_idx:, m], new_labels[f_idx:, m + 1] = \
                        new_labels[f_idx:, m + 1], new_labels[f_idx:, m]
        return new_labels

    # -------------------------------------------------
    # 6. Plot slowness curves with colours per class
    # -------------------------------------------------
    def plot_slowness(self, ax: Optional[plt.Axes] = None,
                     show: bool = True) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 5))

        dispersion = self._assemble_dispersion()
        labels = self._classify_modes(dispersion)
        labels = self._swap_crossing(dispersion, labels)

        colors = ["k", "r", "b", "g"]
        names = ["Stoneley", "Fast", "Slow", "Other"]
        plotted = set()

        for m in range(self.nmodes):
            col = colors[labels[0, m]]
            lab = names[labels[0, m]]
            ax.plot(dispersion["freq"] / 1e3,
                   dispersion["slowness"][:, m] * 1e3,
                   color=col, lw=2,
                   label=lab if lab not in plotted else "")
            plotted.add(lab)

        ax.set_xlabel("Frequency (kHz)")
        ax.set_ylabel("Slowness (s/km)")
        ax.set_title("Dispersion curves (classified)")
        ax.legend()
        ax.grid(alpha=0.3)
        if show:
            plt.show()
        return ax

    # -------------------------------------------------
    # 7. Polar kinetic-energy map
    # -------------------------------------------------
    def plot_energy_map(self, freq_idx: int = 0, mode: int = 0,
                       ax: Optional[plt.Axes] = None,
                       show: bool = True) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots(subplot_kw=dict(projection='polar'))

        r = self.results[freq_idx]["R"][:, 0]
        phi = self.results[freq_idx]["Phi"][0, :]
        ke = self.results[freq_idx]["KE"][:, :, mode]

        pcol = ax.pcolormesh(phi, r, ke, shading='auto', cmap='magma')
        ax.set_title(f"KE density  f={self.freqs[freq_idx]/1e3:.1f} kHz  mode={mode}")
        plt.colorbar(pcol, ax=ax, pad=0.1)
        if show:
            plt.show()
        return ax

    # -------------------------------------------------
    # 8. High-level API
    # -------------------------------------------------
    def run(self, out_dir: Optional[Path] = None) -> Dict[str, np.ndarray]:
        dispersion = self._assemble_dispersion()
        labels = self._classify_modes(dispersion)
        labels = self._swap_crossing(dispersion, labels)

        self.plot_slowness(show=False)
        if out_dir:
            out_dir = Path(out_dir)
            out_dir.mkdir(exist_ok=True)
            # save curves
            np.savez(out_dir / "dispersion_curves.npz",
                     freq=dispersion["freq"],
                     slowness=dispersion["slowness"],
                     velocity=dispersion["velocity"],
                     labels=labels)
            # save figure
            plt.savefig(out_dir / "slowness_classified.png", dpi=200)
            plt.close()
        return {"dispersion": dispersion, "labels": labels}