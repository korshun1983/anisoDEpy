#!/usr/bin/env python3
# ------------------------------------------------------------------
#  anisoDEpy – dispersion curves for cylindrically-layered
#              anisotropic waveguides (SAFE, Tri6, Python)
# ------------------------------------------------------------------
import sys
from pathlib import Path
import numpy as np
import tkinter as tk
from tkinter import filedialog
import time

# allow local imports when running from any folder
sys.path.insert(0, str(Path(__file__).resolve().parent))

from geometry_builder import load_model, build_cylindrical_for_frequency
from geometry_builder.gmsh_builder import get_min_velocity
from field_solver import build_global_matrices, solve_safe
from scipy.sparse.linalg import eigs
from post_processor.processor import TEProcessor          # stage-2
from post_processor.interpreter import TEInterpreter      # stage-3


# ------------------------------------------------------------------
# GUI helpers
# ------------------------------------------------------------------
def pick_json_file() -> Path:
    """Open file-dialog starting inside local models/ folder."""
    root = tk.Tk()
    root.withdraw()
    root.update()

    models_dir = Path(__file__).with_name("models").resolve()
    file = filedialog.askopenfilename(
        title="Select JSON model file",
        initialdir=models_dir,
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
    )
    root.destroy()
    if not file:
        print("No file selected – exiting.")
        sys.exit(0)
    return Path(file)


def ask_mesh_preview() -> bool:
    """Ask user if mesh preview is required."""
    ans = input("\nDo you want to preview the mesh? (y/n): ").strip().lower()
    return ans in {"y", "yes"}


def ask_detailed_mesh() -> bool:
    """Choose between simplified or detailed mesh plot."""
    choice = input("\n1 – simplified   2 – detailed   [1/2] (default: 1): ").strip()
    return choice == "2"


def ask_continue() -> bool:
    """Ask whether to proceed with dispersion calculation."""
    ans = input("\nContinue with dispersion calculation? (y/n): ").strip().lower()
    return ans in {"y", "yes"}


# ------------------------------------------------------------------
# mesh visualisation
# ------------------------------------------------------------------
def plot_mesh_preview(mesh, detailed: bool = False):
    """Fast or detailed mesh plot."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
    except ImportError:
        print("Matplotlib not available – skipping mesh preview")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # outline (always drawn)
    corner_idx = mesh.tri6[:, :3]
    corner_coords = mesh.coord[corner_idx]

    nelem = len(mesh.tri6)
    segments = np.zeros((nelem * 3, 2, 2))
    segments[0::3] = np.stack((corner_coords[:, 0], corner_coords[:, 1]), axis=1)
    segments[1::3] = np.stack((corner_coords[:, 1], corner_coords[:, 2]), axis=1)
    segments[2::3] = np.stack((corner_coords[:, 2], corner_coords[:, 0]), axis=1)

    ax.add_collection(LineCollection(segments, colors='blue', linewidths=0.3, alpha=0.6))

    # detailed view: mid-side nodes + centres
    if detailed and len(mesh.tri6[0]) == 6:
        mids = np.unique(mesh.tri6[:, 3:6].ravel())
        ax.scatter(mesh.coord[mids, 0], mesh.coord[mids, 1],
                   c='green', s=6, marker='s', label='Mid-side nodes', zorder=4)
        centres = corner_coords.mean(axis=1)
        ax.scatter(centres[:, 0], centres[:, 1],
                   c='orange', s=4, marker='^', label='Triangle centres', zorder=4)
        ax.legend(loc='upper right', fontsize=8)

    ax.set_aspect('equal')
    ax.autoscale()
    ax.set_title(f'Mesh preview: {mesh.nnod} nodes, {mesh.nelem} elements')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------
# main workflow
# ------------------------------------------------------------------
def main():
    json_file = pick_json_file()
    model = load_model(json_file)

    print(f"Loaded model: {json_file.name}")
    print(f"Layers: {len(model['Model']['DomainType'])}")
    f_range = model['Model']['f_array_range']
    print(f"Frequency range: {f_range['start']}-{f_range['end']} kHz, step {f_range['step']} kHz")

    # optional mesh preview
    if ask_mesh_preview():
        demo_mesh = build_cylindrical_for_frequency(model, frequency=f_range['start'])
        plot_mesh_preview(demo_mesh, detailed=ask_detailed_mesh())
        if not ask_continue():
            print("Calculation cancelled.")
            return

    # frequency list (kHz)
    f_khz = np.arange(f_range['start'], f_range['end'] + f_range['step'], f_range['step'])
    n_freq = len(f_khz)

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    results_for_interp = []           # list of processor outputs for interpreter
    freq_list = []                    # keep frequencies for raw plot

    for idx, f_khz in enumerate(f_khz, 1):
        print(f"\nFrequency {idx}/{n_freq}: {f_khz:.2f} kHz")

        # build frequency-adapted mesh
        mesh = build_cylindrical_for_frequency(model, frequency=f_khz)
        print(f"  Mesh: {mesh.nnod} nodes, {mesh.nelem} elements")

        # ---- solve SAFE ----
        omega = 2 * np.pi * f_khz * 1e3
        K, M, active_dof = build_global_matrices(mesh, model, omega)

        # ---- DEBUG: ----
        print(f'  Before solver: K shape {K.shape}, nnz {K.nnz}, dtype {K.dtype}')
        print(f'  omega = {omega:.3f} rad/s, sigma = {omega * (1 + 0.5j)}')

        t0 = time.time()
        try:
            w, v = solve_safe(K, M, nev=10 , sigma=omega * (1 + 0.5j), which='LR')
            print(f'  Solver OK: {w.size} modes')
            print(f'  eigen_values (rad/s): {w}')
            print(f'  slowness (s/km):      {1.0 / w * 1e3}')

        except Exception as e:
            import traceback
            print('\n[ERROR] Solver failed:')
            traceback.print_exc()
            print(f'  Solver time: {time.time() - t0:.2f} s')
            sys.exit(1)

        nev = w.size  # actual number of modes
        active_dof = np.unique(np.clip(active_dof - 1, 0, mesh.coord.shape[0] - 1))
        v_node = v[::3, :]  # one scalar per node
        assert v_node.shape == (mesh.coord[active_dof].shape[0], nev)

        # ---- stage-2 processing ----
        proc = TEProcessor(nodes=mesh.coord,
                           eigen_vals=w,
                           eigen_vecs=v_node,
                           active_dof=active_dof)
        res = proc.run(nr=60, nphi=128)
        proc.save_npz(out_dir / f"Results-{idx}.npz")

        # ---- keep FULL eigen-values for raw plot ----
        res["eig_val"] = w  # explicit store (nev,)
        results_for_interp.append(res)
        freq_list.append(f_khz)

    freq_hz_all = []
    slow_all = []
    for f_khz, res in zip(freq_list, results_for_interp):
        f_hz = f_khz * 1e3  # Hz
        slow = (1.0 / res["eig_val"]) * 1e6  # μs/m (nev значений)
        # повторяем ту же частоту для каждого собственного значения
        freq_hz_all.extend([f_hz] * slow.size)
        slow_all.extend(slow)

    raw_data = {
        'freq_Hz': np.array(freq_hz_all),
        'slowness_μsm': np.array(slow_all),
    }
    out_dir = Path("output")
    np.savez(out_dir / 'raw_data.npz', **raw_data)

    # ------------------------------------------------------------------
    # OPTIONAL: raw dispersion dots (all modes, no filtering)
    # ------------------------------------------------------------------
    raw_plot = input("\nShow raw dispersion curves (dots, all modes) ? (y/n): ").strip().lower()
    if raw_plot in {"y", "yes"}:
        try:
            import matplotlib.pyplot as plt

            freq_hz_all, slow_all = [], []
            for f_khz, res in zip(freq_list, results_for_interp):
                f_hz = f_khz * 1e3                      # Hz

                slow = (1.0 / res["eig_val"]) * 1e6     # μs/m

                freq_hz_all.extend([f_hz] * slow.size)
                slow_all.extend(slow)

            freq_hz_all = np.array(freq_hz_all)
            slow_all = np.array(slow_all)

            plt.figure(figsize=(7, 5))
            plt.scatter(freq_hz_all / 1e3, slow_all, s=8, c='k', marker='o')
            plt.xlabel('Frequency, kHz')
            plt.ylabel('Slowness, μs/m')
            plt.title('Raw dispersion – all calculated modes (dots). Python')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print("Raw dispersion plot failed:", e)
    #
    # else:
    #     # stage-3 interpretation
    #     dummy_mask = np.zeros(mesh.coord.shape[0], dtype=bool)
    #     interpreter = TEInterpreter(results_for_interp,
    #                                 nodes=mesh.coord,
    #                                 pml_mask=dummy_mask,
    #                                 adj_mask=dummy_mask)
    #     summary = interpreter.run(out_dir=out_dir)
    #     print("\nStage-3 interpretation complete – figures saved to", out_dir.resolve())


if __name__ == "__main__":
    main()