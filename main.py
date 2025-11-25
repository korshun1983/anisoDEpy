#!/usr/bin/env python3
# ------------------------------------------------------------------
#  anisoDEpy  –  dispersion curves for cylindrically layered
#               anisotropic wave-guides (SAFE, Tri6, Python).
# ------------------------------------------------------------------
import sys
from pathlib import Path
import numpy as np
import tkinter as tk
from tkinter import filedialog

# allow local imports when running from any folder
sys.path.insert(0, str(Path(__file__).resolve().parent))

from geometry_builder import load_model, build_cylindrical_for_frequency
from field_solver import build_global_matrices, solve_safe
from post_processor import plot_slowness


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


def compare_mesh_for_frequencies(model, frequencies):
    """Compare mesh sizes for different frequencies."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available for comparison")
        return

    min_velocity = get_min_velocity(model)

    fig, axes = plt.subplots(1, len(frequencies), figsize=(5 * len(frequencies), 5))
    if len(frequencies) == 1:
        axes = [axes]

    for idx, freq in enumerate(frequencies):
        mesh = build_cylindrical_for_frequency(model, frequency=freq)

        # Simple fast plot
        elements = mesh.tri6[:, :3]  # Use only corners for plotting
        coords = mesh.coord

        for element in elements:
            triangle = element[[0, 1, 2, 0]]  # Close the triangle
            x = coords[triangle, 0]
            y = coords[triangle, 1]
            axes[idx].plot(x, y, 'b-', linewidth=0.5, alpha=0.6)

        axes[idx].set_aspect('equal')
        axes[idx].set_title(f'{freq} kHz: {mesh.nnod} nodes, {mesh.nelem} elements')
        axes[idx].set_xlabel('X (m)')
        axes[idx].set_ylabel('Y (m)')

    plt.tight_layout()
    plt.show()


def plot_mesh_preview(mesh, model):
    """Fast mesh visualization using matplotlib collections."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
    except ImportError:
        print("Matplotlib not available for mesh preview")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # Extract corner nodes (first 3 nodes of each Tri6 element)
    corner_indices = mesh.tri6[:, :3]
    corner_coords = mesh.coord[corner_indices]  # Shape: (nelem, 3, 2)

    # Create line segments for triangle edges - CORRECTED
    segments = []
    for tri in corner_coords:
        # Three edges of the triangle
        segments.append([tri[0], tri[1]])  # edge 0-1
        segments.append([tri[1], tri[2]])  # edge 1-2
        segments.append([tri[2], tri[0]])  # edge 2-0

    segments = np.array(segments)

    line_collection = LineCollection(segments,
                                     colors='blue',
                                     linewidths=0.5,
                                     alpha=0.7)
    ax.add_collection(line_collection)

    # Draw all nodes - FAST with single scatter call
    ax.scatter(mesh.coord[:, 0], mesh.coord[:, 1],
               c='red', s=8, alpha=0.8, zorder=3, label='All nodes')

    # Highlight mid-side nodes and center nodes
    if len(mesh.tri6[0]) == 6:  # Only for Tri6 elements
        # Mid-side nodes are indices 3,4,5 in Tri6
        mid_side_indices = mesh.tri6[:, 3:6].flatten()
        mid_side_coords = mesh.coord[np.unique(mid_side_indices)]

        # Calculate approximate center of each triangle
        centers = np.mean(corner_coords, axis=1)

        # Plot mid-side nodes and centers
        ax.scatter(mid_side_coords[:, 0], mid_side_coords[:, 1],
                   c='green', s=6, alpha=0.8, marker='s', zorder=4,
                   label='Mid-side nodes')
        ax.scatter(centers[:, 0], centers[:, 1],
                   c='orange', s=4, alpha=0.8, marker='^', zorder=4,
                   label='Triangle centers')

        ax.legend(loc='upper right', fontsize=8)

    ax.set_aspect('equal')
    ax.set_title(f'Detailed Mesh View: {mesh.nnod} nodes, {mesh.nelem} elements')
    ax.set_xlabel('X coordinate (m)')
    ax.set_ylabel('Y coordinate (m)')
    ax.grid(True, alpha=0.3)

    # Set reasonable limits
    x_min, x_max = np.min(mesh.coord[:, 0]), np.max(mesh.coord[:, 0])
    y_min, y_max = np.min(mesh.coord[:, 1]), np.max(mesh.coord[:, 1])
    margin = max(x_max - x_min, y_max - y_min) * 0.05
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)

    # Add model information
    model_info = f"Layers: {len(model['Model']['DomainType'])}\n"
    model_info += f"Elements: {mesh.nelem}\n"
    model_info += f"Nodes: {mesh.nnod}\n"
    model_info += f"Frequency range: {model['Model']['f_array_range']['start']}-{model['Model']['f_array_range']['end']} kHz"

    ax.text(0.02, 0.98, model_info, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.show()


def plot_mesh_preview_fast(mesh, model):
    """Ultra-fast mesh visualization - contours only."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
    except ImportError:
        print("Matplotlib not available for mesh preview")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # Extract only corner nodes for faster rendering
    corner_indices = mesh.tri6[:, :3]
    corner_coords = mesh.coord[corner_indices]  # Shape: (nelem, 3, 2)

    # Create ALL line segments at once - CORRECTED VERSION
    nelem = len(mesh.tri6)
    segments = np.zeros((nelem * 3, 2, 2))  # 3 edges per element

    # Edge 0: node0 -> node1
    segments[0::3, 0, :] = corner_coords[:, 0, :]
    segments[0::3, 1, :] = corner_coords[:, 1, :]

    # Edge 1: node1 -> node2
    segments[1::3, 0, :] = corner_coords[:, 1, :]
    segments[1::3, 1, :] = corner_coords[:, 2, :]

    # Edge 2: node2 -> node0
    segments[2::3, 0, :] = corner_coords[:, 2, :]
    segments[2::3, 1, :] = corner_coords[:, 0, :]

    line_collection = LineCollection(segments,
                                     colors='blue',
                                     linewidths=0.3,
                                     alpha=0.6)
    ax.add_collection(line_collection)

    # Optional: plot only corner nodes for very large meshes
    if mesh.nnod < 10000:  # Only if mesh is not too big
        corner_nodes = np.unique(corner_indices.flatten())
        ax.scatter(mesh.coord[corner_nodes, 0], mesh.coord[corner_nodes, 1],
                   c='red', s=2, alpha=0.6, zorder=2)

    ax.set_aspect('equal')
    ax.set_title(f'Simplified Mesh View: {mesh.nnod} nodes, {mesh.nelem} elements')
    ax.set_xlabel('X coordinate (m)')
    ax.set_ylabel('Y coordinate (m)')

    # Auto-scale
    ax.autoscale()

    # Simple info
    ax.text(0.02, 0.98, f'Elements: {mesh.nelem}\nNodes: {mesh.nnod}',
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white"))

    plt.tight_layout()
    plt.show()


def main():
    json_file = pick_json_file()
    model = load_model(json_file)

    print(f"Loaded model: {json_file.name}")
    print(f"Layers: {len(model['Model']['DomainType'])}")
    print(f"Frequency range: {model['Model']['f_array_range']['start']}-{model['Model']['f_array_range']['end']} kHz")
    print(f"Step: {model['Model']['f_array_range']['step']} kHz")

    # Ask for mesh preview
    response = input("\nDo you want to preview the mesh? (y/n): ").strip().lower()

    # Calculate minimum velocity once
    from geometry_builder.gmsh_builder import get_min_velocity
    min_velocity = get_min_velocity(model)
    print(f"Minimum wave velocity in model: {min_velocity:.0f} m/s")

    f_arr = model['Model']['f_array']
    out = []

    for i, f in enumerate(f_arr):
        print(f"\nProcessing frequency {i + 1}/{len(f_arr)}: {f:.2f} kHz")

        # Build mesh optimized for this frequency
        mesh = build_cylindrical_for_frequency(model, frequency=f)
        print(f"Mesh for {f} kHz: {mesh.nnod} nodes, {mesh.nelem} elements")

        # Show mesh preview only for first frequency or if requested
        if i == 0 and response in ['y', 'yes']:
            # Ask for visualization type
            viz_choice = input("\nChoose mesh visualization type:\n"
                               "1 - Simplified (triangular elements only)\n"
                               "2 - Detailed (with mid-side nodes and centers)\n"
                               "Enter choice [1/2] (default: 1): ").strip()

            if viz_choice == '2':
                print("Plotting detailed mesh preview...")
                plot_mesh_preview(mesh, model)
            else:
                print("Plotting simplified mesh preview...")
                plot_mesh_preview_fast(mesh, model)

            # Ask if user wants to continue with calculation
            cont = input("\nContinue with dispersion calculation? (y/n): ").strip().lower()
            if cont not in ['y', 'yes']:
                print("Calculation cancelled.")
                return

        # Solve for this frequency
        omega = 2 * np.pi * f * 1e3  # rad/s
        K, M, dof = build_global_matrices(mesh, model, omega)
        w, v = solve_safe(K, M, nev=50, sigma=omega * 1.1)
        out.append((f, w, v))

    plot_slowness(out, model)


if __name__ == "__main__":
    main()y