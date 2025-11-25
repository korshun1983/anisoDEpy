import sys
from pathlib import Path
import numpy as np
import tkinter as tk
from tkinter import filedialog

# add local packages
sys.path.insert(0, str(Path(__file__).resolve().parent))

from geometry_builder import load_model, build_cylindrical
from field_solver import build_global_matrices, solve_safe
from post_processor import plot_slowness


def pick_json_file() -> Path:
    """Open file-dialog starting inside the local models/ folder."""
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


def main():
    json_file = pick_json_file()
    model = load_model(json_file)
    mesh  = build_cylindrical(model)

    f_arr = model['Model']['f_array']
    out   = []
    for f in f_arr:
        omega = 2 * np.pi * f * 1e3
        K, M, dof = build_global_matrices(mesh, model, omega)
        w, v      = solve_safe(K, M, nev=50, sigma=omega * 1.1)
        out.append((f, w, v))

    plot_slowness(out, model)


if __name__ == "__main__":
    main()