"""
Mesh visualisation test (no Gmsh GUI) – rev1 branch
User picks any JSON -> Tri6 mesh is drawn with Matplotlib.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geometry_builder import load_model, build_cylindrical


def test_plot_mesh():
    root = tk.Tk()
    root.withdraw()
    models = Path(__file__).resolve().parent.parent / "models"
    file = filedialog.askopenfilename(
        title="Pick JSON to visualise mesh",
        initialdir=models,
        filetypes=[("JSON", "*.json")]
    )
    root.destroy()
    if not file:
        print("Cancelled"); return

    model = load_model(file)
    mesh  = build_cylindrical(model)

    nodes = mesh.coord
    tri6  = mesh.tri6

    plt.figure(figsize=(6, 6))
    for el in tri6:
        x = nodes[el[[0, 1, 2, 0]], 0]   # close triangle (corner nodes only)
        y = nodes[el[[0, 1, 2, 0]], 1]
        plt.plot(x, y, lw=0.5, color="steelblue")

    plt.gca().set_aspect("equal")
    plt.title("Tri6 mesh – corner outline")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.show()


if __name__ == "__main__":
    test_plot_mesh()