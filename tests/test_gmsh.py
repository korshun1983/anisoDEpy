import sys
from pathlib import Path
# allow local imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geometry_builder import load_model, build_cylindrical


def test_gmsh():
    model = load_model("BakkenB-00.json.json")
    mesh  = build_cylindrical(model)
    assert mesh.tri6.shape[1] == 6, "Tri6 must have 6 nodes per element"
    print("Gmsh OK")


if __name__ == "__main__":
    test_gmsh()