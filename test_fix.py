# D:\Работа\python\anisoDEpy\test_fix.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.absolute()))

import json
from utils import debug_print

# Включаем максимальный дебаг
import os
os.environ["SAFE_DEBUG_LEVEL"] = "5"

print("=== Testing imports ===")
try:
    from stage2.st2_2_prepare_model_methods_sp_safe import st2_2_prepare_model_methods_sp_safe
    print("✓ st2_2_prepare_model_methods_sp_safe imported successfully")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

print("\n=== Testing Stage 1 ===")
with open('models/Bakken-B/BakkenB-00.json', 'r') as f:
    model_data = json.load(f)

from stage1.st1_set_model import st1_set_model
InputParam = st1_set_model(model_data)
print(f"✓ Stage 1 complete: {len(InputParam.get('Methods', {}))} methods")

print("\n=== Testing Stage 2 ===")
from stage2.st2_prepare_model_sp_safe import st2_prepare_model_sp_safe
try:
    CompStruct = st2_prepare_model_sp_safe(InputParam)
    print(f"✓ Stage 2 complete: {len(CompStruct['Methods'])} methods")
    print(f"  Methods: {list(CompStruct['Methods'].keys())}")
except Exception as e:
    print(f"✗ Stage 2 failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n=== Testing Stage 3 ===")
try:
    # This will generate the mesh
    MeshNodes, BoundaryEdges, MeshTri, MeshProps, _ = CompStruct['Methods']['PrepareMesh'](CompStruct)
    print(f"✓ Mesh generation SUCCESS!")
    print(f"  Nodes: {MeshNodes.shape[1]}")
    print(f"  Elements: {MeshTri.shape[1]}")
    print(f"  Boundary edges: {BoundaryEdges.shape[1] if BoundaryEdges.size > 0 else 0}")
except Exception as e:
    print(f"✗ Mesh generation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n🎉 All tests PASSED!")