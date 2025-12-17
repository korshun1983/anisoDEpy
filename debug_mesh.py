import json
from stage1.st1_set_model import st1_set_model
from stage2.st2_prepare_model_sp_safe import st2_prepare_model_sp_safe

# Загрузка модели
with open('models/Bakken-B/BakkenB-00.json', 'r') as f:
    model_data = json.load(f)

print("=== STAGE 1 ===")
InputParam = st1_set_model(model_data)
print(f"Keys in InputParam: {list(InputParam.keys())}")
print(f"Keys in InputParam['Advanced']: {list(InputParam.get('Advanced', {}).keys())}")

print("\n=== STAGE 2 ===")
CompStruct = st2_prepare_model_sp_safe(InputParam)
print(f"Keys in CompStruct: {list(CompStruct.keys())}")
print(f"Keys in CompStruct['Advanced']: {list(CompStruct.get('Advanced', {}).keys())}")
print(f"Keys in CompStruct['Methods']: {list(CompStruct.get('Methods', {}).keys())}")

print("\n=== STAGE 3 ===")
try:
    result = CompStruct['Methods']['PrepareMesh'](CompStruct)
    print("Mesh generation SUCCESS!")
    print(f"MeshNodes shape: {result[0].shape}")
except Exception as e:
    print(f"Mesh generation FAILED: {e}")
    import traceback
    traceback.print_exc()