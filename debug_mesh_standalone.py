#!/usr/bin/env python3
"""
debug_mesh_standalone.py
========================
Отладка сетки с ПРАВИЛЬНОЙ структурой CompStruct
"""

import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Добавляем корень проекта в путь
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from utils import debug_print, timer
from routines.mesh.prepare_mesh_sp_safe import prepare_mesh_sp_safe


def debug_mesh_generation():
    """Генерируем сетку для Bakken-B модели корректным путем"""
    debug_print("=" * 70, level=1)
    debug_print("DEBUG: Mesh Generation Test (Production Way)", level=1)
    debug_print("=" * 70, level=1)

    # 1. Создаём ПОЛНЫЙ CompStruct как после Stage 1 и Stage 2
    CompStruct = {
        'Config': {
            'ProblemType': 'spectrum',
            'NumMethod': 'SAFE',
            'FreqUnits': 'kHz',
            'SloUnits': 'us/ft',
            'root_path': str(ROOT_DIR / 'routines'),
            'solver_path': str(ROOT_DIR / 'routines' / 'spectrum')
        },
        'Model': {
            'DomainRx': [0.1, 2.0],
            'DomainRy': [0.1, 2.0],
            'DomainTheta': [0, 0],
            'DomainEcc': [0, 0],
            'DomainEccAngle': [0, 0],
            'DomainType': ['fluid', 'HTTI'],
            'BCType': ['FS', 'rigid'],
            'LDomain_in_LSH': 'yes',
            'AddDomainLoc': 'ext',
            'AddDomainType': 'abc',
            'AddDomainL': 1.0,
            'PML_factor': 10,
            'PML_degree': 2.0,
            'PML_method': 2.0,
            'ABC_factor': 0.1,
            'ABC_degree': 1.0,
            'ABC_account_r': 'yes',
            'DomainParam': [
                [1000.0, 2.25e9],
                [2600.0, 40.9e9, 8.5e9, 26.9e9, 10.5e9, 15.3e9, 0, 0]
            ],
            'DomainNth': [12, 12],
            'mud_domain': 1,
            'RefDomainType': ['HTTI'],
            'RefDomainParam': [[2600.0, 40.9e9, 8.5e9, 26.9e9, 10.5e9, 15.3e9, 0, 0]],
            'f_array': [1.0, 2.0, 3.0, 4.0, 5.0],
            'f_array_range': {'start': 1.0, 'step': 1.0, 'end': 5.0},
            'N_disp': 5
        },
        'Advanced': {
            'num_eig_max': 10,
            'EigSearchStart': 1.0,
            'VisualizeMesh': True,
            'VisualizeGeometry': True,
            'N_nodes': 10,
            'NEdge_nodes': 4,
            'EigsOptions': {'disp': 0, 'tol': 1e-8}
        },
        'Mesh': {
            'ext_boundary_shape': 'cir',
            'hmax': 0.16,
            'dhmax': 0.25,
            'output': 'yes'
        },
        'Data': {
            'N_domain': 2,
            'DVarNum': np.array([1, 3])
        },
        'Methods': {},
        'Misc': {
            'F_conv': 1000.0,
            'S_conv': 304.8
        }
    }

    # 2. Устанавливаем методы (как в Stage 2.2)
    debug_print("Setting up methods...", level=2)
    from routines.mesh.mesh_generator import (
        prepare_mesh_bh, meshfaces, add_nodes_cubic, find_bedges,
        find_edge_orient, make_cont_bedges
    )
    from routines.physics.prepare_physprop_fluid_sp_safe import prepare_physprop_fluid_sp_safe
    from routines.physics.prepare_physprop_htti_sp_safe import prepare_physprop_htti_sp_safe

    Methods = CompStruct['Methods']
    Methods['PrepareMesh'] = prepare_mesh_sp_safe
    Methods['PrepareMeshBH'] = prepare_mesh_bh
    Methods['MeshFaces'] = meshfaces  # ВАЖНО: имя должно совпадать!
    Methods['AddNodesCubic'] = add_nodes_cubic
    Methods['FindBEdges'] = find_bedges
    Methods['MakeContBEdges'] = make_cont_bedges
    Methods['FindEdgeOrient'] = find_edge_orient

    n_domain = CompStruct['Data']['N_domain']
    Methods['PreparePhysProp'] = [None] * n_domain
    for ii_d in range(n_domain):
        domain_type = CompStruct['Model']['DomainType'][ii_d]
        Methods['PreparePhysProp'][ii_d] = (
            prepare_physprop_fluid_sp_safe if domain_type == 'fluid'
            else prepare_physprop_htti_sp_safe
        )

    # 3. Запускаем генерацию сетки
    debug_print("STAGE 3: Mesh Generation", level=1)
    with timer("Mesh Generation"):
        result = prepare_mesh_sp_safe(CompStruct)

        if result is None or len(result) != 5:
            raise RuntimeError(f"PrepareMesh returned invalid result: {result}")

        MeshNodes, BoundaryEdges, MeshTri, MeshProps, CompStruct = result

    # 4. Валидация
    debug_print(f"Mesh generation complete: {MeshNodes.shape[1]} nodes, {MeshTri.shape[1]} elements", level=1)

    # 5. Визуализация
    _visualize_mesh(MeshNodes, MeshTri, "debug_mesh_final.png")

    debug_print("=" * 70, level=1)
    debug_print(f"SUCCESS: {MeshNodes.shape[1]} nodes, {MeshTri.shape[1]} elements", level=1)
    debug_print("=" * 70, level=1)


def _visualize_mesh(MeshNodes, MeshTri, filename):
    """Сохраняет визуализацию сетки с разделением по доменам"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # Все элементы
    tri_linear = MeshTri[:3, :].T - 1
    ax1.triplot(MeshNodes[0, :], MeshNodes[1, :], tri_linear, 'b-', lw=0.5)
    ax1.plot(MeshNodes[0, :], MeshNodes[1, :], 'r.', markersize=2)
    ax1.set_title("All Elements")
    ax1.axis('equal')

    # Домены разными цветами
    unique_domains = np.unique(MeshTri[3, :])
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_domains)))

    for domain_id, color in zip(unique_domains, colors):
        mask = MeshTri[3, :] == domain_id
        tri_domain = MeshTri[:3, mask].T - 1
        ax2.triplot(MeshNodes[0, :], MeshNodes[1, :], tri_domain,
                    color=color, lw=0.5, label=f'Domain {domain_id}')

    ax2.set_title("Elements by Domain")
    ax2.axis('equal')
    ax2.legend(loc='upper right', fontsize=8)

    plt.savefig(filename, dpi=150, bbox_inches='tight')
    debug_print(f"✓ Mesh visualization saved: {filename}", level=1)


if __name__ == "__main__":
    debug_print("Starting mesh debug...", level=0)
    try:
        debug_mesh_generation()
    except Exception as e:
        debug_print(f"ERROR: {e}", level=0)
        import traceback

        traceback.print_exc()
        sys.exit(1)