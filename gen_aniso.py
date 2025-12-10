#!/usr/bin/env python3
"""
===============================================================================
SAFE (Spectral Analysis of Finite Elements) for Anisotropic Media
Main Driver Script - Python Implementation v2.5 (FULLY FIXED)
===============================================================================
НОВОЕ:
1. Визуализация средствами gmsh (3D окно)
2. Логирование радиуса PML слоя
3. Отладочная информация о геометрии
===============================================================================
"""

import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
from scipy.io import savemat
import tkinter as tk
from tkinter import filedialog

# === ОТЛАДКА: импорт для визуализации ===
import matplotlib.pyplot as plt
# =========================================

# Project modules
from core.config import InputParam, CompStruct
from methods.stage1 import initialize_model
from methods import stage2
from methods.stage3 import run_stage3_matrix_assembly
from methods.stage4 import compute_solution as stage4_compute
from routines.io_utils import cleanup_output_dir
from routines.matrix_assembly import (
    em_tensor_vti,
    rotate_c_ij,
    rot_matrix
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler('safe_pipeline.log', mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def select_parameter_file(initial_dir: Optional[Path] = None) -> Path:
    """GUI file selection for model parameter file (.json)"""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    if initial_dir is None:
        initial_dir = Path(__file__).parent / "models"

    try:
        selected_file = filedialog.askopenfilename(
            title="Select Model Parameter File (JSON)",
            initialdir=str(initial_dir),
            filetypes=[
                ("JSON files", "*.json"),
                ("All files", "*.*")
            ]
        )
    except Exception as e:
        logger.error(f"GUI error: {e}")
        selected_file = input("Enter parameter file path: ")

    if not selected_file:
        logger.error("No file selected. Exiting.")
        sys.exit(1)

    return Path(selected_file)


def validate_parameter_file(param_file: Path) -> None:
    """Validate selected parameter file exists and has correct format."""
    if not param_file.exists():
        logger.error(f"Parameter file does not exist: {param_file}")
        sys.exit(1)

    if param_file.suffix.lower() != '.json':
        logger.error(f"Parameter file must be JSON format: {param_file}")
        sys.exit(1)

    logger.info(f"  Validated parameter file: {param_file}")


def validate_json_structure(json_data: Dict[str, Any]) -> None:
    """
    Comprehensive JSON validation against SAFE schema.
    """
    required_top = ["Model", "Advanced"]
    for key in required_top:
        if key not in json_data:
            raise ValueError(f"Missing required top-level key: '{key}'")

    model = json_data["Model"]

    domain_keys = ["DomainRx", "DomainRy", "DomainType", "DomainParam", "BCType"]
    for key in domain_keys:
        if key not in model:
            raise ValueError(f"Model missing required key: '{key}'")

    n_boundaries = len(model["DomainRx"])
    n_domains = len(model["DomainType"])

    # DomainRx должен быть на 1 больше, чем DomainType
    if n_boundaries != n_domains + 1:
        raise ValueError(
            f"Model.DomainRx length ({n_boundaries}) should be DomainType length + 1 ({n_domains + 1})"
        )

    # Массивы ГРАНИЦ должны совпадать с n_boundaries
    boundary_arrays = ['DomainRy', 'DomainTheta', 'DomainEcc', 'DomainEccAngle']
    for key in boundary_arrays:
        if key in model and len(model[key]) != n_boundaries:
            raise ValueError(
                f"Model.{key} length ({len(model[key])}) should match DomainRx ({n_boundaries})"
            )

    # === КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: DomainNth относится к ДОМЕНАМ ===
    if 'DomainNth' in model and len(model['DomainNth']) != n_domains:
        raise ValueError(
            f"Model.DomainNth length ({len(model['DomainNth'])}) should match DomainType ({n_domains})"
        )
    # ============================================================================

    # Остальные проверки остаются без изменений...

    # === КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Проверка BCType с учетом PML ===
    has_additional = model.get("AddDomainType", "none").lower() != "none"
    # Базовая модель: BCType имеет длину n_domains + 1 (для внешней границы)
    # После добавления PML: BCType будет иметь длину n_domains + 1 (автоматически)
    expected_bc_length = n_domains + 1  # Всегда n_domains + 1 для базовой модели

    if len(model["BCType"]) != expected_bc_length:
        raise ValueError(
            f"BCType length ({len(model['BCType'])}) should be n_domains + 1 ({expected_bc_length})"
        )
    # ============================================================================

    if "f_array_range" not in model:
        raise ValueError("Model missing f_array_range (start, step, end)")

    far = model["f_array_range"]
    if not all(k in far for k in ["start", "step", "end"]):
        raise ValueError("f_array_range must contain start, step, end")

    for i, params in enumerate(model["DomainParam"]):
        domain_type = model["DomainType"][i].lower()
        if domain_type == "fluid" and len(params) < 2:
            raise ValueError(f"Domain {i + 1} (fluid) needs [rho, lambda]")
        if domain_type == "htti" and len(params) < 7:
            raise ValueError(f"Domain {i + 1} (HTTI) needs [rho, c11, c13, c33, c44, c66, theta]")

    if "AddDomainLoc" in model and model["AddDomainLoc"].lower() not in ['ext', 'int']:
        raise ValueError("AddDomainLoc must be 'ext' or 'int'")


def register_physics_methods(InputParam: InputParam) -> InputParam:
    """Explicitly register rotation methods needed for HTTI physics."""
    InputParam.Methods['em_tensor_VTI'] = em_tensor_vti
    InputParam.Methods['rot_c_ij'] = rotate_c_ij
    InputParam.Methods['rot_matrix'] = rot_matrix

    required_methods = [
        'St2_PrepareModel',
        'St2_1_PrepareModelParams',
        'St2_2_PrepareModelMethods'
    ]

    for method in required_methods:
        if method not in InputParam.Methods or InputParam.Methods[method] is None:
            raise ValueError(f"Required method '{method}' not registered in Stage 1")

    return InputParam


def setup_additional_domains(CompStruct: CompStruct) -> CompStruct:
    """
    Append ABC/PML domain parameters if needed.
    Replicates MATLAB's domain extension logic.

    VAR IMPORTANTE: PML radius = last radius + AddDomainL (in meters or wavelengths)
    """
    add_type = CompStruct.Model['AddDomainType'].lower()

    if add_type == 'abc+pml':
        add_type = 'pml+abc'
        CompStruct.Model['AddDomainType'] = add_type

    if add_type != 'none':
        CompStruct.Model['AddDomain_Exist'] = 'yes'

        if CompStruct.Model['AddDomainLoc'].lower() == 'ext':
            # CALCULO DEL RADIO DE LA CAPA PML
            last_radius = CompStruct.Model['DomainRx'][-1]
            add_length = CompStruct.Model['AddDomainL']

            # Если LDomain_in_LSH = 'yes', то AddDomainL в длинах волны V_SH
            # Для простоты сейчас считаем, что в метрах (как в примере)
            new_radius = last_radius + add_length * 1.0

            # === ЛОГИРОВАНИЕ РАДИУСА PML ===
            logging.info(f"      Original last radius: {last_radius:.4f} m")
            logging.info(f"      AddDomainL: {add_length}")
            logging.info(f"      NEW PML layer radius: {new_radius:.4f} m")
            # =================================

            # Добавляем новый радиус
            CompStruct.Model['DomainRx'].append(new_radius)
            CompStruct.Model['DomainRy'].append(new_radius)
            CompStruct.Model['DomainTheta'].append(CompStruct.Model['DomainTheta'][-1])
            CompStruct.Model['DomainEcc'].append(CompStruct.Model['DomainEcc'][-1])
            CompStruct.Model['DomainEccAngle'].append(CompStruct.Model['DomainEccAngle'][-1])
            CompStruct.Model['DomainParam'].append(CompStruct.Model['DomainParam'][-1])
            CompStruct.Model['DomainType'].append(CompStruct.Model['DomainType'][-1])
            CompStruct.Model['BCType'].append('rigid')
            CompStruct.Model['DomainNth'].append(CompStruct.Model['DomainNth'][-1])
            CompStruct.Model['BCType'][-2] = 'SSstiff'

            # === КРИТИЧЕСКОЕ ОБНОВЛЕНИЕ DataParameters ===
            # Добавляем переменную для нового PML-домена (копия последнего)
            last_domain_vars = CompStruct.Data.DVarNum[-1]
            CompStruct.Data.DVarNum.append(last_domain_vars)

            # Обновляем количество доменов и интерфейсов
            CompStruct.Data.N_domain = len(CompStruct.Model['DomainType'])
            CompStruct.Data.N_interface = CompStruct.Data.N_domain - 1
            # === КОНЕЦ ОБНОВЛЕНИЯ ===

            logging.info(f"      Added {add_type.upper()} external layer")
            logging.info(f"      New BCType: {CompStruct.Model['BCType']}")
            logging.info(f"      Domains: {CompStruct.Model['DomainType']}")
            logging.info(f"      Radii: {CompStruct.Model['DomainRx']}")
            logging.info(f"      Updated DVarNum: {CompStruct.Data.DVarNum}")
            logging.info(f"      Updated N_domain: {CompStruct.Data.N_domain}")
            logging.info(f"      Updated N_interface: {CompStruct.Data.N_interface}")

        elif CompStruct.Model['AddDomainLoc'].lower() == 'int':
            logging.warning("Internal PML/ABC not yet implemented")

    else:
        CompStruct.Model['AddDomain_Exist'] = 'no'

    return CompStruct


def visualize_mesh_gmsh(CompStruct: CompStruct):
    """
    ВИЗУАЛИЗАЦИЯ СРЕДСТВАМИ GMSH (3D окно)
    Показывает геометрию и сетку в интерактивном окне gmsh
    """
    try:
        import gmsh

        logger.info("  Opening GMSH visualization...")

        # Получаем данные сетки
        MeshNodes = CompStruct.FEMatrices['MeshNodes']
        MeshTri = CompStruct.FEMatrices['MeshTri']

        # Инициализируем gmsh для визуализации
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)

        model = gmsh.model()
        model.add("visualization")

        # Создаем новую геометрию для визуализации
        factory = model.occ

        # Создаем диски для каждого домена (для наглядности)
        radii = CompStruct.Model['DomainRx']
        for i, r in enumerate(radii):
            if r > 0:
                factory.addDisk(0, 0, 0, r, r, tag=i + 100)  # Теги для геометрии

        factory.synchronize()

        # Добавляем точки узлов
        for i in range(MeshNodes.shape[1]):
            model.geo.addPoint(MeshNodes[0, i], MeshNodes[1, i], 0, tag=i + 1000)

        # Добавляем элементы (упрощенно, только для визуализации)
        # На самом деле gmsh уже знает сетку, мы могли бы использовать оригинальную модель
        # Но для простоты открываем окно с геометрией

        # Запускаем gmsh GUI
        gmsh.fltk.initialize()
        gmsh.fltk.run()

        gmsh.finalize()
        logger.info("  GMSH visualization closed")

    except Exception as e:
        logger.warning(f"GMSH visualization failed: {e}")
        logger.info("  Falling back to matplotlib visualization...")
        visualize_mesh_final_debug(CompStruct)


def visualize_mesh_final_debug(CompStruct: CompStruct):
    """
    РАСШИРЕННАЯ ОТЛАДОЧНАЯ визуализация сетки (matplotlib)
    """
    if not hasattr(CompStruct, 'FEMatrices') or CompStruct.FEMatrices is None:
        logger.warning("FEMatrices not found, skipping visualization")
        return

    MeshNodes = CompStruct.FEMatrices.get('MeshNodes')
    MeshTri = CompStruct.FEMatrices.get('MeshTri')
    BoundaryEdges = CompStruct.FEMatrices.get('BoundaryEdges')

    if MeshNodes is None or MeshTri is None:
        logger.warning("Mesh data incomplete, skipping visualization")
        return

    logger.info("  Creating DEBUG mesh visualization...")

    domain_types = CompStruct.Model['DomainType']
    n_domains = len(domain_types)
    domain_rx = np.array(CompStruct.Model['DomainRx'])

    _validate_domain_assignment(MeshNodes, MeshTri, domain_rx)

    colors = ['blue', 'red', 'green', 'orange', 'purple']
    styles = ['-', '--', '-.', ':', '-']
    n_elements = MeshTri.shape[1]
    max_plot_elements = min(n_elements, 2000)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, width_ratios=[2, 1, 1], height_ratios=[3, 1])

    # === ГРАФИК 1: ВСЕ домены вместе ===
    ax_main = fig.add_subplot(gs[0, 0])
    ax_main.set_title(f'ALL DOMAINS - {n_domains} domains', fontsize=14, fontweight='bold')

    # Границы доменов
    for i, r in enumerate(domain_rx):
        if r > 0:
            theta = np.linspace(0, 2 * np.pi, 200)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            ax_main.plot(x, y, 'k-', linewidth=2, alpha=0.7, label=f'Boundary r={r:.3f}m')

    # Элементы
    for el in range(max_plot_elements):
        node_ids = MeshTri[:3, el].astype(int)
        nodes = MeshNodes[:, node_ids]
        x_coords = np.append(nodes[0, :], nodes[0, 0])
        y_coords = np.append(nodes[1, :], nodes[1, 0])

        domain_id = int(MeshTri[-1, el]) - 1
        color = colors[domain_id % len(colors)]
        style = styles[domain_id % len(styles)]

        ax_main.plot(x_coords, y_coords, linestyle=style, color=color,
                     linewidth=0.5, alpha=0.6)

    # Граничные ребра
    if BoundaryEdges.shape[1] > 0:
        for edge_idx in range(min(BoundaryEdges.shape[1], 500)):
            n1, n2 = BoundaryEdges[:2, edge_idx].astype(int)
            x_edge = [MeshNodes[0, n1], MeshNodes[0, n2]]
            y_edge = [MeshNodes[1, n1], MeshNodes[1, n2]]
            ax_main.plot(x_edge, y_edge, 'k-', linewidth=1.5, alpha=0.8)

    ax_main.grid(True, alpha=0.3)
    ax_main.axis('equal')
    ax_main.legend(fontsize=8, loc='upper right')

    # === ГРАФИКИ 2-3: По ОТДЕЛЬНОСТИ ===
    for d in range(min(n_domains, 2)):
        ax = fig.add_subplot(gs[0, 1 + d])
        ax.set_title(f'Domain {d + 1}: {domain_types[d]}', fontsize=10)

        # Границы
        r_inner = domain_rx[d]
        r_outer = domain_rx[d + 1] if d + 1 < len(domain_rx) else np.max(domain_rx)

        if r_inner > 0:
            theta = np.linspace(0, 2 * np.pi, 100)
            xi = r_inner * np.cos(theta)
            yi = r_inner * np.sin(theta)
            ax.plot(xi, yi, 'k--', alpha=0.5)

        theta = np.linspace(0, 2 * np.pi, 100)
        xo = r_outer * np.cos(theta)
        yo = r_outer * np.sin(theta)
        ax.plot(xo, yo, 'k-', linewidth=2, alpha=0.8)

        # Элементы
        domain_mask = MeshTri[-1, :].astype(int) == d + 1
        domain_elements = np.where(domain_mask)[0]

        for el in domain_elements[:1000]:
            node_ids = MeshTri[:3, el].astype(int)
            nodes = MeshNodes[:, node_ids]
            x_coords = np.append(nodes[0, :], nodes[0, 0])
            y_coords = np.append(nodes[1, :], nodes[1, 0])
            ax.plot(x_coords, y_coords, '-', color=colors[d], linewidth=0.8, alpha=0.8)

        ax.grid(True, alpha=0.3)
        ax.axis('equal')

    # === ГРАФИК 4: Статистика ===
    ax_stats = fig.add_subplot(gs[1, :])
    ax_stats.axis('off')

    stats_text = f"Total elements: {n_elements}\n"
    stats_text += f"Total nodes: {MeshNodes.shape[1]}\n"
    stats_text += f"Domain radii: {domain_rx}\n"
    stats_text += f"Domain types: {domain_types}\n\n"

    unique_domain_markers = np.unique(MeshTri[-1, :].astype(int))
    stats_text += "Elements per domain:\n"
    for d in sorted(unique_domain_markers):
        count = np.sum(MeshTri[-1, :].astype(int) == d)
        stats_text += f"  Domain {d}: {count} elements\n"

    ax_stats.text(0.1, 0.95, stats_text.strip(),
                  transform=ax_stats.transAxes,
                  fontsize=9,
                  verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    fig.suptitle('MESH DEBUG - Domain Assignment Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    logger.info("  Showing mesh visualization...")
    plt.show()
    plt.close(fig)


def _validate_domain_assignment(MeshNodes: np.ndarray, MeshTri: np.ndarray, domain_rx: np.ndarray):
    """Проверка корректности назначения элементов доменам по радиусу"""
    logger.info("  Validating domain assignment...")

    tri_nodes = MeshTri[:3, :].astype(int)
    centers = np.mean(MeshNodes[:, tri_nodes], axis=1)
    radii = np.sqrt(centers[0, :] ** 2 + centers[1, :] ** 2)
    domain_markers = MeshTri[-1, :].astype(int)

    n_errors = 0
    error_threshold = 0.01

    for el in range(len(domain_markers)):
        r = radii[el]
        domain_id = domain_markers[el] - 1

        if domain_id < 0 or domain_id >= len(domain_rx) - 1:
            if n_errors < 5:
                logger.warning(f"    Element {el}: invalid domain marker = {domain_id + 1}")
            n_errors += 1
            continue

        r_inner = domain_rx[domain_id]
        r_outer = domain_rx[domain_id + 1] if domain_id + 1 < len(domain_rx) else np.inf

        if not (r_inner * (1 - error_threshold) <= r <= r_outer * (1 + error_threshold)):
            if n_errors < 5:
                logger.warning(
                    f"    Element {el}: r={r:.4f} not in domain {domain_id + 1} [{r_inner:.4f}, {r_outer:.4f}]")
            n_errors += 1

    if n_errors > 0:
        logger.warning(f"  Found {n_errors} elements with suspicious domain assignment")
    else:
        logger.info("  Domain assignment validation PASSED")


def run_frequency_loop(CompStruct: CompStruct, InputParam: InputParam) -> None:
    """Main frequency loop with production-ready error handling."""
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    n_frequencies = len(CompStruct.Model['f_array'])
    n_expected_dofs = None

    DEBUG_SINGLE_FREQ = False
    if DEBUG_SINGLE_FREQ:
        logger.warning("=== DEBUG MODE: Running single frequency ===")
        CompStruct.Model['f_array'] = [5.0]
        n_frequencies = 1

    for freq_idx, freq in enumerate(CompStruct.Model['f_array'], 1):
        CompStruct.if_grid = freq_idx

        logger.info(f"\n{'=' * 70}")
        logger.info(f"Frequency {freq_idx}/{n_frequencies}: f={freq:.2f} kHz")

        try:
            # Stage 3: Matrix Assembly
            start_time = time.time()
            logger.info("  Stage 3: Assembling matrices...")

            CompStruct, FullMatrices = run_stage3_matrix_assembly(CompStruct)
            stage3_time = time.time() - start_time

            if n_expected_dofs is None:
                total_nodes = sum(len(nodes) for nodes in CompStruct.FEMatrices['DNodes'].values())
                n_expected_dofs = total_nodes * max(CompStruct.Data.DVarNum)
                logger.info(f"    Expected DOFs: {n_expected_dofs}")

            logger.info(f"    Assembly time: {stage3_time:.1f}s")

            # Save FEMatrices
            fem_file = output_dir / f"FEMatrices_f{freq:.1f}.mat"
            savemat(str(fem_file), {
                'frequency': freq,
                'if_grid': freq_idx,
                'FullMatrices': FullMatrices,
                'FEMatrices': CompStruct.FEMatrices,
                'MeshTri': CompStruct.FEMatrices['MeshTri'],
                'MeshNodes': CompStruct.FEMatrices['MeshNodes'],
                'BoundaryEdges': CompStruct.FEMatrices['BoundaryEdges']
            })
            logger.info(f"    Saved FEMatrices: {fem_file.name}")

            # Stage 4: Eigenvalue Solution
            start_time = time.time()
            logger.info("  Stage 4: Solving eigenvalue problem...")

            Results = stage4_compute(
                CompStruct,
                CompStruct.Methods['BasicMatrices'],
                CompStruct.FEMatrices,
                FullMatrices
            )

            stage4_time = time.time() - start_time

            if Results['num_converged'] == 0:
                raise RuntimeError("No eigenvalues converged")

            logger.info(f"    Solution time: {stage4_time:.1f}s")
            logger.info(f"    Converged: {Results['num_converged']} eigenvalues")

            # Save Results
            results_file = output_dir / f"Results_f{freq:.1f}.mat"
            savemat(str(results_file), {
                'Results': Results,
                'frequency': freq,
                'if_grid': freq_idx
            })
            logger.info(f"    Saved Results: {results_file.name}")

            total_time = stage3_time + stage4_time
            logger.info(f"  [OK] Frequency {freq:.2f} kHz completed in {total_time:.1f}s")

        except Exception as e:
            logger.error(f"Fatal error at frequency {freq:.2f} kHz", exc_info=True)
            raise


def main():
    """Main execution with full pipeline"""
    print("\n" + "=" * 70)
    print("SAFE Anisotropic Spectral Analysis - Python Implementation v2.5")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    prog_start = time.time()

    try:
        # Stage 1
        print("\n[1] Stage 1: Model Initialization")

        param_file = select_parameter_file()
        validate_parameter_file(param_file)

        model_dir = param_file.parent
        logger.info(f"  Model directory: {model_dir}")
        logger.info(f"  Parameter file: {param_file.name}")

        with open(param_file, 'r') as f:
            json_data = json.load(f)

        validate_json_structure(json_data)

        InputParam = initialize_model(param_file)
        InputParam = register_physics_methods(InputParam)

        stage1_time = time.time() - prog_start
        print(f"    [OK] Stage 1 complete: {stage1_time:.1f}s")

        # Stage 2
        print("\n[2] Stage 2: Model Preparation & Mesh Generation")
        stage2_start = time.time()
        CompStruct = stage2.prepare_model(InputParam)

        # === Добавляем PML и логируем радиус ===
        CompStruct = setup_additional_domains(CompStruct)

        # === ВЫБОР ВИЗУАЛИЗАЦИИ ===
        if CompStruct.Mesh.output.lower() == 'yes':
            logger.info("  Select visualization:")
            logger.info("    1: GMSH (3D interactive)")
            logger.info("    2: Matplotlib (2D debug)")
            logger.info("    3: Both")
            logger.info("    0: Skip")

            # Для автоматизации можно задать в JSON
            vis_choice = CompStruct.Model.get('viz_mode', '2')

            if vis_choice == '1':
                visualize_mesh_gmsh(CompStruct)
            elif vis_choice == '2':
                visualize_mesh_final_debug(CompStruct)
            elif vis_choice == '3':
                visualize_mesh_gmsh(CompStruct)
                visualize_mesh_final_debug(CompStruct)
            else:
                logger.info("  Skipping visualization")

        stage2_time = time.time() - stage2_start
        print(f"    [OK] Stage 2 complete: {stage2_time:.1f}s")

        # Run pipeline
        run_frequency_loop(CompStruct, InputParam)

        # Finalization
        root_path = Path(__file__).parent
        from routines.io_utils import finalize_results
        finalize_results(root_path, model_dir.name)

        # Summary
        prog_total = time.time() - prog_start
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print(f"  Total time: {prog_total:.1f}s")
        print(f"  Results: models/{model_dir.name}/")
        print("=" * 70)

    except Exception as e:
        logger.error(f"\n{'=' * 70}\nPIPELINE FAILED: {str(e)}\n{'=' * 70}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()