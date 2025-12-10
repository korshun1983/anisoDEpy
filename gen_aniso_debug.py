#!/usr/bin/env python3
"""
===============================================================================
SAFE MESH DEBUG DRIVER - Only Inner Domain
ЗАПУСКАЙТЕ ЭТОТ СКРИПТ ДЛЯ ОТЛАДКИ ВМЕСТО gen_aniso.py
===============================================================================
"""

import json
import logging
import sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler('debug_mesh.log', mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Project modules
from core.config import InputParam, CompStruct
from methods.stage1 import initialize_model
from routines.matrix_assembly import em_tensor_vti, rotate_c_ij, rot_matrix
from routines.meshgen_debug import prepare_mesh_debug_only_inner


def main():
    print("\n" + "=" * 70)
    print("SAFE MESH DEBUG - Inner Domain Only")
    print("=" * 70)

    # Выбор файла параметров
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    param_file = Path(filedialog.askopenfilename(
        title="Select JSON parameter file",
        initialdir=str(Path(__file__).parent / "models"),
        filetypes=[("JSON files", "*.json")]
    ))

    if not param_file:
        logger.error("No file selected")
        sys.exit(1)

    logger.info(f"Loading parameters from {param_file}")

    # Инициализация модели
    InputParam = initialize_model(param_file)

    # Регистрация методов
    InputParam.Methods['em_tensor_VTI'] = em_tensor_vti
    InputParam.Methods['rot_c_ij'] = rotate_c_ij
    InputParam.Methods['rot_matrix'] = rot_matrix

    # Создание CompStruct (ИСПРАВЛЕНО: переименована переменная)
    # Используем 'comp_struct_instance' чтобы избежать конфликта имен
    comp_struct_instance = CompStruct()
    comp_struct_instance.Model = InputParam.Model
    comp_struct_instance.Mesh = InputParam.Mesh
    comp_struct_instance.Advanced = InputParam.Advanced
    comp_struct_instance.Data = InputParam.Data
    comp_struct_instance.Config = InputParam.Config

    # Генерация сетки ТОЛЬКО для внутреннего домена
    logger.info("=" * 70)
    logger.info("GENERATING MESH FOR INNER DOMAIN ONLY")
    logger.info(f"Domain radius: {comp_struct_instance.Model['DomainRx'][0]} m")
    logger.info("=" * 70)

    mesh_data = prepare_mesh_debug_only_inner(comp_struct_instance)

    # Сохранение результата
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    logger.info("Saving debug mesh...")
    from scipy.io import savemat
    savemat(output_dir / "debug_inner_mesh.mat", {
        'MeshNodes': mesh_data['MeshNodes'],
        'MeshTri': mesh_data['MeshTri'],
        'MeshProps': mesh_data['MeshProps']
    })

    print("\n" + "=" * 70)
    print("DEBUG COMPLETED")
    print(f"Results saved to: {output_dir}/debug_inner_mesh.mat")
    print("Check debug_mesh.log for details")
    print("=" * 70)


if __name__ == "__main__":
    main()