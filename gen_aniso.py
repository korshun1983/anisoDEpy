#!/usr/bin/env python3
"""
gen_aniso.py
============
EXACT MATLAB equivalent - main entry point with JSON file selection dialog.
"""

import os
import sys
import json
from pathlib import Path

# For file dialog
try:
    import tkinter as tk
    from tkinter import filedialog

    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
    debug_print("WARNING: tkinter not available, using command-line fallback", level=1)

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from stage1.st1_set_model import st1_set_model
from stage2.st2_prepare_model_sp_safe import st2_prepare_model_sp_safe
from utils import debug_print, timer


def select_model_file() -> str:
    """
    Open file dialog to select JSON model file.
    Returns empty string if dialog canceled or no GUI available.
    """
    if not TKINTER_AVAILABLE:
        return ""

    try:
        root = tk.Tk()
        root.withdraw()  # Hide main window
        root.attributes('-topmost', True)  # Bring to front

        # Default directory: models/
        models_dir = Path(__file__).parent / "models"
        initial_dir = str(models_dir) if models_dir.exists() else "."

        file_path = filedialog.askopenfilename(
            title="Select JSON Model File for SAFE Solver",
            filetypes=[
                ("JSON Model Files", "*.json"),
                ("All Files", "*.*")
            ],
            initialdir=initial_dir
        )

        root.destroy()  # Clean up
        return file_path
    except Exception as e:
        debug_print(f"Could not open file dialog: {e}", level=1)
        return ""


def load_model_from_json(file_path: str) -> dict:
    """Load model configuration from JSON file"""
    try:
        with open(file_path, 'r') as f:
            model_data = json.load(f)
        debug_print(f"Loaded model from: {Path(file_path).name}", level=2)
        debug_print(f"  Domains: {len(model_data.get('Model', {}).get('DomainType', []))}", level=3)
        debug_print(f"  Frequency range: {model_data.get('Model', {}).get('f_array_range', {})}", level=3)
        return model_data
    except Exception as e:
        debug_print(f"ERROR loading JSON file: {e}", level=0)
        raise


def main():
    """
    Main execution function - exact MATLAB workflow with model file selection.
    """
    debug_print("=" * 80, level=1)
    debug_print("SAFE SOLVER (Python Implementation)", level=1)
    debug_print("Based on ANISO_SAFE D5 - Timur Zharnikov", level=1)
    debug_print("=" * 80, level=1)

    # =========================================================================
    # Stage 0: Model File Selection
    # =========================================================================
    debug_print("STAGE 0: Selecting Model File", level=1)

    model_file = select_model_file()

    if not model_file:
        # Fallback to default model
        model_file = Path(__file__).parent / "models" / "bakken_b.json"

        if not model_file.exists():
            debug_print("ERROR: Default model file not found!", level=0)
            debug_print("Please create models/bakken_b.json or select a model file.", level=0)
            return 1
        else:
            debug_print(f"Using default model: {model_file}", level=2)

    # Load model data
    model_data = load_model_from_json(str(model_file))

    # =========================================================================
    # Stage 1: Model Setup
    # =========================================================================
    with timer("STAGE 1: Model Initialization"):
        InputParam = st1_set_model(model_data)  # Pass model data to stage1

    # =========================================================================
    # Stage 2: Model Preparation
    # =========================================================================
    with timer("STAGE 2: Model Preparation"):
        CompStruct = st2_prepare_model_sp_safe(InputParam)

    # =========================================================================
    # Stage 3: Mesh Generation
    # =========================================================================
    debug_print("STAGE 3: Mesh Generation", level=1)
    with timer("Mesh Generation"):
        MeshNodes, BoundaryEdges, MeshTri, MeshProps, CompStruct = CompStruct['Methods']['PrepareMesh'](CompStruct)

    debug_print(f"Mesh generation complete: {MeshNodes.shape[1]} nodes, {MeshTri.shape[1]} elements", level=1)

    # Verify Stage 1 results
    _verify_stage1(InputParam)

    debug_print("=" * 80, level=1)
    debug_print("ALL STAGES COMPLETED SUCCESSFULLY", level=1)
    debug_print("Model and mesh are ready for matrix assembly", level=1)
    debug_print("=" * 80, level=1)

    return 0


def _verify_stage1(InputParam):
    """Verify Stage 1 output structure"""
    debug_print("Verifying Stage 1 output...", level=2)

    required_keys = ['Config', 'Methods', 'Model', 'Advanced']
    for key in required_keys:
        if key not in InputParam:
            raise KeyError(f"Stage 1 missing required key: {key}")

    # Verify Config
    config_keys = ['ProblemType', 'NumMethod', 'root_path', 'solver_path']
    for key in config_keys:
        if key not in InputParam['Config']:
            raise KeyError(f"Config missing key: {key}")

    # Verify Model
    model_keys = ['DomainRx', 'DomainRy', 'DomainType', 'f_array', 'N_disp']
    for key in model_keys:
        if key not in InputParam['Model']:
            raise KeyError(f"Model missing key: {key}")

    debug_print(f"  Config: {len(InputParam['Config'])} parameters", level=3)
    debug_print(
        f"  Model: {InputParam['Model']['N_disp']} frequencies, {len(InputParam['Model']['DomainType'])} layers",
        level=3)
    debug_print(f"  Advanced: {len(InputParam['Advanced'])} parameters", level=3)
    debug_print("Stage 1 verification PASSED", level=2)


if __name__ == "__main__":
    sys.exit(main())