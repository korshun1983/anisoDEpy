#!/usr/bin/env python3
"""
Gen_Aniso - Main computation script for acoustic waveguide analysis
Step 1: Initialization - Python implementation
"""

import sys
from pathlib import Path
import time
from typing import Optional

# Import from project modules
from config.structures import CompStruct, InputParam
from routines.st1_functions import St1_SetModel


# ============================================================================
# FILE DIALOG FUNCTION
# ============================================================================

def select_json_file() -> Optional[Path]:
    """
    Open file dialog to select JSON file
    Returns Path object or None if cancelled
    """
    try:
        from tkinter import Tk, filedialog

        root = Tk()
        root.withdraw()
        root.attributes('-topmost', True)

        file_path = filedialog.askopenfilename(
            title="Select JSON Model File",
            filetypes=[
                ("JSON files", "*.json"),
                ("All files", "*.*")
            ],
            initialdir = Path.cwd()/"models"
        )

        root.destroy()

        return Path(file_path) if file_path else None

    except ImportError:
        print("Warning: tkinter not available. Install python3-tk or specify file via command line.")
        return None
    except Exception as e:
        print(f"Warning: Could not open file dialog: {e}")
        return None


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def gen_aniso(model_name: str = None, json_path: str = None):
    """
    Main computation script - analogous to gen_aniso.m
    Currently implements only Step 1 (Initialization)
    """
    print('\n' + '=' * 80)
    print('Gen_Aniso Program has been started!')

    tStart_Prog = time.time()

    # Handle file selection
    if json_path is None:
        if len(sys.argv) > 1:
            json_path = sys.argv[1]
            if model_name is None and len(sys.argv) > 2:
                model_name = sys.argv[2]
        else:
            print('No JSON file specified. Opening file dialog...')
            selected_file = select_json_file()

            if selected_file is None:
                print("Error: No file selected. Exiting.")
                sys.exit(1)

            json_path = str(selected_file)
            if model_name is None:
                model_name = selected_file.stem

    model_file = Path(json_path)
    if not model_file.exists():
        print(f"Error: Model file '{json_path}' not found.")
        print(f"Working directory: {Path.cwd()}")

        print("\nWould you like to select a file manually? (y/n)")
        response = input().strip().lower()
        if response == 'y':
            selected_file = select_json_file()
            if selected_file and selected_file.exists():
                json_path = str(selected_file)
                model_file = selected_file
                if model_name is None:
                    model_name = selected_file.stem
            else:
                print("No valid file selected. Exiting.")
                sys.exit(1)
        else:
            sys.exit(1)

    if model_name is None:
        model_name = model_file.stem

    print(f'The used model is {model_name}')
    print('=' * 80 + '\n')

    # Step 1: Initialization
    print('Running Step 1: Setting up model parameters...\n')
    tStart_St1 = time.time()

    input_param = St1_SetModel(model_file)

    print(f'Time for Step 1 Program = {time.time() - tStart_St1:.1f}s\n')

    # Prepare CompStruct
    comp_struct = CompStruct(
        Config=input_param.Config,
        Model=input_param.Model,
        Advanced=input_param.Advanced,
        Methods=input_param.Methods,
        f_grid=input_param.Model.f_array,
        ModelInitial=input_param.Model
    )

    # Summary
    print('\n' + '=' * 80)
    print('Model Initialization Summary:')
    print('=' * 80)
    print(f"Problem Type: {input_param.Config.ProblemType}")
    print(f"Numerical Method: {input_param.Config.NumMethod}")
    print(f"Frequency Range: {input_param.Model.f_min:.1f} - {input_param.Model.f_max:.1f} kHz")
    print(f"Number of Frequencies: {input_param.Model.N_disp}")
    print(f"Number of Domains: {len(input_param.Model.DomainType)}")
    print(f"Domain Types: {', '.join(input_param.Model.DomainType)}")
    print(f"Domain Radii (Rx): {input_param.Model.DomainRx} m")
    print(f"AddDomain Type: {input_param.Model.AddDomainType}")
    print(f"AddDomain Exists: {input_param.Model.AddDomain_Exist}")
    print(f"Max Eigenvalues: {input_param.Advanced.num_eig_max}")
    print(f"Search Start Velocity: {input_param.Advanced.EigSearchStart} km/s")
    print('=' * 80)

    print(f'\nTime for Gen_Aniso Program (Step 1) = {time.time() - tStart_Prog:.1f}s')
    print('=' * 80 + '\n')

    return comp_struct


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    try:
        result = gen_aniso()

        print("\n[OK] Initialization completed successfully!")
        print(f"Result structure ready with {len(result.Model.DomainType)} domains.")
        print("\nReady for Step 2: Preparing model...")

    except Exception as e:
        print(f"\n[BAD] Error during initialization: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)