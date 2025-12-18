#!/usr/bin/env python3
"""
Gen_Aniso - Main computation script for acoustic waveguide analysis
Step 1 & 2: Initialization and Model Preparation
"""

import sys
from pathlib import Path
import time
from typing import Optional

# Import from project modules
from config.structures import CompStruct, InputParam
from routines.st1_functions import St1_SetModel
from routines.st2_functions import St2_PrepareModel_sp_SAFE


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
            initialdir=Path.cwd() / "model"
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
    Implements Steps 1 and 2

    Parameters:
        model_name: Name of the model (optional)
        json_path: Path to JSON file (optional, opens dialog if not provided)
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

    # Step 2: Prepare Model
    print('Running Step 2: Preparing model for computation...\n')
    tStart_St2 = time.time()

    comp_struct = St2_PrepareModel_sp_SAFE(input_param)

    print(f'Time for Step 2 Program = {time.time() - tStart_St2:.1f}s\n')

    # Summary
    print('\n' + '=' * 80)
    print('Model Preparation Summary:')
    print('=' * 80)
    print(f"Problem Type: {comp_struct.Config.ProblemType}")
    print(f"Numerical Method: {comp_struct.Config.NumMethod}")
    print(f"Frequency Range: {comp_struct.Model.f_min:.1f} - {comp_struct.Model.f_max:.1f} kHz")
    print(f"Number of Frequencies: {comp_struct.Model.N_disp}")
    print(f"Number of Domains: {comp_struct.Data['N_domain']}")
    print(f"Domain Types: {', '.join(comp_struct.Model.DomainType)}")
    print(f"Variables per domain: {comp_struct.Data['DVarNum']}")
    print(f"Domain Radii (Rx): {comp_struct.Model.DomainRx} m")
    print(f"AddDomain Type: {comp_struct.Model.AddDomainType}")
    print(f"AddDomain Exists: {comp_struct.Model.AddDomain_Exist}")
    print(f"Max Eigenvalues: {comp_struct.Advanced.num_eig_max}")
    print(f"Search Start Velocity: {comp_struct.Advanced.EigSearchStart} km/s")
    print(f"Unit conversions: F={comp_struct.Misc['F_conv']}, S={comp_struct.Misc['S_conv']}")

    # List assigned methods
    print(f"\nAssigned methods: {len(comp_struct.Methods.PreparePhysProp)} domain methods, "
          f"{len(comp_struct.Methods.KM_el_matrix)} element matrices")
    print('=' * 80)

    print(f'\nTime for Gen_Aniso Program (Steps 1-2) = {time.time() - tStart_Prog:.1f}s')
    print('=' * 80 + '\n')

    return comp_struct


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    try:
        result = gen_aniso()

        print("\n✅ Steps 1-2 completed successfully!")
        print(f"Result structure ready with {result.Data['N_domain']} domains.")
        print("\nReady for Step 3: Preparing basic matrices...")

    except Exception as e:
        print(f"\n❌ Error during initialization: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)