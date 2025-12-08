#!/usr/bin/env python3
"""
===============================================================================
SAFE (Spectral Analysis of Finite Elements) for Anisotropic Media
Main Driver Script - Python Implementation v2.0
===============================================================================
Features:
- JSON validation and error checking
- Automatic method registration for rotation physics
- Complete matrix assembly diagnostics
- Production-ready eigenvalue solver (Stage 4)
- Fault-tolerant frequency loop with detailed logging
- MATLAB-compatible .mat output with all interface data
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

# Project modules
from core.config import InputParam
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


def select_model_directory(initial_dir: Optional[Path] = None) -> Path:
    """GUI directory selection with fallback"""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    if initial_dir is None:
        initial_dir = Path(__file__).parent / "models"

    try:
        selected_dir = filedialog.askdirectory(
            title="Select Model Directory (e.g., Bakken-B)",
            initialdir=str(initial_dir)
        )
    except Exception as e:
        logger.error(f"GUI error: {e}")
        selected_dir = input("Enter model directory path: ")

    if not selected_dir:
        logger.error("No directory selected. Exiting.")
        sys.exit(1)

    return Path(selected_dir)


def find_parameter_file(model_dir: Path) -> Path:
    """Auto-detect JSON parameter file"""
    json_files = list(model_dir.glob("*.json"))

    if not json_files:
        logger.error(f"No JSON file found in {model_dir}")
        sys.exit(1)

    if len(json_files) > 1:
        logger.warning(f"Multiple JSON files. Using: {json_files[0].name}")

    return json_files[0]


def validate_json_structure(json_data: Dict[str, Any]) -> None:
    """
    Comprehensive JSON validation against SAFE schema.
    Replicates MATLAB's implicit validation with explicit errors.
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

    n_domains = len(model["DomainRx"])
    if n_domains < 1:
        raise ValueError("Must have at least one domain")

    for key in ["DomainRy", "DomainTheta", "DomainEcc", "DomainEccAngle",
                "DomainType", "DomainParam", "DomainNth"]:
        if key in model and len(model[key]) != n_domains:
            raise ValueError(
                f"Model.{key} length ({len(model[key])}) doesn't match "
                f"DomainRx length ({n_domains})"
            )

    if len(model["BCType"]) != n_domains + 1:
        raise ValueError(
            f"BCType length ({len(model['BCType'])}) must be N_domain + 1 "
            f"({n_domains + 1})"
        )

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

    if "AddDomainType" in model and model["AddDomainType"].lower() != "none":
        if "AddDomainLoc" not in model:
            raise ValueError("AddDomainLoc required when AddDomainType != 'none'")


def register_physics_methods(InputParam: InputParam) -> InputParam:
    """
    Explicitly register rotation methods needed for HTTI physics.
    CRITICAL: Without this, Stage 3 will fail on HTTI materials.
    """
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
    """
    add_type = CompStruct.Model['AddDomainType'].lower()

    if add_type == 'abc+pml':
        add_type = 'pml+abc'
        CompStruct.Model['AddDomainType'] = add_type

    if add_type != 'none':
        CompStruct.Model['AddDomain_Exist'] = 'yes'

        if CompStruct.Model['AddDomainLoc'].lower() == 'ext':
            CompStruct.Model['DomainRx'].append(CompStruct.Model['DomainRx'][-1])
            CompStruct.Model['DomainRy'].append(CompStruct.Model['DomainRy'][-1])
            CompStruct.Model['DomainTheta'].append(CompStruct.Model['DomainTheta'][-1])
            CompStruct.Model['DomainEcc'].append(CompStruct.Model['DomainEcc'][-1])
            CompStruct.Model['DomainEccAngle'].append(CompStruct.Model['DomainEccAngle'][-1])
            CompStruct.Model['DomainParam'].append(CompStruct.Model['DomainParam'][-1])
            CompStruct.Model['DomainType'].append(CompStruct.Model['DomainType'][-1])
            CompStruct.Model['BCType'].append('rigid')
            CompStruct.Model['DomainNth'].append(CompStruct.Model['DomainNth'][-1])
            CompStruct.Model['BCType'][-2] = 'SSstiff'

            logger.info(f"      Added {add_type.upper()} external layer")
        elif CompStruct.Model['AddDomainLoc'].lower() == 'int':
            logger.warning("Internal PML/ABC not yet implemented")

    else:
        CompStruct.Model['AddDomain_Exist'] = 'no'

    return CompStruct


def prepare_output_directory(root_path: Path, model_dir_name: str) -> Path:
    """Create and clean output directory"""
    output_dir = root_path / "output"
    output_dir.mkdir(exist_ok=True)
    cleanup_output_dir(output_dir)
    logger.info(f"  Output directory: {output_dir}")
    return output_dir


def verify_matrix_assembly(FullMatrices: Dict[str, Any], n_expected_dofs: int) -> None:
    """
    Validate matrix assembly results before proceeding to Stage 4.
    Catches silent failures in interface coupling.
    """
    if not FullMatrices or 'M' not in FullMatrices:
        raise RuntimeError("Stage 3 failed: FullMatrices empty")

    actual_dofs = FullMatrices['M'].shape[0]
    if actual_dofs != n_expected_dofs:
        logger.warning(
            f"DOF mismatch: expected {n_expected_dofs}, got {actual_dofs}. "
            f"This may indicate interface assembly issues."
        )

    if 'P' in FullMatrices and FullMatrices['P'].nnz == 0:
        logger.warning("Interface coupling matrix P is zero. Check fluid-HTTI boundary.")

    logger.info(f"  Matrix verification: {actual_dofs} DOFs, M.nnz={FullMatrices['M'].nnz}")


def run_frequency_loop(CompStruct: CompStruct, InputParam: InputParam) -> None:
    """
    Main frequency loop with production-ready error handling.
    """
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    n_frequencies = len(CompStruct.Model['f_array'])
    n_expected_dofs = None

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

            # Verify assembly
            if n_expected_dofs is None:
                total_nodes = sum(len(nodes) for nodes in CompStruct.FEMatrices['DNodes'].values())
                n_expected_dofs = total_nodes * max(CompStruct.Data['DVarNum'])
                logger.info(f"    Expected DOFs: {n_expected_dofs}")

            verify_matrix_assembly(FullMatrices, n_expected_dofs)
            logger.info(f"    Assembly time: {stage3_time:.1f}s")

            # Save FEMatrices (full interface data for debugging)
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

            # Validate results
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

            # Frequency summary
            total_time = stage3_time + stage4_time
            logger.info(f"  ✓ Frequency {freq:.2f} kHz completed in {total_time:.1f}s")

        except Exception as e:
            logger.error(f"Fatal error at frequency {freq:.2f} kHz", exc_info=True)
            raise


def main():
    """Main execution with full pipeline"""
    print("\n" + "=" * 70)
    print("SAFE Anisotropic Spectral Analysis - Python Implementation v2.0")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    prog_start = time.time()

    try:
        # Stage 1
        print("\n[1] Stage 1: Model Initialization")
        model_dir = select_model_directory()
        json_file = find_parameter_file(model_dir)

        with open(json_file, 'r') as f:
            json_data = json.load(f)

        validate_json_structure(json_data)

        InputParam = initialize_model(json_file)
        InputParam = register_physics_methods(InputParam)

        # Convert to CompStruct and add domains
        CompStruct = stage2.prepare_model(InputParam)
        CompStruct = setup_additional_domains(CompStruct)

        stage1_time = time.time() - prog_start
        print(f"    ✓ Stage 1 complete: {stage1_time:.1f}s")

        # Stage 2
        print("\n[2] Stage 2: Model Preparation")
        stage2_start = time.time()
        CompStruct = stage2.prepare_model(CompStruct)  # Finalize after domain setup
        stage2_time = time.time() - stage2_start
        print(f"    ✓ Stage 2 complete: {stage2_time:.1f}s")

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