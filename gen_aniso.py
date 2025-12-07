#!/usr/bin/env python3
"""
===============================================================================
SAFE (Spectral Analysis of Finite Elements) for Anisotropic Media
Main Driver Script - Python Implementation
===============================================================================
Replicates gen_aniso.m functionality with modern Python features:
- GUI directory selection for model JSON files
- JSON-based parameter input
- Sparse FEM matrix assembly
- Frequency-loop processing with timing
- MATLAB-compatible .mat output format
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
from core.config import InputParam, CompStruct
from methods.stage1 import initialize_model
from methods import stage2
from methods.stage3 import prepare_basic_matrices as stage3_prepare
from methods.stage4 import compute_solution as stage4_compute
from routines.io_utils import cleanup_output_dir

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def select_model_directory(initial_dir: Optional[Path] = None) -> Path:
    """
    Opens GUI dialog for user to select model directory containing JSON parameter file.

    Args:
        initial_dir: Starting directory for file dialog. Defaults to ./models/

    Returns:
        Path to selected model directory

    Raises:
        SystemExit: If no directory is selected
    """
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    if initial_dir is None:
        initial_dir = Path(__file__).parent / "models"

    selected_dir = filedialog.askdirectory(
        title="Select Model Directory (e.g., Bakken-B)",
        initialdir=str(initial_dir)
    )

    if not selected_dir:
        logger.error("No directory selected. Exiting.")
        sys.exit(1)

    return Path(selected_dir)


def find_parameter_file(model_dir: Path) -> Path:
    """
    Automatically finds the first .json file in the model directory.

    Args:
        model_dir: Directory to search for JSON files

    Returns:
        Path to parameter file

    Raises:
        SystemExit: If no JSON file is found
    """
    json_files = list(model_dir.glob("*.json"))

    if not json_files:
        logger.error(f"No JSON parameter file found in {model_dir}")
        sys.exit(1)

    param_file = json_files[0]
    logger.info(f"Found parameter file: {param_file.name}")
    return param_file


def setup_additional_domains(InputParam: InputParam) -> InputParam:
    """
    Replicates MATLAB logic for PML/ABC domain extension.
    Appends domain parameters if additional layer exists.

    Args:
        InputParam: Input parameter structure

    Returns:
        Modified InputParam with extended domains if needed
    """
    add_type = InputParam.Model['AddDomainType'].lower()

    # Normalize abc+pml to pml+abc
    if add_type == 'abc+pml':
        add_type = 'pml+abc'
        InputParam.Model['AddDomainType'] = add_type

    if add_type != 'none':
        valid_types = ['pml', 'abc', 'pml+abc', 'same']
        if add_type in valid_types:
            InputParam.Model['AddDomain_Exist'] = 'yes'

            if InputParam.Model['AddDomainLoc'].lower() == 'ext':
                # Append domain parameters for external layer
                InputParam.Model['DomainRx'].append(InputParam.Model['DomainRx'][-1])
                InputParam.Model['DomainRy'].append(InputParam.Model['DomainRy'][-1])
                InputParam.Model['DomainTheta'].append(InputParam.Model['DomainTheta'][-1])
                InputParam.Model['DomainEcc'].append(InputParam.Model['DomainEcc'][-1])
                InputParam.Model['DomainEccAngle'].append(InputParam.Model['DomainEccAngle'][-1])
                InputParam.Model['DomainParam'].append(InputParam.Model['DomainParam'][-1])
                InputParam.Model['DomainType'].append(InputParam.Model['DomainType'][-1])
                InputParam.Model['BCType'].append('rigid')
                InputParam.Model['DomainNth'].append(InputParam.Model['DomainNth'][-1])
                # Update second-to-last BC to SSstiff
                InputParam.Model['BCType'][-2] = 'SSstiff'

                logger.info(f"      Added {add_type.upper()} layer at external boundary")
        else:
            logger.error(f"Invalid AddDomainType: {add_type}")
            sys.exit(1)
    else:
        InputParam.Model['AddDomain_Exist'] = 'no'

    return InputParam


def validate_model_parameters(InputParam: InputParam) -> None:
    """
    Validates model parameter consistency.

    Args:
        InputParam: Input parameter structure to validate

    Raises:
        ValueError: If parameters are inconsistent
    """
    n_domains = len(InputParam.Model['DomainRx'])

    # Check array lengths
    required_arrays = ['DomainRy', 'DomainTheta', 'DomainEcc', 'DomainEccAngle',
                       'DomainType', 'DomainNth']
    for array_name in required_arrays:
        if len(InputParam.Model[array_name]) != n_domains:
            raise ValueError(
                f"Domain array length mismatch: {array_name} has "
                f"{len(InputParam.Model[array_name])} elements, expected {n_domains}"
            )

    # Check BCType length (should be n_domains + 1)
    if len(InputParam.Model['BCType']) != n_domains + 1:
        raise ValueError(
            f"BCType length mismatch: has {len(InputParam.Model['BCType'])} "
            f"elements, expected {n_domains + 1}"
        )

    # Check frequency array
    if len(InputParam.Model['f_array']) == 0:
        raise ValueError("Frequency array is empty")

    # Check PML method if PML/ABC is used
    if InputParam.Model['AddDomain_Exist'] == 'yes':
        if InputParam.Model['PML_method'] not in [1, 2]:
            raise ValueError("PML_method must be 1 (circular) or 2 (rectangular)")

    logger.info("    Parameter validation passed")


def prepare_output_directory(root_path: Path, dir_name: str) -> Path:
    """
    Creates/cleans output directory structure for temporary files.

    Args:
        root_path: Project root path
        dir_name: Model directory name

    Returns:
        Path to output directory
    """
    output_dir = root_path / "output"
    output_dir.mkdir(exist_ok=True)
    cleanup_output_dir(output_dir)  # Remove old files
    logger.info(f"  Output directory: {output_dir}")
    return output_dir


def run_frequency_loop(CompStruct: CompStruct, InputParam: InputParam) -> None:
    """
    Main frequency loop - executes Stage 3 (matrix assembly) and Stage 4 (solution)
    for each frequency point.

    Args:
        CompStruct: Computation structure with prepared model
        InputParam: Original input parameters
    """
    root_path = Path(CompStruct.Config.root_path)
    output_dir = root_path / "output"

    n_frequencies = len(CompStruct.Model['f_array'])

    for freq_idx, freq in enumerate(CompStruct.Model['f_array'], 1):
        CompStruct.if_grid = freq_idx

        logger.info(f"\n{'=' * 70}")
        logger.info(f"Frequency {freq_idx}/{n_frequencies}: f={freq:.2f} kHz")

        try:
            # Stage 3: Matrix Assembly
            start_time = time.time()
            logger.info("  Stage 3: Assembling matrices...")

            BasicMatrices, FEMatrices, FullMatrices = stage3_prepare(
                CompStruct, InputParam
            )

            stage3_time = time.time() - start_time
            logger.info(f"    Matrix assembly: {stage3_time:.1f}s")

            # Save FEMatrices to .mat file (MATLAB compatibility)
            # Note: We don't save the entire CompStruct to avoid redundancy
            fem_file = output_dir / f"FEMatrices-{freq:.1f}.mat"
            fem_dict = {
                'DomainRx': np.array(CompStruct.Model['DomainRx']),
                'DomainRy': np.array(CompStruct.Model['DomainRy']),
                'frequency': freq,
                'if_grid': freq_idx,
                # Extract key matrices and mesh data
                'MMatrix_d': FEMatrices.get('MMatrix_d'),
                'MeshNodes': FEMatrices.get('MeshNodes'),
                'BoundaryEdges': FEMatrices.get('BoundaryEdges'),
                'MeshTri': FEMatrices.get('MeshTri'),
                'MeshProps': FEMatrices.get('MeshProps'),
                'PhysProp': FEMatrices.get('PhysProp'),
                'DElements': FEMatrices.get('DElements'),
                'DEMeshProps': FEMatrices.get('DEMeshProps'),
                'DNodes': FEMatrices.get('DNodes'),
                'DNodesRem': FEMatrices.get('DNodesRem'),
                'DNodesComp': FEMatrices.get('DNodesComp'),
                'DTakeFromVarPos': FEMatrices.get('DTakeFromVarPos'),
                'DPutToVarPos': FEMatrices.get('DPutToVarPos'),
                'DZeroVarPos': FEMatrices.get('DZeroVarPos'),
                'BNodes': FEMatrices.get('BNodes'),
                'BNodesFull': FEMatrices.get('BNodesFull'),
            }

            # Filter out None values
            fem_dict = {k: v for k, v in fem_dict.items() if v is not None}

            savemat(str(fem_file), fem_dict)
            logger.info(f"    Saved FEMatrices to {fem_file.name}")

            # Stage 4: Eigenvalue Solution
            start_time = time.time()
            logger.info("  Stage 4: Solving eigenvalue problem...")

            Results = stage4_compute(
                CompStruct, BasicMatrices, FEMatrices, FullMatrices
            )

            stage4_time = time.time() - start_time
            logger.info(f"    Solution time: {stage4_time:.1f}s")

            # Save Results
            results_file = output_dir / f"Results-{freq:.1f}.mat"
            savemat(str(results_file), {
                'Results': Results,
                'frequency': freq,
                'if_grid': freq_idx
            })
            logger.info(f"    Saved Results to {results_file.name}")

            # Cleanup visualization if enabled
            if CompStruct.Mesh.output == 'yes' and hasattr(CompStruct.Mesh, 'fig_handle'):
                try:
                    import matplotlib.pyplot as plt
                    plt.close(CompStruct.Mesh.fig_handle)
                except Exception as e:
                    logger.warning(f"Could not close mesh figure: {e}")

            # Log total time for this frequency
            total_time = stage3_time + stage4_time
            logger.info(f"  Frequency {freq:.2f} kHz completed in {total_time:.1f}s")

        except Exception as e:
            logger.error(f"Error at frequency {freq:.2f} kHz: {str(e)}")
            raise


def finalize_results(root_path: Path, model_dir_name: str) -> None:
    """
    Moves computed results from output/ to model directory.

    Args:
        root_path: Project root path
        model_dir_name: Name of model directory (e.g., 'Bakken-B')
    """
    output_dir = root_path / "output"
    target_dir = root_path / "models" / model_dir_name

    # Ensure target directory exists
    target_dir.mkdir(parents=True, exist_ok=True)

    # Move all .mat files
    moved_files = []
    for mat_file in output_dir.glob("*.mat"):
        try:
            shutil.move(str(mat_file), str(target_dir / mat_file.name))
            moved_files.append(mat_file.name)
        except Exception as e:
            logger.error(f"Failed to move {mat_file.name}: {e}")

    if moved_files:
        logger.info(f"Moved {len(moved_files)} files to {target_dir}")
        for f in moved_files:
            logger.debug(f"  - {f}")
    else:
        logger.warning("No result files to move!")


def main():
    """
    Main execution function - orchestrates entire SAFE computation pipeline.
    """
    print("\n" + "=" * 70)
    print("SAFE Anisotropic Spectral Analysis - Python Implementation")
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    prog_start_time = time.time()

    try:
        # Stage 1: Model Initialization
        print("\n[1] Initializing model parameters...")
        model_dir = select_model_directory()
        json_file = find_parameter_file(model_dir)

        InputParam = initialize_model(json_file)
        InputParam = setup_additional_domains(InputParam)
        validate_model_parameters(InputParam)

        # Stage 2: Model Preparation
        print("\n[2] Preparing computation structure...")
        stage2_start = time.time()
        CompStruct = stage2.prepare_model(InputParam)
        stage2_time = time.time() - stage2_start
        print(f"    Time: {stage2_time:.1f}s")

        # Setup output directory
        root_path = Path(__file__).parent
        output_dir = prepare_output_directory(root_path, model_dir.name)

        # Stage 3 & 4: Frequency Loop
        print("\n[3] Starting frequency loop...")
        loop_start = time.time()
        run_frequency_loop(CompStruct, InputParam)
        loop_time = time.time() - loop_start

        # Finalization
        print("\n[4] Finalizing results...")
        finalize_results(root_path, model_dir.name)

        # Summary
        prog_total_time = time.time() - prog_start_time
        print("\n" + "=" * 70)
        print(f"Pipeline completed successfully!")
        print(f"  Total time: {prog_total_time:.1f}s")
        print(f"  Stage 2 prep: {stage2_time:.1f}s")
        print(f"  Frequency loop: {loop_time:.1f}s")
        print(f"  Results: models/{model_dir.name}/")
        print("=" * 70)

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()