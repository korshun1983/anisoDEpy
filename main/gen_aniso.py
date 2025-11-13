import os
import sys
import time
import json
import logging
from typing import Dict, Any, List
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# ===============================================================================
# gen_aniso.py - Simplified main computation script for waveguide mode analysis
# No inputs required - configuration loaded from JSON
# ===============================================================================

def load_config(case_dir: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file in the case directory.
    Assumes the JSON file is named {case_name}.json (e.g., Bakken-B.json)
    """
    json_file = os.path.join(case_dir, f"{os.path.basename(case_dir)}.json")
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"Configuration file not found: {json_file}")

    with open(json_file, 'r') as f:
        config = json.load(f)

    logger.info(f"Loaded configuration from {json_file}")
    return config


def generate_mesh_for_frequency(
        config: Dict[str, Any],
        frequency_khz: float,
        case_dir: str
) -> Dict[str, Any]:
    """
    Generate mesh for a specific frequency using the mesh generator.
    This replaces the MATLAB-based mesh generation with our Python version.
    """
    # Import here to avoid dependency if not needed
    from input_param_2mesh import SimpleMeshGenerator

    # Temporarily create a mesh generator instance
    mesh_gen = SimpleMeshGenerator.__new__(SimpleMeshGenerator)
    mesh_gen.config = config
    mesh_gen.frequency_khz = frequency_khz
    mesh_gen.hmax = mesh_gen.calculate_wavelength_based_hmax(frequency_khz)
    mesh_gen.has_additional_domain = (
            config['Model'].get('AddDomainLoc') == 'ext' and
            config['Model'].get('AddDomainType', 'none').lower() not in ['none', 'same']
    )
    mesh_gen.domain_rx = config['Model']['DomainRx']
    mesh_gen.domain_ry = config['Model']['DomainRy']
    mesh_gen.domain_theta = config['Model']['DomainTheta']
    mesh_gen.domain_ecc = config['Model']['DomainEcc']
    mesh_gen.domain_ecc_angle = config['Model']['DomainEccAngle']
    mesh_gen.domain_nth = config['Model']['DomainNth']
    mesh_gen.add_domain_L = config['Model'].get('AddDomainL', 1.0)
    mesh_gen.add_domain_type = config['Model'].get('AddDomainType', 'abc')
    mesh_gen.ext_boundary_shape = config['Mesh'].get('ext_boundary_shape', 'cir')

    mesh_gen.gmsh_initialized = False

    try:
        mesh_gen.generate_mesh()
        mesh_gen.extract_mesh_data()
        return mesh_gen.mesh_data
    finally:
        mesh_gen.finalize_gmsh()


def compute_dispersion_curve(config: Dict[str, Any], case_dir: str) -> None:
    """
    Main computation loop: generate mesh, assemble matrices, solve for each frequency
    """
    model_config = config['Model']
    frequency_range = model_config['f_array_range']
    f_start = frequency_range['start']
    f_end = frequency_range['end']
    f_step = frequency_range.get('step', 0.5)

    frequencies_khz = np.arange(f_start, f_end + f_step, f_step)
    logger.info(f"Computing dispersion curve for {len(frequencies_khz)} frequency points")

    # Initialize results storage
    all_results = []

    for idx, freq_khz in enumerate(frequencies_khz):
        logger.info(f"Processing frequency {idx + 1}/{len(frequencies_khz)}: {freq_khz} kHz")

        # Step 1 & 2: Generate mesh for this frequency (replaces St1_SetModel and St2_PrepareModel)
        t_mesh = time.perf_counter()
        mesh_data = generate_mesh_for_frequency(config, freq_khz, case_dir)
        logger.debug(f"Mesh generation: {time.perf_counter() - t_mesh:.1f}s")

        # Step 3: Prepare matrices (replaces St3_PrepareBasicMatrices)
        t_matrices = time.perf_counter()
        BasicMatrices, FEMatrices = assemble_matrices(config, mesh_data, freq_khz)
        logger.debug(f"Matrix assembly: {time.perf_counter() - t_matrices:.1f}s")

        # Step 4: Solve eigenvalue problem (replaces St4_ComputeSolution)
        t_solve = time.perf_counter()
        Results = solve_eigenvalue_problem(config, BasicMatrices, FEMatrices, freq_khz)
        logger.debug(f"Solution: {time.perf_counter() - t_solve:.1f}s")

        # Store results
        all_results.append({
            'frequency_khz': freq_khz,
            'eigenvalues': Results['eigenvalues'],
            'eigenvectors': Results['eigenvectors'],
            'mesh_info': {
                'num_nodes': len(mesh_data['nodes']),
                'num_elements': sum(len(e['tags']) for e in mesh_data['elements'].values())
            }
        })

        logger.info(f"  Completed in {time.perf_counter() - t_mesh:.1f}s")

    # Save final results
    output_file = os.path.join(case_dir, 'dispersion_results.json')
    save_results(all_results, output_file)
    logger.info(f"Results saved to {output_file}")


def assemble_matrices(config: Dict[str, Any], mesh_data: Dict[str, Any], freq_khz: float) -> tuple:
    """
    Assemble the spectral method matrices for the eigenvalue problem.
    This replaces St3_PrepareBasicMatrices.
    """
    # Extract mesh data
    nodes = mesh_data['nodes']
    elements = mesh_data['elements']
    element_domains = mesh_data['element_domains']

    # Get material properties from config
    domain_types = config['Model']['DomainType']
    domain_params = config['Model']['DomainParam']

    # Build mass and stiffness matrices using SAFE method
    # This is a placeholder for the actual implementation
    # The real implementation would use finite element assembly
    n_dofs = len(nodes) * 3  # Assuming 3 DOFs per node (e.g., ux, uy, uz)

    # Initialize sparse matrices
    from scipy.sparse import lil_matrix

    M = lil_matrix((n_dofs, n_dofs))  # Mass matrix
    K = lil_matrix((n_dofs, n_dofs))  # Stiffness matrix

    # Assemble element matrices
    # ... (actual FEM assembly code would go here)

    # For now, return dummy matrices
    BasicMatrices = {
        'M': M,
        'K': K,
        'n_dofs': n_dofs
    }

    FEMatrices = {
        'mesh_nodes': nodes,
        'mesh_elements': elements,
        'element_domains': element_domains,
        'freq_khz': freq_khz
    }

    return BasicMatrices, FEMatrices


def solve_eigenvalue_problem(
        config: Dict[str, Any],
        BasicMatrices: Dict[str, Any],
        FEMatrices: Dict[str, Any],
        freq_khz: float
) -> Dict[str, Any]:
    """
    Solve the generalized eigenvalue problem: (K - ω²M)φ = 0
    This replaces St4_ComputeSolution
    """
    from scipy.sparse.linalg import eigsh
    from scipy.sparse import identity

    K = BasicMatrices['K']
    M = BasicMatrices['M']
    n_modes = config.get('Advanced', {}).get('N_modes', 10)

    # Convert frequency to angular frequency
    omega = 2 * np.pi * freq_khz * 1000  # rad/s

    # Form the dynamic stiffness matrix
    # For waveguide problem: [K - ω²M]φ = 0 becomes eigenvalue problem in k (wavenumber)
    # This is a simplified version - actual SAFE method is more complex
    A = K - omega ** 2 * M

    # Solve for eigenvalues (k²) and eigenvectors
    # Using shift-invert mode to find modes near zero
    try:
        eigenvalues, eigenvectors = eigsh(
            A,
            k=min(n_modes, A.shape[0] - 1),
            M=M,
            sigma=0,
            which='LM'
        )
    except Exception as e:
        logger.warning(f"Eigenvalue solve failed at {freq_khz} kHz: {e}")
        eigenvalues = np.array([])
        eigenvectors = np.array([])

    return {
        'frequency_khz': freq_khz,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors
    }


def save_results(results: List[Dict[str, Any]], output_file: str) -> None:
    """Save dispersion curve results to JSON file"""
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = []
    for r in results:
        serializable_results.append({
            'frequency_khz': r['frequency_khz'],
            'eigenvalues': r['eigenvalues'].tolist() if hasattr(r['eigenvalues'], 'tolist') else r['eigenvalues'],
            'eigenvectors': r['eigenvectors'].tolist() if hasattr(r['eigenvectors'], 'tolist') else r['eigenvectors'],
            'mesh_info': r['mesh_info']
        })

    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    logger.info(f"Saved {len(results)} frequency points to {output_file}")


def main():
    """Main entry point"""
    print('\n' + '=' * 75)
    print('GEN_ANISO - Waveguide Mode Analysis (Python Version)')
    print('=' * 75)

    # Select case directory (uncomment desired case)
    # case_dir = 'Mesaverde-30'
    # case_dir = 'Mesaverde-60'
    # case_dir = 'Mesaverde-HTI'
    # case_dir = 'Mesaverde-0C'
    # case_dir = 'Mesaverde-1'
    # case_dir = 'Mesaverde-0-F'
    # case_dir = 'Mesaverde-30-F'
    # case_dir = 'Mesaverde-60-F'
    # case_dir = 'Mesaverde-HTI-F'
    # case_dir = 'Mesaverde-HTI-F1'
    # case_dir = 'Mesaverde-HTI-F2'
    case_dir = 'Bakken-B'
    # case_dir = 'Bakken-HTI'
    # case_dir = 'Bakken-HTI-F'
    # case_dir = 'Cotton-30'
    # case_dir = 'Cotton-60'
    # case_dir = 'Cotton-HTI'
    # case_dir = 'Usari25-HTI'
    # case_dir = 'Hornby1-HTI'
    # case_dir = 'Minas-HTI'
    # case_dir = 'Horne-HTI'
    # case_dir = 'Barnett-HTI'
    # case_dir = 'JH-HTI'
    # case_dir = 'Deger-HTI'
    # case_dir = 'JJ-HTI'
    # case_dir = 'Austin-HTI'
    # case_dir = 'SlowForm-HTI'
    # case_dir = 'FastForm-HTI'

    if not os.path.isdir(case_dir):
        logger.error(f"Case directory not found: {case_dir}")
        sys.exit(1)

    logger.info(f"Selected case: {case_dir}")

    try:
        # Load configuration
        config = load_config(case_dir)

        # Run computation
        t_start = time.perf_counter()
        compute_dispersion_curve(config, case_dir)

        logger.info(f"Total computation time: {time.perf_counter() - t_start:.1f}s")
        print('=' * 75)

    except Exception as e:
        logger.error(f"Computation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()