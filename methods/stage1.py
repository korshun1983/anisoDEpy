# methods/stage1.py
"""
===============================================================================
Stage 1: Model Initialization Pipeline
Replicates MATLAB St1_SetModel.m orchestration logic
===============================================================================
"""

from pathlib import Path
from typing import Dict, Callable
from core.config import InputParam, MethodsContainer
from methods import stage2, stage3, stage4
from routines import meshgen, matrix_assembly, asymptotes


# -----------------------------------------------------------------------------
# Sub-stage implementations
# -----------------------------------------------------------------------------

def st1_1_set_model_config() -> InputParam:
    """
    Replicates St1_1_SetModelConfig.m
    Initializes default configuration parameters.
    """
    return InputParam()  # Already has all defaults from dataclass


def st1_2_prepare_model_methods(InputParam: InputParam) -> InputParam:
    """
    Replicates St1_2_PrepareModelMethods.m
    Assigns function handles based on configuration.
    """
    # Set root path (project directory)
    root_path = Path(__file__).parent.parent
    InputParam.Config.root_path = root_path

    # Assign utility methods (always available)
    InputParam.Methods['chebdif'] = None  # Will be implemented when needed
    InputParam.Methods['em_tensor_VTI'] = matrix_assembly.em_tensor_vti
    InputParam.Methods['rot_c_ij'] = matrix_assembly.rotate_c_ij
    InputParam.Methods['rot_matrix'] = matrix_assembly.rotation_matrix
    InputParam.Methods['V_phase_VTI_exact_RPH'] = asymptotes.v_phase_vti_exact_rph
    InputParam.Methods['MeshFaces'] = meshgen.mesh_faces
    InputParam.Methods['em_tensor_VTI'] = matrix_assembly.em_tensor_vti
    InputParam.Methods['rot_c_ij'] = matrix_assembly.rotate_c_ij
    InputParam.Methods['rot_matrix'] = matrix_assembly.rot_matrix

    # Problem-specific method branching
    if InputParam.Config.ProblemType == 'spectrum':
        if InputParam.Config.NumMethod == 'SAFE':
            # Asymptotes for SAFE method
            InputParam.Methods['ComputeAsymptotes'] = asymptotes.compute_asymptotes_safe

            # Stage 2 methods
            InputParam.Methods['St2_PrepareModel'] = stage2.prepare_model
            InputParam.Methods['St2_1_PrepareModelParams'] = stage2.prepare_model_params
            InputParam.Methods['St2_2_PrepareModelMethods'] = stage2.prepare_model_methods

            # Stage 3 methods
            InputParam.Methods['St3_PrepareBasicMatrices'] = stage3.prepare_basic_matrices

            # Stage 4 methods
            InputParam.Methods['St4_ComputeSolution'] = stage4.compute_solution

            # Set solver path (for debugging/info only)
            solver_path = root_path / 'methods' / 'spectrum'
            InputParam.Config.solver_path = solver_path

    return InputParam


def st1_3_set_model_user(InputParam: InputParam, json_file: Path) -> InputParam:
    """
    Replicates St1_3_SetModelUser_sp_SAFE.m but loads from JSON
    instead of hardcoded MATLAB parameters.
    """
    from core.config import create_input_from_json

    # Load user parameters from JSON
    with open(json_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    # Merge with existing InputParam
    InputParam = create_input_from_json(json_data)
    InputParam._json_file = json_file

    # Post-processing (replicates MATLAB logic)
    if 'f_array' not in InputParam.Model:
        # Generate frequency array from range if needed
        far = InputParam.Model['f_array_range']
        InputParam.Model['f_array'] = np.arange(
            far['start'], far['end'] + far['step'] / 2, far['step']
        ).tolist()

    InputParam.Model['N_disp'] = len(InputParam.Model['f_array'])

    return InputParam


def st1_4_set_model_advanced(InputParam: InputParam) -> InputParam:
    """
    Replicates St1_4_SetModelAdvanced_sp_SAFE.m
    Sets advanced parameters that should not be modified by typical users.
    """
    # Mesh visualization flag
    InputParam.Advanced.VisualizeMesh = True

    # Element order (cubic elements with 10 nodes)
    InputParam.Advanced.N_nodes = 10
    InputParam.Advanced.NEdge_nodes = 4

    # Eigenvalue solver options
    InputParam.Advanced.EigsOptions.disp = 0
    InputParam.Advanced.EigsOptions.tol = 1e-8

    # Source parameters (for excitation classification)
    InputParam.Advanced.Source.xc = 0.0
    InputParam.Advanced.Source.yc = 0.0
    InputParam.Advanced.Source.r0x = 0.06
    InputParam.Advanced.Source.r0y = 0.06
    InputParam.Advanced.Source.theta_r = 0.0
    InputParam.Advanced.Source.theta0 = 0.0
    InputParam.Advanced.Source.sigma = 0.02
    InputParam.Advanced.Source.Plim = 5e-4
    InputParam.Advanced.Source.symmetry = 0

    return InputParam


# -----------------------------------------------------------------------------
# Main orchestrator
# -----------------------------------------------------------------------------

def initialize_model(json_file: Path) -> InputParam:
    """
    Main Stage 1 orchestrator - replicates St1_SetModel.m
    Executes the full initialization pipeline in correct order.
    """
    logger = logging.getLogger(__name__)
    logger.info("Stage 1: Initializing model parameters...")

    # Step 1.1: Configuration defaults
    InputParam = st1_1_set_model_config()
    logger.info("  ✓ Configuration defaults set")

    # Step 1.2: Method mapping
    InputParam = st1_2_prepare_model_methods(InputParam)
    logger.info("  ✓ Methods assigned")

    # Step 1.3: User parameters (from JSON)
    InputParam = st1_3_set_model_user(InputParam, json_file)
    logger.info(f"  ✓ User parameters loaded from {json_file.name}")

    # Step 1.4: Advanced parameters
    InputParam = st1_4_set_model_advanced(InputParam)
    logger.info("  ✓ Advanced parameters set")

    # Validation
    validate_input_param(InputParam)
    logger.info("  ✓ Parameter validation passed")

    return InputParam


def validate_input_param(InputParam: InputParam) -> None:
    """
    Validate parameter consistency.
    Replicates MATLAB's implicit validation checks.
    """
    # Check domain array lengths
    n_layers = len(InputParam.Model['DomainRx'])
    required_keys = ['DomainRy', 'DomainTheta', 'DomainEcc', 'DomainEccAngle', 'DomainType']
    for key in required_keys:
        if len(InputParam.Model[key]) != n_layers:
            raise ValueError(f"Domain array length mismatch: {key}")

    # Check PML/ABC consistency
    if InputParam.Model['AddDomainType'].lower() != 'none':
        if InputParam.Model['AddDomainLoc'].lower() not in ['ext', 'int']:
            raise ValueError("AddDomainLoc must be 'ext' or 'int'")

    # Check frequency array
    if len(InputParam.Model['f_array']) == 0:
        raise ValueError("Frequency array is empty")

    # Check BCType length (should be n_layers + 1)
    if len(InputParam.Model['BCType']) != n_layers + 1:
        raise ValueError("BCType length should be n_layers + 1")