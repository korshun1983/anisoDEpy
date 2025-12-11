# methods/stage1.py
"""
===============================================================================
Stage 1: Model Initialization Pipeline
Replicates MATLAB St1_SetModel.m orchestration logic
===============================================================================
"""

import logging
import json
from pathlib import Path
from typing import Dict, Callable
from core.config import InputParam, MethodsContainer
from methods import stage2, stage3, stage4
from routines import meshgen, matrix_assembly
from routines.asymptotes import compute_asymptotes_safe, v_phase_vti_exact_rph

logger = logging.getLogger(__name__)


def st1_1_set_model_config() -> InputParam:
    """Replicates St1_1_SetModelConfig.m"""
    return InputParam()


def st1_2_prepare_model_methods(InputParam: InputParam) -> InputParam:
    """Replicates St1_2_PrepareModelMethods.m"""
    root_path = Path(__file__).parent.parent
    InputParam.Config.root_path = root_path

    # Assign utility methods
    InputParam.Methods['chebdif'] = None
    InputParam.Methods['em_tensor_VTI'] = matrix_assembly.em_tensor_vti
    InputParam.Methods['rot_c_ij'] = matrix_assembly.rotate_c_ij
    InputParam.Methods['rot_matrix'] = matrix_assembly.rot_matrix
    InputParam.Methods['V_phase_VTI_exact_RPH'] = v_phase_vti_exact_rph

    # ИСПРАВЛЕНИЕ: используем правильное имя функции из meshgen.py
    InputParam.Methods['MeshFaces'] = meshgen.prepare_mesh

    # Problem-specific method branching
    if InputParam.Config.ProblemType == 'spectrum':
        if InputParam.Config.NumMethod == 'SAFE':
            # Asymptotes for SAFE method
            InputParam.Methods['ComputeAsymptotes'] = compute_asymptotes_safe

            # Stage 2 methods
            InputParam.Methods['St2_PrepareModel'] = stage2.prepare_model
            InputParam.Methods['St2_1_PrepareModelParams'] = stage2.prepare_model_params
            InputParam.Methods['St2_2_PrepareModelMethods'] = stage2.prepare_model_methods

            # Stage 3 methods
            InputParam.Methods['St3_PrepareBasicMatrices'] = stage3.run_stage3_matrix_assembly

            # Stage 4 methods
            InputParam.Methods['St4_ComputeSolution'] = stage4.compute_solution

            # Set solver path
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

    # Create new InputParam from JSON
    InputParam_from_json = create_input_from_json(json_data)

    # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: копируем только данные, сохраняя Methods
    InputParam.Model = InputParam_from_json.Model
    InputParam.Config = InputParam_from_json.Config
    InputParam.Mesh = InputParam_from_json.Mesh
    InputParam.Advanced = InputParam_from_json.Advanced
    InputParam.Misc = InputParam_from_json.Misc
    InputParam.Data = InputParam_from_json.Data
    InputParam.Asymp = InputParam_from_json.Asymp

    InputParam._json_file = json_file

    # Post-processing
    if 'f_array' not in InputParam.Model:
        far = InputParam.Model['f_array_range']
        InputParam.Model['f_array'] = np.arange(
            far['start'], far['end'] + far['step'] / 2, far['step']
        ).tolist()

    InputParam.Model['N_disp'] = len(InputParam.Model['f_array'])

    # === ДОБАВЛЕНИЕ: Set AddDomain_Exist immediately after loading model ===
    InputParam.Model['AddDomainType'] = InputParam.Model.get('AddDomainType', 'none').lower()
    if InputParam.Model['AddDomainType'] == 'abc+pml':
        InputParam.Model['AddDomainType'] = 'pml+abc'

    if InputParam.Model['AddDomainType'] != 'none':
        InputParam.Model['AddDomain_Exist'] = 'yes'
    else:
        InputParam.Model['AddDomain_Exist'] = 'no'

    return InputParam


def st1_4_set_model_advanced(InputParam: InputParam) -> InputParam:
    """Replicates St1_4_SetModelAdvanced_sp_SAFE.m"""
    # Mesh visualization flag
    InputParam.Advanced.VisualizeMesh = True

    # Element order (cubic elements with 10 nodes)
    InputParam.Advanced.N_nodes = 10
    InputParam.Advanced.NEdge_nodes = 4

    # Eigenvalue solver options
    InputParam.Advanced.EigsOptions.disp = 0
    InputParam.Advanced.EigsOptions.tol = 1e-8

    # Source parameters
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


# -------------------------------------------------------------------------
# Main orchestrator
# -------------------------------------------------------------------------

def initialize_model(json_file: Path) -> InputParam:
    """Main Stage 1 orchestrator - replicates St1_SetModel.m"""
    logger.info("Stage 1: Initializing model parameters...")

    # Step 1.1: Configuration defaults
    InputParam = st1_1_set_model_config()
    logger.info("  [OK] Configuration defaults set")

    # Step 1.2: Method mapping
    InputParam = st1_2_prepare_model_methods(InputParam)
    logger.info("  [OK] Methods assigned")

    # Step 1.3: User parameters (from JSON)
    InputParam = st1_3_set_model_user(InputParam, json_file)
    logger.info(f"  [OK] User parameters loaded from {json_file.name}")

    # Step 1.4: Advanced parameters
    InputParam = st1_4_set_model_advanced(InputParam)
    logger.info("  [OK] Advanced parameters set")

    # Validation
    validate_input_param(InputParam)
    logger.info("  [OK] Parameter validation passed")

    return InputParam


def validate_input_param(InputParam: InputParam) -> None:
    """
    Validate parameter consistency.
    """
    logger.debug("  Validating InputParam...")

    # === MATLAB COMPATIBILITY: DomainRx are layer outer radii ===
    n_layers = len(InputParam.Model['DomainRx'])
    n_domains = len(InputParam.Model['DomainType'])

    if n_layers != n_domains:
        raise ValueError(f"DomainRx length ({n_layers}) must equal DomainType ({n_domains})")

    # Geometry arrays must match n_layers
    for key in ["DomainRy", "DomainTheta", "DomainEcc", "DomainEccAngle"]:
        actual_len = len(InputParam.Model[key])
        if actual_len != n_layers:
            raise ValueError(f"{key} length ({actual_len}) must match DomainRx ({n_layers})")

    # Domain arrays must match n_domains
    if 'DomainNth' in InputParam.Model and len(InputParam.Model['DomainNth']) != n_domains:
        raise ValueError(f"DomainNth length must match DomainType ({n_domains})")

    if len(InputParam.Model['DomainParam']) != n_domains:
        raise ValueError(f"DomainParam length must be {n_domains}")

    # BCType must match n_domains
    if len(InputParam.Model['BCType']) != n_domains:
        raise ValueError(f"BCType length must be {n_domains}")
    # ============================================================

    if len(InputParam.Model['f_array']) == 0:
        raise ValueError("Frequency array is empty")

    if InputParam.Model['AddDomainType'].lower() != 'none':
        if InputParam.Model['AddDomainLoc'].lower() not in ['ext', 'int']:
            raise ValueError("AddDomainLoc must be 'ext' or 'int'")