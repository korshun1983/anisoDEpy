# routines/__init__.py

from .matrix_assembly import (
    assemble_basic_matrices,
    matrices_parts_htti,
    matrices_parts_fluid,
    ic_matrices_fluid_htti,
    ic_matrices_ff_ss,
    prepare_physprop_htti,
    prepare_physprop_fluid,
    em_tensor_vti,
    rotate_c_ij,
    rot_matrix
)

from .meshgen import prepare_mesh
from .io_utils import cleanup_output_dir, finalize_results
from .asymptotes import compute_asymptotes_safe, v_phase_vti_exact_rph

__all__ = [
    'assemble_basic_matrices',
    'matrices_parts_htti',
    'matrices_parts_fluid',
    'prepare_physprop_htti',
    'prepare_physprop_fluid',
    'prepare_mesh',
    'em_tensor_vti',
    'rotate_c_ij',
    'rot_matrix',
    'ic_matrices_fluid_htti',
    'ic_matrices_ff_ss',
    'compute_asymptotes_safe',  # Key function
    'v_phase_vti_exact_rph',     # Key function
    'cleanup_output_dir',
    'finalize_results'
]