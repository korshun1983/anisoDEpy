"""
SAFE Pipeline Stage Implementations
"""

from .stage1 import initialize_model, st1_1_set_model_config, st1_2_prepare_model_methods
from .stage2 import prepare_model, prepare_model_params, prepare_model_methods
from .stage3 import run_stage3_matrix_assembly

try:
    from .stage4 import compute_solution
    __all__ = [
        'initialize_model', 'prepare_model', 'prepare_model_params',
        'prepare_model_methods', 'run_stage3_matrix_assembly', 'compute_solution'
    ]
except ImportError:
    # Stage 4 might not be ready yet
    __all__ = [
        'initialize_model', 'prepare_model', 'prepare_model_params',
        'prepare_model_methods', 'run_stage3_matrix_assembly'
    ]