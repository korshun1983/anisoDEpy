"""
St2_PrepareModel_sp_SAFE.py
===========================
EXACT MATLAB equivalent of St2_PrepareModel_sp_SAFE.m
"""

from utils import debug_print


def st2_prepare_model_sp_safe(InputParam):
    """
    Stage 2: Prepare model for computation.

    Parameters
    ----------
    InputParam : dict
        Input parameters from Stage 1

    Returns
    -------
    CompStruct : dict
        Computational structure ready for mesh generation
    """
    debug_print("=" * 70, level=1)
    debug_print("STAGE 2: Preparing Model", level=1)
    debug_print("=" * 70, level=1)

    # Retain all necessary information (MATLAB: CompStruct = InputParam)
    CompStruct = InputParam.copy()

    # Stage 2.1: Prepare model parameters
    debug_print("Stage 2.1: Preparing model parameters...", level=2)
    CompStruct = st2_1_prepare_model_params_sp_safe(CompStruct)

    # Stage 2.2: Prepare model methods
    debug_print("Stage 2.2: Preparing model methods...", level=2)
    CompStruct = st2_2_prepare_model_methods_sp_safe(CompStruct)

    debug_print("Stage 2 complete: Model ready for mesh generation", level=1)
    debug_print("=" * 70, level=1)

    return CompStruct


# Import substages at bottom
from .st2_1_prepare_model_params_sp_safe import st2_1_prepare_model_params_sp_safe
from .st2_2_prepare_model_methods_sp_safe import st2_2_prepare_model_methods_sp_safe