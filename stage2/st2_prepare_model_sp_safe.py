# D:\Работа\python\anisoDEpy\stage2\st2_prepare_model_sp_safe.py
"""
St2_PrepareModel_sp_safe.py
===========================
"""
import copy
from utils import debug_print

from stage2.st2_1_prepare_model_params_sp_safe import st2_1_prepare_model_params_sp_safe
from stage2.st2_2_prepare_model_methods_sp_safe import st2_2_prepare_model_methods_sp_safe

def st2_prepare_model_sp_safe(InputParam):
    """Stage 2: Prepare model for computation."""
    debug_print("=" * 70, level=1)
    debug_print("STAGE 2: Preparing Model", level=1)
    debug_print("=" * 70, level=1)

    CompStruct = copy.deepcopy(InputParam)

    # Убедимся, что структуры существуют
    CompStruct.setdefault('Advanced', {})
    CompStruct.setdefault('Methods', {})

    debug_print("Stage 2.1: Preparing model parameters...", level=2)
    CompStruct = st2_1_prepare_model_params_sp_safe(CompStruct)

    debug_print("Stage 2.2: Preparing model methods...", level=2)
    CompStruct = st2_2_prepare_model_methods_sp_safe(CompStruct)

    debug_print("Stage 2 complete", level=1)
    debug_print("=" * 70, level=1)
    return CompStruct