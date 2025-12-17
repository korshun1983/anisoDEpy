"""
St1_SetModel.py
===============
EXACT MATLAB equivalent of St1_SetModel.m - modified to accept JSON model data.
"""

import numpy as np
# Импортируем ОБЕ функции из st1_3
from .st1_1_set_model_config import st1_1_set_model_config
from .st1_2_prepare_model_methods import st1_2_prepare_model_methods
from .st1_3_set_model_user import (
    st1_3_set_model_user_from_json,
    st1_3_set_model_user_sp_safe
)
from .st1_4_set_model_advanced_sp_safe import st1_4_set_model_advanced_sp_safe
from utils import debug_print


def st1_set_model(model_data: dict = None):
    """
    Main entry point for Stage 1.
    Calls all substages in exact MATLAB sequence.

    Parameters
    ----------
    model_data : dict, optional
        Loaded JSON model data. If None, uses placeholder model.

    Returns
    -------
    InputParam : dict
        Complete input parameter structure
    """
    debug_print("=" * 70, level=1)
    debug_print("STAGE 1: Setting Up Model Configuration", level=1)
    debug_print("=" * 70, level=1)

    # Initialize empty structure (like MATLAB's InputParam = struct)
    InputParam = {}

    # Stage 1.1: Configuration
    debug_print("Stage 1.1: Setting configuration...", level=2)
    InputParam = st1_1_set_model_config()

    # Stage 1.2: Prepare methods
    debug_print("Stage 1.2: Preparing model methods...", level=2)
    InputParam = st1_2_prepare_model_methods(InputParam)

    # Stage 1.3: User model parameters
    debug_print("Stage 1.3: Setting user model parameters...", level=2)
    if model_data is not None:
        # Use JSON data
        InputParam = st1_3_set_model_user_from_json(InputParam, model_data)
    else:
        # Use placeholder
        InputParam = st1_3_set_model_user_sp_safe(InputParam)

    # Stage 1.4: Advanced parameters
    debug_print("Stage 1.4: Setting advanced parameters...", level=2)
    InputParam = st1_4_set_model_advanced_sp_safe(InputParam)

    debug_print("Stage 1 complete: Model structure ready", level=1)
    debug_print("=" * 70, level=1)

    return InputParam