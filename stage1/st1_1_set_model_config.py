"""
St1_1_SetModelConfig.py
=======================
EXACT MATLAB equivalent of St1_1_SetModelConfig.m
"""

from utils import debug_print  # Import from root


def st1_1_set_model_config():
    """
    Set configuration parameters.

    Returns
    -------
    InputParam : dict
        Structure with Config parameters
    """
    debug_print("  St1_1: Setting configuration parameters...", level=3)

    InputParam = {}
    InputParam['Config'] = {}
    Config = InputParam['Config']

    # MATLAB equivalent parameters
    Config['ProblemType'] = 'spectrum'
    Config['NumMethod'] = 'SAFE'
    Config['SpeedUp'] = 'no'
    Config['SaveData'] = 'yes'
    Config['OuterBC'] = 'fixed'
    Config['PML'] = 'r2'
    Config['Eccentricity'] = 'no'
    Config['Symmetry'] = 'none'
    Config['EigenVar'] = 'k'
    Config['SloUnits'] = 'us/ft'
    Config['FreqUnits'] = 'kHz'
    Config['PressureUnits'] = 'GPa'
    Config['CheckAsymptote'] = 'yes'
    Config['DisplayAttenuation'] = 'yes'

    debug_print("  St1_1: Configuration complete", level=3)

    return InputParam