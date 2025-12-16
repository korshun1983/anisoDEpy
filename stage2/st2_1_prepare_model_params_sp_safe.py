"""
St2_1_PrepareModelParams_sp_SAFE.py
===================================
EXACT MATLAB equivalent of St2_1_PrepareModelParams_sp_SAFE.m
"""

import numpy as np
from utils import debug_print


def st2_1_prepare_model_params_sp_safe(CompStruct):
    """
    Stage 2.1: Prepare model parameters for spectral method.

    Parameters
    ----------
    CompStruct : dict
        Computational structure

    Returns
    -------
    CompStruct : dict
        Updated computational structure
    """
    debug_print("  St2_1: Preparing model parameters...", level=3)

    # Initialize Misc structure
    if 'Misc' not in CompStruct:
        CompStruct['Misc'] = {}

    # Define unit conversion factors for frequency
    freq_units = CompStruct['Config'].get('FreqUnits', 'kHz')
    if freq_units == 'Hz':
        CompStruct['Misc']['F_conv'] = 1.0
    elif freq_units == 'kHz':
        CompStruct['Misc']['F_conv'] = 1e3

    debug_print(f"    Frequency conversion: {CompStruct['Misc']['F_conv']}", level=4)

    # Define unit conversion factors for slowness
    slo_units = CompStruct['Config'].get('SloUnits', 'us/ft')
    if slo_units == 'us/m':
        CompStruct['Misc']['S_conv'] = 1e3
    elif slo_units == 'us/ft':
        CompStruct['Misc']['S_conv'] = 0.3048 * 1e3

    debug_print(f"    Slowness conversion: {CompStruct['Misc']['S_conv']}", level=4)

    # Initialize Data structure
    if 'Data' not in CompStruct:
        CompStruct['Data'] = {}

    # Compute number of computational domains
    CompStruct['Data']['N_domain'] = len(CompStruct['Model']['DomainType'])
    debug_print(f"    Number of domains: {CompStruct['Data']['N_domain']}", level=4)

    # Compute number of variables for each layer
    CompStruct['Data']['DVarNum'] = np.zeros(CompStruct['Data']['N_domain'], dtype=int)

    for ii_d in range(CompStruct['Data']['N_domain']):
        domain_type = CompStruct['Model']['DomainType'][ii_d]
        if domain_type == 'fluid':
            CompStruct['Data']['DVarNum'][ii_d] = 1
        elif domain_type == 'HTTI':
            CompStruct['Data']['DVarNum'][ii_d] = 3
        else:
            debug_print(f"Unknown domain type: {domain_type}", level=0)
            CompStruct['Data']['DVarNum'][ii_d] = 3

    debug_print(f"    DVarNum: {CompStruct['Data']['DVarNum']}", level=4)

    # Initialize Asymp structure
    CompStruct['Asymp'] = {}

    debug_print("  St2_1: Model parameters preparation complete", level=3)

    return CompStruct