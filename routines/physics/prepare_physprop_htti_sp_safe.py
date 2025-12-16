"""
PreparePhysProp_HTTI_sp_SAFE.py
================================
EXACT MATLAB equivalent of PreparePhysProp_HTTI_sp_SAFE.m
"""

import numpy as np
from utils import debug_print


def prepare_physprop_htti_sp_safe(CompStruct, ii_l):
    """
    Prepare physical properties for HTTI layer.

    Parameters
    ----------
    CompStruct : dict
        Computational structure
    ii_l : int
        Layer number (0-based index)

    Returns
    -------
    PhysProp : dict
        Physical properties structure
    """
    debug_print(f"      PreparePhysProp_HTTI: Layer {ii_l + 1}", level=4)

    PhysProp = {}

    # Extract parameters: [density, c11, c13, c33, c44, c66, theta, phi]
    params = CompStruct['Model']['DomainParam'][ii_l]

    PhysProp['rho'] = params[0]
    PhysProp['c_VTI'] = params[1:6]  # c11, c13, c33, c44, c66
    PhysProp['theta'] = params[6] if len(params) > 6 else 0.0

    if len(params) == 8:
        PhysProp['phi'] = params[7]
    else:
        PhysProp['phi'] = 0.0

    debug_print(f"        rho: {PhysProp['rho']:.3f}, theta: {PhysProp['theta']:.3f} rad", level=4)
    debug_print(f"        c_VTI: {PhysProp['c_VTI']}", level=4)

    # TODO: Need em_tensor_VTI, rot_matrix, rot_c_ij functions
    # For now, store original parameters
    PhysProp['c_ij'] = None  # Will be computed later

    return PhysProp