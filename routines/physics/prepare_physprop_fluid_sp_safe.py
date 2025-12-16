"""
PreparePhysProp_fluid_sp_SAFE.py
=================================
EXACT MATLAB equivalent of PreparePhysProp_fluid_sp_SAFE.m
"""

import numpy as np
from utils import debug_print


def prepare_physprop_fluid_sp_safe(CompStruct, ii_l):
    """
    Prepare physical properties for fluid layer.

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
    debug_print(f"      PreparePhysProp_fluid: Layer {ii_l + 1}", level=4)

    PhysProp = {}

    # Extract parameters: [density, lambda]
    PhysProp['rho'] = CompStruct['Model']['DomainParam'][ii_l][0]
    PhysProp['lambda'] = CompStruct['Model']['DomainParam'][ii_l][1]

    debug_print(f"        rho: {PhysProp['rho']:.3f}, lambda: {PhysProp['lambda'] / 1e9:.3f} GPa", level=4)

    return PhysProp