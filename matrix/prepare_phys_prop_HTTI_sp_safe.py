# File 3: prepare_phys_prop_HTTI_sp_safe.py
import numpy as np


def prepare_phys_prop_HTTI_sp_safe(comp_struct, ii_l):
    """
    Prepare physical properties for HTTI (Homogeneous Tilted Transversely Isotropic) layer

    Inputs:
        comp_struct : dict
            Dictionary containing model parameters
        ii_l : int
            Layer number (0-indexed)

    Outputs:
        phys_prop : dict
            Dictionary containing HTTI properties
    """
    phys_prop = {}

    # Extract VTI parameters and inclination angles
    phys_prop['rho'] = comp_struct['Model']['DomainParam'][ii_l][0]
    phys_prop['c_VTI'] = comp_struct['Model']['DomainParam'][ii_l][1:6]  # C11, C33, C44, C66, C13
    phys_prop['theta'] = comp_struct['Model']['DomainParam'][ii_l][6]  # Inclination angle

    # Optional azimuth angle
    if len(comp_struct['Model']['DomainParam'][ii_l]) == 8:
        phys_prop['phi'] = comp_struct['Model']['DomainParam'][ii_l][7]
    else:
        phys_prop['phi'] = 0.0

    # Compute rotated elastic moduli tensor
    c_ij, c_ijkl = comp_struct['Methods']['em_tensor_VTI'](phys_prop['c_VTI'])
    rot_m = comp_struct['Methods']['rot_matrix'](phys_prop['theta'], phys_prop['phi'])
    c_ij_rot, c_ij_rot_back = comp_struct['Methods']['rot_c_ij'](c_ij, rot_m)

    phys_prop['c_ij'] = c_ij_rot_back

    return phys_prop