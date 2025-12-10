# routines/asymptotes.py
"""
Asymptotic velocity calculations for SAFE method
Replicates ComputeAsymptotesSAFE.m and V_phase_VTI_exact_RPH.m
"""

import numpy as np
from typing import Dict, Any, Tuple


def compute_asymptotes_safe(CompStruct: Any) -> Dict[str, float]:
    """
    Replicates ComputeAsymptotesSAFE.m
    Computes asymptotic velocities for mode classification
    """
    asymptotes = {}

    # Check for mud domain (fluid)
    if 'mud_domain' in CompStruct.Model and CompStruct.Model['mud_domain'] > 0:
        mud_domain = CompStruct.Model['mud_domain']
        mud_props = CompStruct.Model['DomainParam'][mud_domain - 1]
        rho_mud = mud_props[0]
        lambda_mud = mud_props[1]
        asymptotes['V_mud'] = np.sqrt(lambda_mud / rho_mud) / 1e3  # km/s

    # Check outer domain type
    n_domain = CompStruct.Data['N_domain']
    outer_domain_type = CompStruct.Model['DomainType'][n_domain - 1]

    if outer_domain_type.lower() == 'htti':
        formation_props = CompStruct.Model['DomainParam'][n_domain - 1]
        rho = formation_props[0] * 1e3  # Convert to kg/m³
        c_main = formation_props[1:6]  # [c11, c13, c33, c44, c66]
        theta = formation_props[6]  # Inclination angle (degrees)

        # Compute VTI phase velocities
        v_qp, v_qsv, v_sh = v_phase_vti_exact_rph(rho, c_main, theta)

        asymptotes['V_qP'] = v_qp / 1e3  # km/s
        asymptotes['V_qSV'] = v_qsv / 1e3  # km/s
        asymptotes['V_SH'] = v_sh / 1e3  # km/s

        # Compute Stoneley wave if mud exists
        if 'V_mud' in asymptotes:
            c44 = c_main[3]
            asymptotes['V_St'] = 1.0 / np.sqrt(rho_mud * (1.0 / lambda_mud + 1.0 / c44))

    return asymptotes


def v_phase_vti_exact_rph(rho: float, c: np.ndarray, theta_deg: float) -> Tuple[float, float, float]:
    """
    Replicates V_phase_VTI_exact_RPH.m
    Computes exact phase velocities for VTI media using Rock Physics Handbook method
    """
    theta = np.deg2rad(theta_deg)
    c11, c13, c33, c44, c66 = c
    c12 = c11 - 2 * c66

    # Direction cosines
    n1 = np.sin(theta)
    n3 = np.cos(theta)

    # Christoffel matrix elements
    A = c11 * n1 ** 2 + c44 * n3 ** 2
    B = (c13 + c44) * n1 * n3
    C = c44 * n1 ** 2 + c33 * n3 ** 2

    # Solve quadratic for qP and qSV
    term1 = A + C
    term2 = A * C - B ** 2

    a, b_val = 1.0, -term1
    c_val = term2

    discriminant = b_val ** 2 - 4 * a * c_val
    discriminant = max(discriminant, 0.0)

    sqrt_disc = np.sqrt(discriminant)
    x1 = (-b_val + sqrt_disc) / (2 * a)
    x2 = (-b_val - sqrt_disc) / (2 * a)

    x_qp = max(x1, x2)
    x_qsv = min(x1, x2)

    # SH wave
    x_sh = c66 * n1 ** 2 + c44 * n3 ** 2

    # Convert to velocities
    v_qp = np.sqrt(x_qp / rho)
    v_qsv = np.sqrt(x_qsv / rho)
    v_sh = np.sqrt(x_sh / rho)

    return v_qp, v_qsv, v_sh