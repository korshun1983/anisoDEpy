import numpy as np


def dynl_matrix(nlmatrix, tri_props):
    """
    Compute y-derivative expansion matrix for shape functions.
    [T.Zharnikov, SMR v0.3_08.2014]
    Converted to Python 2025

    Args:
        nlmatrix: Shape function expansion coefficients (10x3x3x3)
        tri_props: Triangle properties with c coefficients

    Returns:
        dynl_matrix: y-derivative expansion coefficients (10x3x3x3)
    """

    # Allocate memory
    dynl_matrix = np.zeros((10, 3, 3, 3))

    # d/dy L_i = 1/(2*delta) * c_i
    dyl = 0.5 * tri_props['c']

    # Compute dyNLMatrix elements
    for nn in range(10):
        for ii in range(3):
            for jj in range(3):
                for kk in range(3):
                    term1 = dyl[0] * (ii + 1) * nlmatrix[nn, ii + 1, jj, kk] if ii < 2 else 0
                    term2 = dyl[1] * (jj + 1) * nlmatrix[nn, ii, jj + 1, kk] if jj < 2 else 0
                    term3 = dyl[2] * (kk + 1) * nlmatrix[nn, ii, jj, kk + 1] if kk < 2 else 0
                    dynl_matrix[nn, ii, jj, kk] = term1 + term2 + term3

    return dynl_matrix