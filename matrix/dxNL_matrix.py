import numpy as np


def dxnl_matrix(nlmatrix, tri_props):
    """
    Compute x-derivative expansion matrix for shape functions.
    [T.Zharnikov, SMR v0.3_08.2014]
    Converted to Python 2025

    Args:
        nlmatrix: Shape function expansion coefficients (10x3x3x3)
        tri_props: Triangle properties with b coefficients

    Returns:
        dxnl_matrix: x-derivative expansion coefficients (10x3x3x3)
    """

    # Allocate memory for cubic elements (10 shape functions, up to 2nd power)
    dxnl_matrix = np.zeros((10, 3, 3, 3))

    # Coefficients of derivatives of interpolating functions
    # L_i = 1/(2*delta) * {a_i + b_i * x + c_i * y}
    # d/dx L_i = 1/(2*delta) * b_i
    # Factor 1/delta is omitted
    dxl = 0.5 * tri_props['b']

    # Compute dxNLMatrix elements
    for nn in range(10):
        for ii in range(3):
            for jj in range(3):
                for kk in range(3):
                    # Avoid index out of bounds - MATLAB allows this, Python doesn't
                    term1 = dxl[0] * (ii + 1) * nlmatrix[nn, ii + 1, jj, kk] if ii < 2 else 0
                    term2 = dxl[1] * (jj + 1) * nlmatrix[nn, ii, jj + 1, kk] if jj < 2 else 0
                    term3 = dxl[2] * (kk + 1) * nlmatrix[nn, ii, jj, kk + 1] if kk < 2 else 0
                    dxnl_matrix[nn, ii, jj, kk] = term1 + term2 + term3

    return dxnl_matrix
