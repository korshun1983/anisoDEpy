# File 6: L1L2L3_int_matrix.py
import numpy as np
from math import factorial


def L1L2L3_int_matrix(n_degree):
    """
    Compute integrals of form ∫∫ L1^a L2^b L3^c dxdy over triangle
    Using exact formula: 2*a!b!c!/(a+b+c+2)!

    Inputs:
        n_degree : int
            Polynomial degree

    Outputs:
        lint_matrix : ndarray
            3D matrix of coefficients (n_degree × n_degree × n_degree)
    """
    lint_matrix = np.zeros((n_degree, n_degree, n_degree))

    for aa in range(1, n_degree + 1):
        for bb in range(aa, n_degree + 1):
            for cc in range(bb, n_degree + 1):
                a, b, c = aa - 1, bb - 1, cc - 1
                val = 2 * factorial(a) * factorial(b) * factorial(c) / factorial(a + b + c + 2)

                # Assign to all symmetric permutations
                indices = [
                    (aa - 1, bb - 1, cc - 1), (aa - 1, cc - 1, bb - 1),
                    (bb - 1, aa - 1, cc - 1), (bb - 1, cc - 1, aa - 1),
                    (cc - 1, aa - 1, bb - 1), (cc - 1, bb - 1, aa - 1)
                ]
                for i, j, k in indices:
                    lint_matrix[i, j, k] = val

    return lint_matrix