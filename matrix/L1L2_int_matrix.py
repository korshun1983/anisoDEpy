# File 5: L1L2_int_matrix.py
import numpy as np
from math import factorial


def L1L2_int_matrix(n_degree):
    """
    Compute integrals of form ∫ L1^a L2^b ds over edge
    Using exact formula: a!b!/(a+b+1)!

    Inputs:
        n_degree : int
            Polynomial degree

    Outputs:
        ledge_int_matrix : ndarray
            Matrix of coefficients (n_degree × n_degree)
    """
    ledge_int_matrix = np.zeros((n_degree, n_degree))

    for aa in range(1, n_degree + 1):
        for bb in range(aa, n_degree + 1):
            a, b = aa - 1, bb - 1
            val = factorial(a) * factorial(b) / factorial(a + b + 1)
            ledge_int_matrix[aa - 1, bb - 1] = val
            ledge_int_matrix[bb - 1, aa - 1] = val  # Symmetric

    return ledge_int_matrix