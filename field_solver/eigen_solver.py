from scipy.sparse.linalg import eigs
import numpy as np


def solve_safe(K, M, nev=10, sigma=None):
    """
    Complex SAFE eigen-solver (replaces real eigsh).
    K, M - sparse matrices (may be complex).
    nev  - number of modes.
    sigma - complex shift (rad/s), default omega*1.1j.
    """
    if sigma is None:
        sigma = 1.0j * 1.1          # small imaginary shift if not given

    w, v = eigs(A=K, M=M, k=nev, sigma=sigma, which='LM')
    # w - complex eigen-values (rad/s), v - eigen-vectors
    return w, v