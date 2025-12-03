# eigen_solver.py
from scipy.sparse.linalg import eigs
import numpy as np
import time


def solve_safe(K, M, nev=10, sigma=None, which='LR'):
    """
    Complex SAFE eigen-solver with timing.
    K, M - sparse matrices (may be complex).
    nev  - number of modes.
    sigma - complex shift (rad/s), default omega*1.1j.
    which - 'LR' or 'LM' (LR is faster for large problems).
    """
    if sigma is None:
        sigma = 1.0j * 1.1

    t0 = time.time()
    print('  Starting eigs...', end=' ')
    w, v = eigs(A=K, M=M, k=nev, sigma=sigma, which=which)
    print(f'done in {time.time() - t0:.2f}s')

    return w, v