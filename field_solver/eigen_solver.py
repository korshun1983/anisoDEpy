from scipy.sparse.linalg import eigsh
import numpy as np


def solve_safe(K, M, nev, sigma=1.0):
    """
    Solve quadratic eigen-value problem  (K - ω²M) v = 0
    by shift-and-invert:  (K - σM)⁻¹ M v = θ v,  θ = 1/(ω² - σ)
    Returns ω (real, rad/s) and eigen-vectors v.
    """
    A = K - sigma * M

    # θ = 1/(ω² - σ)  →  largest-magnitude θ gives ω closest to σ
    theta, v = eigsh(A, M=M, k=nev, sigma=sigma, which='LM')

    # ω² = σ + 1/θ
    omega2 = sigma + 1.0 / theta
    # ensure non-negative (round-off guard)
    omega2 = np.maximum(omega2, 0.0)
    omega = np.sqrt(omega2)

    return omega, v