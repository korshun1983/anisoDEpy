# methods/stage4.py
"""
===============================================================================
Stage 4: Eigenvalue Solution for SAFE Method
Replicates St4_ComputeSolution_sp_SAFE.m with scipy.sparse.linalg.eigs
===============================================================================
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


def compute_solution(CompStruct: Any, BasicMatrices: Dict, FEMatrices: Dict,
                     FullMatrices: Dict) -> Dict[str, Any]:
    """
    Main Stage 4 orchestrator - solves generalized eigenvalue problem.

    Problem formulation:
    (K1 + i*k*K2 + k²*K3 - ω²*M - i*ω*P) * u = 0

    Linearized as:
    A = [0, I; -K, -i*K2] where K = K1 - ω²*M + i*ω*P
    B = [I, 0; 0, K3]

    Then solves: A * x = λ * B * x  where λ = k (wavenumber)
    """
    logger.info("      Stage 4: Solving eigenvalue problem...")

    # Get current frequency
    freq = CompStruct.Model['f_array'][CompStruct.if_grid - 1]
    omega = 2 * np.pi * freq * 1e3  # Convert kHz to Hz (MATLAB: omega_var_1000)

    logger.info(f"        Frequency: {freq:.2f} kHz, ω = {omega:.2e} rad/s")

    # Extract matrices
    K1 = FullMatrices['K1']
    K2 = FullMatrices['K2']
    K3 = FullMatrices['K3']
    M = FullMatrices['M']
    P = FullMatrices['P']

    # Form composite stiffness matrix: K = K1 - ω²*M + i*ω*P
    K = K1 - omega ** 2 * M + 1j * omega * P

    logger.debug(f"        K1 norm: {sp.linalg.norm(K1):.2e}")
    logger.debug(f"        M norm:  {sp.linalg.norm(M):.2e}")
    logger.debug(f"        K norm:  {sp.linalg.norm(K):.2e}")

    # Matrix dimensions
    n_dof = K.shape[0]
    Z = sp.csr_matrix((n_dof, n_dof), dtype=complex)
    I = sp.eye(n_dof, format='csr', dtype=complex)

    # Form block matrices A and B
    # A = [0, I; -K, -i*K2]
    A_top = sp.hstack([Z, I])
    A_bottom = sp.hstack([-K, -1j * K2])
    A = sp.vstack([A_top, A_bottom], format='csr')

    # B = [I, 0; 0, K3]
    B_top = sp.hstack([I, Z])
    B_bottom = sp.hstack([Z, K3])
    B = sp.vstack([B_top, B_bottom], format='csr')

    logger.debug(f"        A matrix: {A.shape}, nnz={A.nnz}")
    logger.debug(f"        B matrix: {B.shape}, nnz={B.nnz}")

    # Compute starting value for eigenvalue search
    # k_start = ω / EigSearchStart (where EigSearchStart is a velocity)
    v_start = CompStruct.Advanced['EigSearchStart'] * 1e3  # Convert km/s to m/s
    k_start = omega / v_start

    logger.info(f"        Target velocity: {v_start / 1e3:.2f} km/s, k_start: {k_start:.2e}")

    # Solver options
    num_eigs = CompStruct.Advanced['num_eig_max']
    tol = CompStruct.Advanced['EigsOptions']['tol']
    maxiter = CompStruct.Advanced['EigsOptions'].get('maxiter', 1000)

    try:
        # Solve generalized eigenvalue problem using shift-invert mode
        # scipy's eigs: A * x = lambda * M * x
        # Here M is our B matrix
        eigvals, eigvecs = eigs(
            A,
            k=num_eigs,
            M=B,
            sigma=k_start,
            which='LM',
            tol=tol,
            maxiter=maxiter,
            return_eigenvectors=True
        )

        logger.info(f"        eigs converged: {len(eigvals)} eigenvalues")

    except Exception as e:
        logger.error(f"        eigs failed: {str(e)}")
        logger.error(f"        Matrix condition: A={np.linalg.cond(A.todense()):.2e}")
        raise RuntimeError(f"Eigenvalue solver failed at f={freq} kHz") from e

    # Sort eigenvalues by magnitude (descending, matching MATLAB)
    sort_idx = np.argsort(np.abs(eigvals))[::-1]
    eigvals = eigvals[sort_idx]
    eigvecs = eigvecs[:, sort_idx]

    logger.info(f"        First 5 |k|: {np.abs(eigvals[:5])}")

    # Assemble Results structure (MATLAB-compatible)
    Results = {
        'REig_vals': np.diag(eigvals),  # As diagonal matrix
        'REig_vecs': eigvecs,
        'omega_val': omega,
        'frequency': freq,
        'k_start': k_start,
        'num_converged': len(eigvals)
    }

    # Log eigenvalue summary
    velocities = omega / np.abs(eigvals)
    logger.info(f"        Phase velocities (km/s): {velocities[:5] / 1e3}")

    return Results


def validate_eigenvalue_results(Results: Dict, n_dofs: int) -> None:
    """
    Validate eigenvalue solution quality.
    Replicates MATLAB's post-eigs checks.
    """
    eigvals = np.diag(Results['REig_vals'])
    n_converged = len(eigvals)

    if n_converged == 0:
        raise RuntimeError("No eigenvalues converged")

    # Check for NaN/Inf
    if np.any(np.isnan(eigvals)) or np.any(np.isinf(eigvals)):
        raise RuntimeError("Eigenvalues contain NaN or Inf")

    # Check eigenvector norms
    evecs = Results['REig_vecs']
    evec_norms = np.linalg.norm(evecs, axis=0)
    if np.any(evec_norms < 1e-10):
        logger.warning("Some eigenvectors have near-zero norm")

    logger.info(f"        Validated: {n_converged} eigenvalues, max|k|={np.max(np.abs(eigvals)):.2e}")