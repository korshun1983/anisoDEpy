import numpy as np
import math
from scipy.linalg import toeplitz

def chebdif(N, M)

    #  The function DM =  chebdif(N,M) computes the differentiation
    #  matrices D1, D2, ..., DM on Chebyshev nodes.
    #
    #  Input:
    #  N:        Size of differentiation matrix.
    #  M:        Number of derivatives required (integer).
    #  Note:     0 < M <= N-1.
    #
    #  Output:
    #  DM:       DM(1:N,1:N,ell) contains ell-th derivative matrix, ell=1..M.
    #
    #  The code implements two strategies for enhanced
    #  accuracy suggested by W. Don and S. Solomonoff in
    #  SIAM J. Sci. Comp. Vol. 6, pp. 1253--1268 (1994).
    #  The two strategies are (a) the use of trigonometric
    #  identities to avoid the computation of differences
    #  x(k)-x(j) and (b) the use of the "flipping trick"
    #  which is necessary since sin t can be computed to high
    #  relative precision when t is small whereas sin (pi-t) cannot.

    #  J.A.C. Weideman, S.C. Reddy 1998.

    I = np.identity(N)                          # Identity matrix.
    L = np.identity(N, dtype = bool)            # Logical identity matrix.

    n1 = math.floor(N/2)
    n2  = math.ceil(N/2)     # Indices used for flipping trick.

    k = np.arange(0, N, 1)                        # Compute theta vector.
    k = k.reshape(N, 1)
    th = k * math.pi / (N-1)

    x = math.sin(math.pi * (np.arange(N-1, -1-N, -2)).reshape(N,1)/(2*(N-1))) # Compute Chebyshev points.

    T = np.tile(th/2,(1,N))
    DX = 2*math.sin(T.transpose()+T) * math.sin(T.transpose()-T)         # Trigonometric identity.
    DX = [DX[1:n1,:] -np.flipud(np.fliplr(DX[1:n2,:]))]                  # Flipping trick.
    np.fill_diagonal(DX,1.0)                                            # Put 1's on the main diagonal of DX.

    C = toeplitz(pow((-1),k))               # C is the matrix with
    C[1,:] = C[1,:] * 2
    C[N,:] = C[N,:] * 2     # entries c(k)/c(j)
    C[:,1] = C[:,1]/2
    C[:,N] = C[:,N]/2

    Z = np.linalg.inv(DX)                           # Z contains entries 1/(x(k)-x(j))
    np.fill_diagonal(Z,0.)                      # with zeros on the diagonal.

    D = np.identity(N)                          # D contains diff. matrices.

    DM = np.array(N,N,M)
    for ell in range(1,M,1):
        D = ell * Z * (C * np.tile(np.diag(D),1,N) - D) # Off-diagonals
        D[L] = -np.sum(D.transpose(), axis = 0)                            # Correct main diagonal of D
        DM[:,:,ell] = D                                   # Store current D in DM

    return x, DM