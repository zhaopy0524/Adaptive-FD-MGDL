"""Second-order finite-difference system assembly and reference solve."""

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


def build_fd_system(N: int = 50, kappa: float = 10.0):
    """Build and solve the second-order five-point Helmholtz reference system."""

    # Grid spacing and coordinates
    h = 1.0 / N
    kappa_to_alpha_eps = {100:0.4}
    alpha_eps = kappa_to_alpha_eps.get(kappa, 0.0)
    use_complex = (alpha_eps != 0)
    if use_complex:
        epsilon = alpha_eps * kappa * h
        kappa_c = kappa + 1j * epsilon
    else:
        epsilon = 0
        kappa_c = kappa              
    x = np.linspace(0.0, 1.0, N + 1)  # x1 direction
    y = np.linspace(0.0, 1.0, N + 1)  # x2 direction
    X, Y = np.meshgrid(x, y, indexing="ij")  # X[i,j] = x_i, Y[i,j] = y_j

    # Solution matrix: initialize to zeros and fill boundary values
    if use_complex:
        U = np.zeros((N + 1, N + 1), dtype=complex)
        kap = kappa_c
    else:
        U = np.zeros((N + 1, N + 1), dtype=float)
        kap = kappa

    # Handy constant
    c = np.sqrt(2.0) / 2.0

    # Boundary conditions
    # Gamma_1: x1 = 0, u = 0
    U[0, :] = 0.0

    # Gamma_2: x2 = 0, u = 0
    U[:, 0] = 0.0

    # Gamma_3: x1 = 1, u = sin(c * kappa) * sin(c * kappa * x2)
    U[N, :] = np.sin(c * kap) * np.sin(c * kap * y)

    # Gamma_4: x2 = 1, u = sin(c * kappa * x1) * sin(c * kappa)
    U[:, N] = np.sin(c * kap * x) * np.sin(c * kap)

    # Corner values do not enter the five-point stencil at interior points.

    # Number of interior unknowns (excluding boundary points)
    # Unknowns correspond to all interior points i,j = 1,...,N-1
    # Use second-order centered differences at every interior point.
    n_interior = (N - 1) * (N - 1)
    # Interior point coordinates (network inputs)
    coords = []   # each element is (x1, x2)
    for i in range(1, N):
        for j in range(1, N):
            coords.append([x[i], y[j]])
    coords = np.array(coords, dtype=np.float32)  # (n_interior, 2)

    # Sparse matrix A and RHS vector b
    if use_complex:
        A = lil_matrix((n_interior, n_interior), dtype=complex)
        b = np.zeros(n_interior, dtype=complex)
    else:
        A = lil_matrix((n_interior, n_interior), dtype=float)
        b = np.zeros(n_interior, dtype=float)

    inv_h2 = 1.0 / (h * h)

    def idx(i: int, j: int) -> int:
        """Map interior grid index (i, j) with 1 <= i,j <= N-1 to 1D index k in {0, ..., n_interior-1}.

        Uses row-major ordering with j (x2 direction) as the fast index.
        """
        return (i - 1) * (N - 1) + (j - 1)

    # Assemble A u = b for (Delta + kap^2) u = 0 using the
    # second-order five-point stencil at every interior grid point:
    #
    #   (u_{i-1,j} + u_{i+1,j} + u_{i,j-1} + u_{i,j+1}
    #    - 4 u_{i,j}) / h^2 + kap^2 u_{i,j} = 0.
    # Known Dirichlet boundary contributions are moved to the RHS.
    for i in range(1, N):
        for j in range(1, N):
            k = idx(i, j)
            A[k, k] = -4.0 * inv_h2 + kap * kap

            for ni, nj in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                if 1 <= ni <= N - 1 and 1 <= nj <= N - 1:
                    A[k, idx(ni, nj)] += inv_h2
                else:
                    b[k] -= inv_h2 * U[ni, nj]

    # Convert to CSR format for efficient solving
    A_csr = A.tocsr()

    # Solve the linear system
    u_interior = spsolve(A_csr, b)
    if use_complex:
        u_interior = np.real(u_interior)

    # Write the interior solution back into U
    for i in range(1, N):
        for j in range(1, N):
            U[i, j] = u_interior[idx(i, j)]

    return X, Y, U, coords, A_csr, b, u_interior, use_complex, c


__all__ = ["build_fd_system"]
