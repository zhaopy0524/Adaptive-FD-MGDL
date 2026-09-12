"""Common MGDL orchestration over the package-specific FD system."""

from importlib import import_module
from typing import List, Union

import numpy as np
import torch

from .model import MGDL


_package_leaf = __package__.rsplit(".", 1)[-1]
_variant_name = _package_leaf.removeprefix("MGDL_")
_fd_module = import_module(f".{_variant_name}_FD", package=__package__)

build_fd_system = _fd_module.build_fd_system


def solve_helmholtz_MGDL(N: int = 50,
                          kappa: float = 10.0,
                          *,
                          grade_hidden_dims: List[List[int]],
                          grade_activations: List[List[str]],
                          grade_epochs: List[int],
                          grade_lrs: List[float],
                          grade_lr_gammas: Union[List[float], None] = None,
                          use_sparse: bool = True,
                          use_sparse_coo: bool = False,
                          interactive: bool = False):
    """Solve the selected FD system and train its MGDL approximation."""
    X, Y, U, coords, A_csr, rhs, u_interior, use_complex, c = build_fd_system(
        N=N, kappa=kappa
    )

    # Prepare training data
    X_train = coords                     # (n_interior, 2)
    y_train = u_interior.astype(np.float32)  # (n_interior,)

    # Create the MGDL model
    model = MGDL(
        input_dim=2,
        output_dim=1,
        grade_hidden_dims=grade_hidden_dims,
        grade_activations=grade_activations
    )

    # Convert the finite-difference matrix and RHS to torch tensors for the operator loss
    if use_sparse:
        if use_sparse_coo:
            A_coo = A_csr.tocoo()
            indices = torch.tensor(np.vstack((A_coo.row, A_coo.col)), dtype=torch.int64)
            values = torch.tensor(
                A_coo.data.real.astype(np.float32) if use_complex else A_coo.data.astype(np.float32)
            )
            model.A = torch.sparse_coo_tensor(indices, values, size=A_csr.shape, dtype=torch.float32)
        else:
            A_data = A_csr.data.real.astype(np.float32) if use_complex else A_csr.data.astype(np.float32)
            A_indices = A_csr.indices
            A_indptr = A_csr.indptr
            model.A = torch.sparse_csr_tensor(A_indptr, A_indices, A_data, size=A_csr.shape, dtype=torch.float32)
        rhs_data = rhs.real.astype(np.float32) if use_complex else rhs.astype(np.float32)
        model.b = torch.from_numpy(rhs_data)
    else:
        A_dense = A_csr.toarray().real.astype(np.float32) if use_complex else A_csr.toarray().astype(np.float32)
        A_dense = torch.from_numpy(A_dense)
        rhs_data = rhs.real.astype(np.float32) if use_complex else rhs.astype(np.float32)
        rhs_dense = torch.from_numpy(rhs_data.astype(np.float32))
        model.A = A_dense
        model.b = rhs_dense

    # Train
    print("Starting MGDL training...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    u_true_interior = (np.sin(c * kappa * coords[:, 0]) * np.sin(c * kappa * coords[:, 1])).astype(np.float32)

    pred_train = model.fit(
        X=X_train,
        y=y_train,
        grade_epochs=grade_epochs,
        grade_lrs=grade_lrs,
        grade_lr_gammas=grade_lr_gammas,
        device=device,
        verbose=True,
        u_true=u_true_interior,
        interactive=interactive
    )

    # Get predictions on interior points
    u_pred_np = pred_train.cpu().numpy() if isinstance(pred_train, torch.Tensor) else pred_train

    # Build the full solution matrix U_nn
    U_nn = np.real(U).copy()   # Keep boundary values unchanged
    for i in range(1, N):
        for j in range(1, N):
            U_nn[i, j] = u_pred_np[(i - 1) * (N - 1) + (j - 1)]

    # Flatten loss history (kept for backward-compatible return format)
    loss_history_flat = [loss for grade_loss in model.loss_history for loss in grade_loss]

    return X, Y, np.real(U), U_nn, loss_history_flat, model

__all__ = ["build_fd_system", "solve_helmholtz_MGDL"]
