"""Solution plots and error analysis shared by MGDL variants."""

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_solution(
    X: np.ndarray,
    Y: np.ndarray,
    U: np.ndarray,
    title: str = "",
    *,
    save_dir: Optional[str] = None,
    prefix: str = "",
    show: bool = True,
):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X, Y, U, cmap="viridis")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("u(x1, x2)")
    ax.set_title(title)
    fig.tight_layout()
    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, f"{prefix}_surface.png"), dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig)

    fig2 = plt.figure()
    cp = plt.contourf(X, Y, U, levels=50, cmap="viridis")
    plt.colorbar(cp)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title + " (contour)")
    fig2.tight_layout()
    if save_dir is not None:
        fig2.savefig(os.path.join(save_dir, f"{prefix}_contour.png"), dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig2)


def plot_true_and_compare(
    X: np.ndarray,
    Y: np.ndarray,
    U: np.ndarray,
    U_nn: np.ndarray,
    kappa: float,
    *,
    save_dir: Optional[str] = None,
    show: bool = True,
):
    c = np.sqrt(2.0) / 2.0
    U_true = np.sin(c * kappa * X) * np.sin(c * kappa * Y)

    plot_solution(X, Y, U_true, "True solution", save_dir=save_dir, prefix="01_true", show=show)

    # Compute error metrics
    diff_U = U_true - U
    diff_U_nn = U_true - U_nn
    diff_U_Unn = U - U_nn

    rel_err_U = np.linalg.norm(diff_U) / (np.linalg.norm(U_true) + 1e-15)
    rel_err_U_nn = np.linalg.norm(diff_U_nn) / (np.linalg.norm(U_true) + 1e-15)
    rel_err_U_Unn = np.linalg.norm(diff_U_Unn) / (np.linalg.norm(U) + 1e-15)

    max_err_U = np.max(np.abs(diff_U))
    max_err_U_nn = np.max(np.abs(diff_U_nn))
    max_err_U_Unn = np.max(np.abs(diff_U_Unn))

    rse_U = np.sum(diff_U ** 2) / np.sum(U_true ** 2)
    rse_U_nn = np.sum(diff_U_nn ** 2) / np.sum(U_true ** 2)
    rse_U_Unn = np.sum(diff_U_Unn ** 2) / np.sum(U ** 2)

    print("\n========== Error Analysis ==========")
    print(f"True vs finite-difference (U)    : relative L2 error = {rel_err_U:.6e}, infinity-norm error = {max_err_U:.6e}, RSE = {rse_U:.6e}")
    print(f"True vs neural network (U_nn)    : relative L2 error = {rel_err_U_nn:.6e}, infinity-norm error = {max_err_U_nn:.6e}, RSE = {rse_U_nn:.6e}")

    plot_solution(X, Y, U, "Finite difference solution", save_dir=save_dir, prefix="02_fd", show=show)

    fig1 = plt.figure()
    cp1 = plt.contourf(X, Y, diff_U, levels=50, cmap="viridis")
    plt.colorbar(cp1)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Error: U_true - U")
    fig1.tight_layout()
    if save_dir is not None:
        fig1.savefig(os.path.join(save_dir, "04_error_true_minus_fd.png"), dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig1)

    plot_solution(X, Y, U_nn, "MGDL solution", save_dir=save_dir, prefix="03_mgdl", show=show)

    fig2 = plt.figure()
    cp2 = plt.contourf(X, Y, diff_U_nn, levels=50, cmap="viridis")
    plt.colorbar(cp2)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Error: U_true - U_nn")
    fig2.tight_layout()
    if save_dir is not None:
        fig2.savefig(os.path.join(save_dir, "05_error_true_minus_mgdl.png"), dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig2)

    return U_true, {
        "rel_err_U": rel_err_U,
        "rel_err_U_nn": rel_err_U_nn,
        "max_err_U": max_err_U,
        "max_err_U_nn": max_err_U_nn,
        "rse_U": rse_U,
        "rse_U_nn": rse_U_nn,
    }

__all__ = ["plot_solution", "plot_true_and_compare"]
