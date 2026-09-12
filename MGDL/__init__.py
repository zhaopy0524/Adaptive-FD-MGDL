"""Public API for a modularized MGDL finite-difference variant."""

from .model import MGDL, Sin
from .seed import set_global_seed
from .solver import build_fd_system, solve_helmholtz_MGDL
from .visualization import plot_solution, plot_true_and_compare

__all__ = [
    "MGDL",
    "Sin",
    "build_fd_system",
    "plot_solution",
    "plot_true_and_compare",
    "set_global_seed",
    "solve_helmholtz_MGDL",
]
