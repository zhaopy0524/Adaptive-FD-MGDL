"""Reproducibility helpers shared by MGDL experiments."""

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int, *, deterministic: bool = True) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(int(seed)))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass

        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            try:
                torch.backends.cuda.matmul.allow_tf32 = False
            except Exception:
                pass

        if hasattr(torch.backends, "cudnn"):
            try:
                torch.backends.cudnn.allow_tf32 = False
            except Exception:
                pass

__all__ = ["set_global_seed"]
