"""Experiment-output directory helpers."""

import os
from typing import List


def _fmt_float_list(vals: List[float]) -> str:
    return "[" + ",".join(f"{float(v):g}" for v in vals) + "]"


def _fmt_int_list(vals: List[int]) -> str:
    return "[" + ",".join(str(int(v)) for v in vals) + "]"


def _make_run_dir(*, base_dir: str, kappa: float, grade_epochs: List[int], grade_lrs: List[float], grade_lrTs: List[float]) -> str:
    folder = (
        f"k_{float(kappa):g}+grade_{len(grade_epochs)}"
        f"+lr_{_fmt_float_list(grade_lrs)}"
        f"+lrT_{_fmt_float_list(grade_lrTs)}"
        f"+Epoch{_fmt_int_list(grade_epochs)}"
    )
    run_dir = os.path.join(base_dir, folder)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir
