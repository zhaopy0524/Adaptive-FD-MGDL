"""Default experiment configuration retained from the original script."""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

if __package__ in (None, ""):
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if package_root not in sys.path:
        sys.path.insert(0, package_root)
    from MGDL_2th.run_utils import _make_run_dir
    from MGDL_2th.seed import set_global_seed
    from MGDL_2th.solver import solve_helmholtz_MGDL
    from MGDL_2th.visualization import plot_true_and_compare
else:
    from .run_utils import _make_run_dir
    from .seed import set_global_seed
    from .solver import solve_helmholtz_MGDL
    from .visualization import plot_true_and_compare


def main() -> None:
    seed = 42
    set_global_seed(seed, deterministic=True)
    
    N = 500                 # grid size
    kappa = 100             
    grade_hidden_dims = [[256, 256], [256], [256], [256]]
    grade_activations = [['sin', 'sin'], ['relu'], ['relu'], ['relu']]
    grade_epochs = [3000, 3000, 3000, 3000]      # epochs per grade
    grade_lrs = [1e-1, 1e-1, 1e-2, 1e-3]         # learning rate per grade[1e-1,1e-2,3]
    min_grade_lrs = [1e-2, 1e-2, 1e-3, 1e-3]     # target final learning rate after the last epoch of each grade (for ExponentialLR)
    
    
    # Compute ExponentialLR gamma from (initial lr -> final lr after the last epoch of each grade):
    # lr_T = lr_0 * gamma^T  =>  gamma = (lr_T/lr_0)^(1/T)
    grade_lr_gammas = []
    for lr0, lrT, T in zip(grade_lrs, min_grade_lrs, grade_epochs):
        if T <= 0:
            raise ValueError(f"grade_epochs must be positive, got {T}")
        if lr0 <= 0 or lrT <= 0:
            raise ValueError(f"learning rates must be positive, got lr0={lr0}, lrT={lrT}")
        grade_lr_gammas.append((lrT / lr0) ** (1.0 / T))
    use_sparse = True                # assemble matrix in sparse format (does not affect training)
    use_sparse_coo = True           # prefer COO sparse tensor (more mature API than sparse CSR)
    
    # Solve (returns model as well)
    X, Y, U, U_nn, loss_history_flat, model = solve_helmholtz_MGDL(
        N=N,
        kappa=kappa,
        grade_hidden_dims=grade_hidden_dims,
        grade_activations=grade_activations,
        grade_epochs=grade_epochs,
        grade_lrs=grade_lrs,
        grade_lr_gammas=grade_lr_gammas,
        use_sparse=use_sparse,
        use_sparse_coo=use_sparse_coo,
        interactive=True
    )

    # Build the run directory name from the final selected hyperparameters
    trained_grades = [
        i for i, cfg in enumerate(model.grade_train_hparams)
        if cfg is not None
    ]
    final_cfgs = [model.grade_train_hparams[i] for i in trained_grades]
    if len(final_cfgs) == 0:
        final_grade_epochs = [int(v) for v in grade_epochs]
        final_grade_lrs = [float(v) for v in grade_lrs]
        final_grade_lrTs = [float(v) for v in min_grade_lrs]
        final_grade_gammas = [float(v) for v in grade_lr_gammas]
    else:
        final_grade_epochs = [int(cfg["epochs"]) for cfg in final_cfgs]
        final_grade_lrs = [float(cfg["lr0"]) for cfg in final_cfgs]
        final_grade_lrTs = [float(cfg["lrT"]) for cfg in final_cfgs]
        final_grade_gammas = [float(cfg["gamma"]) for cfg in final_cfgs]

    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")
    run_dir = _make_run_dir(
        base_dir=base_dir,
        kappa=kappa,
        grade_epochs=final_grade_epochs,
        grade_lrs=final_grade_lrs,
        grade_lrTs=final_grade_lrTs,
    )

    # Compare and visualize
    U_true, metrics = plot_true_and_compare(
        X, Y, U, U_nn, kappa=kappa, save_dir=run_dir, show=True
    )
    model.print_network_structure()

    rse_history_flat = [v for grade_rse in model.rse_history for v in grade_rse]

    fig_rse = plt.figure(figsize=(8, 6))
    plt.semilogy(range(1, len(rse_history_flat) + 1), rse_history_flat)
    plt.xlabel('Epoch')
    plt.ylabel('RSE')
    plt.title('RSE (U_nn vs U_true) during training')
    plt.grid(True)
    fig_rse.tight_layout()
    fig_rse.savefig(os.path.join(run_dir, "06_rse_history.png"), dpi=300)
    plt.show()

    if len(loss_history_flat) > 0:
        fig_loss = plt.figure(figsize=(8, 6))
        plt.semilogy(range(1, len(loss_history_flat) + 1), loss_history_flat)
        plt.xlabel('Epoch')
        plt.ylabel('operator-loss (A@u - b)^2 mean')
        plt.title('Operator-loss during training (final attempts only)')
        plt.grid(True)
        fig_loss.tight_layout()
        fig_loss.savefig(os.path.join(run_dir, "07_operator_loss.png"), dpi=300)
        plt.show()

    diff_U = U_true - U
    diff_U_nn = U_true - U_nn
    diff_U_Unn = U - U_nn

    np.savez(
        os.path.join(run_dir, "results_arrays.npz"),
        X=X,
        Y=Y,
        U=U,
        U_nn=U_nn,
        U_true=U_true,
        diff_U=diff_U,
        diff_U_nn=diff_U_nn,
        diff_U_Unn=diff_U_Unn,
        rse_history=np.array(rse_history_flat, dtype=float),
        loss_history=np.array(loss_history_flat, dtype=float),
    )

    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    config = {
        "N": int(N),
        "kappa": float(kappa),
        "trained_grades": [int(i) for i in trained_grades],
        "num_grades": int(len(final_grade_epochs)),
        "grade_hidden_dims": grade_hidden_dims,
        "grade_activations": grade_activations,
        "grade_epochs": [int(v) for v in final_grade_epochs],
        "grade_lrs": [float(v) for v in final_grade_lrs],
        "grade_lrTs": [float(v) for v in final_grade_lrTs],
        "grade_lr_gammas": [float(v) for v in final_grade_gammas],
        "grade_train_hparams": model.grade_train_hparams,
        "total_train_time_sec": model.total_train_time_sec,
        "use_sparse": bool(use_sparse),
    }

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    ckpt = {
        "state_dict": model.state_dict(),
        "input_dim": int(model.input_dim),
        "output_dim": int(model.output_dim),
        "grade_hidden_dims": model.grade_hidden_dims,
        "grade_activations": model.grade_activations,
        "trained_grades": [int(i) for i in trained_grades],
        "grade_train_hparams": model.grade_train_hparams,
        "total_train_time_sec": model.total_train_time_sec,
    }
    torch.save(ckpt, os.path.join(run_dir, "model_checkpoint.pt"))



if __name__ == "__main__":
    main()
