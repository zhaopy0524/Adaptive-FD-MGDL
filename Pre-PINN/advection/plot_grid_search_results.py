import os
import json
import matplotlib.pyplot as plt


def load_loss_data(folder_path):
    with open(os.path.join(folder_path, 'losses.json'), 'r') as f:
        loss_data = json.load(f)
    return loss_data


if __name__ == "__main__":
    folders = [
        ('MLP', 'results/grid_search/net:mlp_precond:none_lr:0.001_lambda:1'),
        ('MLP + Fourier Features', 'results/grid_search/net:deep_fourier_basis_1d_precond:none_lr:1e-05_lambda:0.001'),
        ('MLP + (Fourier Features + Preconditioning)', 'results/grid_search/net:deep_fourier_basis_1d_precond:poisson_fourier_basis_1d_lr:0.0001_lambda:0.1')
    ]

    plt.figure(figsize=(6.4, 3.5))
    for title, folder in folders:
        loss_data = load_loss_data(folder)
        epochs = loss_data['epoch']
        losses = loss_data['loss']

        plt.plot(epochs, losses, label=title)

    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='lower right', bbox_to_anchor=(1.15, 0.1), frameon=False)

    # plt.tight_layout(rect=[0, 0, 0.85, 1])

    plt.grid(True, linestyle='--')  # Adds grid

    # Remove outer borders
    for spine in ['top', 'right', 'bottom', 'left']:
        plt.gca().spines[spine].set_visible(False)

    # plt.show()
    plt.savefig('results/mlp_vs_mlpFF.pdf', format='pdf', bbox_inches='tight')
