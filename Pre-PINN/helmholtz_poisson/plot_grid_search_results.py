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
        ('MLP + Preconditioned Fourier Features',
         'results/grid_search/net:deep_fourier_basis_1d_precond:poisson_fourier_basis_1d_lr:0.0001_lambda:0.1')
    ]

    colors = ['#ff7f0e', '#2ca02c']

    plt.figure(figsize=(6.4, 3.5))
    for idx, (title, folder) in enumerate(folders):
        loss_data = load_loss_data(folder)
        epochs = loss_data['epoch']
        losses = loss_data['loss']

        plt.plot(epochs, losses, label=title, color=colors[idx], alpha=0.7)

    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs Epochs')

    # plt.legend(loc='lower right', bbox_to_anchor=(1.15, 0.1), frameon=False)

    plt.grid(True, linestyle='--')  # Adds grid

    # Remove outer borders
    for spine in ['top', 'right', 'bottom', 'left']:
        plt.gca().spines[spine].set_visible(False)

    # plt.show()
    plt.savefig('results/mlp_vs_mlpFF.pdf', format='pdf', bbox_inches='tight')
