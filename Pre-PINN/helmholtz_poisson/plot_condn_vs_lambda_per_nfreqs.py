from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import json
import os

def scan_folder(root_folder):
    results = []
    for root, dirs, files in os.walk(root_folder):
        if 'results.json' in files and 'config.json' in files and 'losses.json' in files:
            with open(os.path.join(root, 'results.json'), 'r') as f:
                result_data = json.load(f)
            with open(os.path.join(root, 'config.json'), 'r') as f:
                config_data = json.load(f)
            with open(os.path.join(root, 'losses.json'), 'r') as f:
                loss_data = json.load(f)

            try:
                final_loss = result_data['final_loss']
            except KeyError:
                print(f"Warning: 'final_loss' not found in {os.path.join(root, 'results.json')}")
                continue

            try:
                min_loss = min(loss_data['loss'])
            except (KeyError, ValueError):
                print(f"Warning: Could not extract minimum loss from {os.path.join(root, 'losses.json')}")
                continue

            try:
                epochs = loss_data['epoch']
                losses = loss_data['loss']
            except KeyError:
                print(f"Warning: Could not extract epochs or losses from {os.path.join(root, 'losses.json')}")
                continue

            results.append({
                'folder': root,
                # 'final_loss': final_loss,
                # 'min_loss': min_loss,
                # 'epochs': epochs,
                # 'losses': losses,
                'lr': config_data['lr'],
                'nfreqs': config_data['nfreqs'],
                'lmbda': config_data['lmbda'],
                'condition_number': result_data['condition_number'],
            })

    return results


def plot_results(results, results_noprecond, title):
    m = .9
    fig, axs = plt.subplots(1, 2, figsize=(m * 6.4 * 2, m * 3.4), constrained_layout=False)
    fig.suptitle(title, fontsize=16, y=1.06)
    plt.subplots_adjust(wspace=0.5)

    def add_line_plot(ax, results, alpha=1):
        grouped_by_nfreqs = defaultdict(list)
        for res in results:
            grouped_by_nfreqs[res['nfreqs']].append(res)

        sorted_nfreqs = sorted(grouped_by_nfreqs.keys())
        norm = plt.Normalize(min(sorted_nfreqs), max(sorted_nfreqs))
        color_palette = {nfreqs: cm.viridis(norm(nfreqs)) for nfreqs in sorted_nfreqs}

        for nfreqs in sorted_nfreqs:
            group = sorted(grouped_by_nfreqs[nfreqs], key=lambda x: float(x['lmbda']))  # Sort by lambda for plotting
            lambdas = [float(d['lmbda']) for d in group]
            condition_numbers = [float(d['condition_number']) for d in group]
            ax.plot(lambdas, condition_numbers, label=f'{nfreqs}',
                    linewidth=2, color=color_palette[nfreqs], alpha=alpha)

        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set(xlabel='$\lambda$', ylabel='Condition Number')
        ax.grid(True, which="major", ls="--", c='gray')
        ax.xaxis.grid(True, which="minor", ls=":", lw=0.25, c='gray')

    add_line_plot(axs[1], results, alpha=0.2)
    axs[1].set_title("With Preconditioning")
    add_line_plot(axs[0], results_noprecond)
    axs[0].set_title("Without Preconditioning")

    for i in range(2):
        axs[i].set_ylim([.5, 1e8 * 5])
        y_ticks = [10 ** k for k in range(0, 10, 2)]  # Example, make sure these values suit your data
        axs[i].set_yticks(y_ticks)
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].spines['bottom'].set_visible(False)
        axs[i].spines['left'].set_visible(False)

    handles, labels = axs[0].get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: int(t[0])))  # Sort legend by nfreqs
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.46, 0.5), frameon=False)

    # plt.tight_layout()
    save_title = title.replace(" ", "_").replace(",", "").replace('$', '').replace("\\", '').replace('º', '')
    save_path = f"results/{save_title}.pdf"
    print(f'Saving as {save_path}')
    # plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # root_folder = "results/grid_search"  # Replace with the root folder containing your experiment sub-folders
    results_noprecond = scan_folder("results/condn_vs_lambda_per_nfreqs/poisson/none_params")
    results = scan_folder("results/condn_vs_lambda_per_nfreqs/poisson/poisson_fourier_basis_1d_params")

    # Print all experiments from lowest to highest minimal loss
    print("All Experiments (sorted by minimal loss):")
    for res in results:
        print(res)
        # print(f"Folder: {res['folder']}")
        # print(f"Final Loss: {res['final_loss']}")
        # print(f"Minimum Loss: {res['min_loss']}")
        # print(f"Config: {res['config']}")
        print('-' * 40)

    print(f'{len(results)} experiments found!')




   # Your scan_folder function here...

        # results_noprecond = scan_folder("results/condn_vs_lambda_per_nfreqs/poisson/none_params")
        # results = scan_folder("results/condn_vs_lambda_per_nfreqs/poisson/poisson_fourier_basis_1d_params")

    plot_results(results, results_noprecond, "Condition Number vs $\lambda$ for Different Nº of Fourier Features")


