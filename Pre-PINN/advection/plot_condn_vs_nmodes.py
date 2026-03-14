import os
import json
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D


def load_and_plot(output_folders, label, pde, fig):
    plt.subplots_adjust(wspace=0.75)

    nfreqs_list = []
    condition_numbers_list = []
    norm = plt.Normalize(0, len(output_folders) - 1)
    color_palette = [cm.viridis(norm(i)) for i in range(len(output_folders))]
    marker = 'o' if label == 'Precond.' else 's'

    for i, output_folder in enumerate(output_folders):
        loss_json_path = os.path.join(output_folder, "losses.json")
        results_json_path = os.path.join(output_folder, "results.json")
        config_json_path = os.path.join(output_folder, "config.json")

        with open(loss_json_path, 'r') as f:
            loss_data = json.load(f)

        with open(results_json_path, 'r') as f:
            result_data = json.load(f)

        with open(config_json_path, 'r') as f:
            config_data = json.load(f)

        epochs = loss_data['epoch']
        loss_res = loss_data['loss_res']
        condition_number = result_data.get('condition_number', None)

        config_data2 = [nfreq*2 for nfreq in config_data['nfreqs']]
        axs[1].plot(epochs, loss_res, label=f'{config_data2}' + r'$\times\pi$',
                    color=color_palette[i], linewidth=2, marker=marker, markevery=50)

        if condition_number:
            nfreqs = int(output_folder.split('_')[-1])
            nfreqs_list.append(nfreqs)
            condition_numbers_list.append(condition_number)

    axs[0].scatter(nfreqs_list, condition_numbers_list, c=color_palette[:len(nfreqs_list)], zorder=5, marker=marker)
    axs[0].set(xlabel=r'$\beta$', ylabel='Condition Number', title=r'$\beta$ vs Condition Number')

    # Customized plot properties
    # axs[1].set(xlabel='Epochs (Log Scale)', ylabel='Loss', title='Loss vs Epochs for Different nfreqs')
    axs[1].set_yscale("log")  # Log scale for x-axis
    axs[0].set_yscale("log")  # Log scale for x-axis
    # axs[0].set_xscale("log")  # Log scale for x-axis
    axs[1].set(xlabel='Epochs', ylabel='Loss', title=r'Loss vs Epochs for Different $\beta$')
    # axs[1].legend(loc='upper right')
    axs[1].grid(True, linestyle='--', linewidth=0.7)
    # axs[0].grid(True, linestyle='--', linewidth=0.7)

    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['bottom'].set_visible(False)
    axs[1].spines['left'].set_visible(False)

    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['bottom'].set_visible(False)
    axs[0].spines['left'].set_visible(False)
    # Custom grid settings for log-log plot
    axs[0].grid(True, which="major", ls="--", c='gray')
    axs[0].xaxis.grid(True, which="minor", ls=":", lw=0.25, c='gray')

    # Add a horizontal grid line at 10^0 (which is 1)
    # axs[0].axhline(1, color='red', linestyle='--', linewidth=0.7)

    # Custom legend for "Nº of Fourier Features" without marker
    if label == 'Precond.':
        custom_handles = [Line2D([0], [0], color=color_palette[i], lw=2) for i in range(len(color_palette))]
        labels = [str(2*n) + r'$\pi$' for n in sorted(set(nfreqs_list))]
        fig.legend(custom_handles, labels, loc='upper left', bbox_to_anchor=(0.44, 0.95), title=r'$\beta$', frameon=False)


    if pde == 'helmholtz':
        axs[1].set_ylim([1e-12, 1e7])
        axs[0].set_ylim([.5, 1e7])
        y_ticks = [10 ** k for k in range(0, 8, 2)]  # Example, make sure these values suit your data
        axs[0].set_yticks(y_ticks)
    elif pde == 'poisson':
        axs[1].set_ylim([1e-17, 1e7])
        axs[0].set_ylim([.5, 1e8+1])
        y_ticks = [10**k for k in range(0, 10, 2)]  # Example, make sure these values suit your data
        axs[0].set_yticks(y_ticks)
    elif pde == 'advection':
        axs[1].set_ylim([1e-12, 1e6])
        axs[0].set_ylim([.5, 1e6])
        y_ticks = [10 ** k for k in range(0, 7, 2)]  # Example, make sure these values suit your data
        axs[0].set_yticks(y_ticks)



if __name__ == '__main__':
    # folders = [f'results/no_precond/nfreqs_{i}' for i in range(6, 100, 10)]
    # folders = [f'results/precond/nfreqs_{i}' for i in range(6, 100, 10)]
    # folders = ([f'results/precond_newlr/nfreqs_{i}' for i in range(6, 100, 10)], 'Parameter preconditioning') # preconditioning on the params
    # folders = ([f'results/no_precond_newlr/nfreqs_{i}' for i in range(6, 100, 10)], 'no preconditioning') # no preconditioning
    # folders = ([f'results/no_precond_newlr_grad/nfreqs_{i}' for i in range(6, 100, 10)], 'Gradient preconditioning')

    # pde = 'helmholtz'
    # pde = 'poisson'
    pde = 'advection'

    args_list = []
    args_list.append(([f'results/condn_vs_betas/{pde}/advection_2d_params/5_30_256_100/betas_{i}' for i in range(3, 33, 3)], 'Precond.'))
    # args_list.append(([f'results/condn_vs_betas/{pde}/{pde}_fourier_basis_1d_params/nfreqs_{i}' for i in range(5, 100, 10)], pde, 'Precond.'))
    args_list.append(([f'results/condn_vs_betas/{pde}/none_params/5_30_256_100/betas_{i}' for i in range(3, 33, 3)], 'No Precond.'))

    m = .9
    fig, axs = plt.subplots(1, 2, figsize=(m * 6.4 * 2, m*3.4), constrained_layout=False)
    plt.subplots_adjust(left=0.1, right=0.95)

    # title = f'{pde.capitalize()}'
    # fig.suptitle(title, fontsize=16, y=1.06)
    for args in args_list:
        load_and_plot(*args, pde, fig)
    fig.legend([plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=10),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10)],
               ['No Precond.', 'Precond.',], loc='upper right', bbox_to_anchor=(0.565, 0.17), frameon=False)

    # save_title = f'{pde}_condn_vs_epochs_notitle'
    save_title = f'{pde}_(5, 30)_params_condn_vs_epochs'
    save_path = f"results/{save_title}.pdf"
    print('saving as.. ', save_path)

    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    # plt.show()