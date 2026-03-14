import os
import json
import math
import matplotlib.pyplot as plt
from matplotlib import cm


def load_and_plot(output_folders, title, pde):

    m = 1.2
    fig, axs = plt.subplots(figsize=(m * 6.4, m*3.4), constrained_layout=False)
    # fig, axs = plt.subplots(1, 2, figsize=(m * 6.4 * 2, m*3.4), constrained_layout=False)
    fig.suptitle(title, fontsize=16, y=1.04)
    plt.subplots_adjust(wspace=0.5)

    betas_list = []
    condition_numbers_list = []
    norm = plt.Normalize(0, len(output_folders) - 1)
    color_palette = [cm.viridis(norm(i)) for i in range(len(output_folders))]

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

        # axs[0].plot(epochs, loss_res, label=r'$2\pi *$' + f'{config_data["pde_param"]}',
        #             color=color_palette[i], linewidth=2)
        axs.plot(epochs, loss_res, label=f'{config_data["pde_param"]*2}' + r'$\pi$',
                    color=color_palette[i], linewidth=2)

        if condition_number:
            betas = int(output_folder.split('_')[-1])
            betas_list.append(betas*2*math.pi)
            condition_numbers_list.append(condition_number)

    # axs[1].scatter(betas_list, condition_numbers_list, s=70, c=color_palette[:len(betas_list)], zorder=5)
    # axs[1].set(xlabel='Values of Beta', ylabel='Condition Number', title='Values of Beta vs Condition Number')

    # # Customized plot properties
    # # axs[0].set(xlabel='Epochs (Log Scale)', ylabel='Loss', title='Loss vs Epochs for Different nfreqs')
    # axs[0].set_yscale("log")  # Log scale for x-axis
    # axs[1].set_yscale("log")  # Log scale for x-axis
    # axs[1].set_xscale("log")  # Log scale for x-axis
    # axs[0].set(xlabel='Epochs', ylabel='Loss', title='Loss vs Epochs for Different Values of Beta')
    # # axs[0].legend(loc='upper right')
    # axs[0].grid(True, linestyle='--', linewidth=0.7)
    # # axs[1].grid(True, linestyle='--', linewidth=0.7)

    # axs[0].spines['top'].set_visible(False)
    # axs[0].spines['right'].set_visible(False)
    # axs[0].spines['bottom'].set_visible(False)
    # axs[0].spines['left'].set_visible(False)
    axs.set_yscale("log")  # Log scale for x-axis
    axs.set(xlabel='Epochs', ylabel='Loss', title=r'Loss vs Epochs for Different Values of $\beta$')
    # axs.legend(loc='upper right')
    axs.grid(True, linestyle='--', linewidth=0.7)
    # axs[1].grid(True, linestyle='--', linewidth=0.7)

    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.spines['bottom'].set_visible(False)
    axs.spines['left'].set_visible(False)

    # axs[1].spines['top'].set_visible(False)
    # axs[1].spines['right'].set_visible(False)
    # axs[1].spines['bottom'].set_visible(False)
    # axs[1].spines['left'].set_visible(False)
    # # Custom grid settings for log-log plot
    # axs[1].grid(True, which="major", ls="--", c='gray')
    # axs[1].xaxis.grid(True, which="minor", ls=":", lw=0.25, c='gray')

    # Add a horizontal grid line at 10^0 (which is 1)
    # axs[1].axhline(1, color='red', linestyle='--', linewidth=0.7)

    # Add some space between subplots
    # handles, labels = axs[0].get_legend_handles_labels()
    handles, labels = axs.get_legend_handles_labels()
    # fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.46, 0.5), frameon=False)
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(.9, 0.5), frameon=False)

    if pde == 'helmholtz':
        axs[0].set_ylim([1e-12, 1e7])
        axs[1].set_ylim([.5, 1e7])
        y_ticks = [10 ** k for k in range(0, 8, 2)]  # Example, make sure these values suit your data
        axs[1].set_yticks(y_ticks)
    elif pde == 'poisson':
        axs[0].set_ylim([1e-17, 1e7])
        axs[1].set_ylim([.5, 1e8+1])
        y_ticks = [10**k for k in range(0, 10, 2)]  # Example, make sure these values suit your data
        axs[1].set_yticks(y_ticks)
    # elif pde == 'advection':
    #     axs[0].set_ylim([1e-12, 1e6])
    #     axs[1].set_ylim([.5, 1e6])
    #     y_ticks = [10 ** k for k in range(0, 7, 2)]  # Example, make sure these values suit your data
    #     axs[1].set_yticks(y_ticks)
    elif pde == 'advection':
        axs.set_ylim([1e-12, 1e6])

    save_title = title.replace(" ", "_").replace(",", "")
    save_path = f"results/{save_title}.pdf"
    print('saving as.. ', save_path)

    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    # plt.show()

if __name__ == '__main__':
    # folders = [f'results/no_precond/nfreqs_{i}' for i in range(6, 100, 10)]
    # folders = [f'results/precond/nfreqs_{i}' for i in range(6, 100, 10)]
    # folders = ([f'results/precond_newlr/nfreqs_{i}' for i in range(6, 100, 10)], 'Parameter preconditioning') # preconditioning on the params
    # folders = ([f'results/no_precond_newlr/nfreqs_{i}' for i in range(6, 100, 10)], 'no preconditioning') # no preconditioning
    # folders = ([f'results/no_precond_newlr_grad/nfreqs_{i}' for i in range(6, 100, 10)], 'Gradient preconditioning')

    args_list = []
    # args_list.append(([f'results/condn_vs_nmodes/helmholtz/helmholtz_fourier_basis_1d_grads/nfreqs_{i}' for i in range(5, 100, 10)],
    #            'Helmholtz, gradient preconditioning',
    #            'helmholtz'))
    # args_list.append(([f'results/condn_vs_nmodes/helmholtz/helmholtz_fourier_basis_1d_params/nfreqs_{i}' for i in range(5, 100, 10)],
    #            'Helmholtz, parameter preconditioning',
    #            'helmholtz'))
    # args_list.append(([f'results/condn_vs_nmodes/helmholtz/none_params/nfreqs_{i}' for i in range(5, 100, 10)],
    #            'Helmholtz, no preconditioning',
    #            'helmholtz'))

    # args_list.append(([f'results/condn_vs_nmodes/poisson/poisson_fourier_basis_1d_grads/nfreqs_{i}' for i in range(5, 100, 10)],
    #         'Poisson, gradient preconditioning',
    #         'poisson'))
    # args_list.append(([f'results/condn_vs_nmodes/poisson/poisson_fourier_basis_1d_params/nfreqs_{i}' for i in range(5, 100, 10)],
    #            'Poisson, parameter preconditioning',
    #            'poisson'))
    # args_list.append(([f'results/condn_vs_nmodes/poisson/none_params/nfreqs_{i}' for i in range(5, 100, 10)],
    #                      'Poisson, no preconditioning',
    #                      'poisson'))
    # args_list.append(([f'results/condn_vs_nmodes/poisson/none_params/nfreqs_{i}' for i in range(5, 100, 10)],
    #                      'Poisson, no preconditioning',
    #                      'poisson'))
    # args_list.append(([f'results/condn_vs_betas/advection/advection_2d_params/5_10/betas_{i}' for i in range(1, 15, 1)],
    #                         'Advection, frequencies (5, 10), parameter preconditioning',
    #                         'advection'))
    # args_list.append(([f'results/condn_vs_betas/advection/none_params/5_10/betas_{i}' for i in range(1, 15, 1)],
    #                         'Advection, frequencies (5, 10), no preconditioning',
    #                         'advection'))
    # args_list.append(([f'results/condn_vs_betas/advection/advection_2d_params/5_20/betas_{i}' for i in range(3, 33, 3)],
    #                         'Advection, frequencies (5, 20), parameter preconditioning',
    #                         'advection'))
    # args_list.append(([f'results/condn_vs_betas/advection/none_params/5_20/betas_{i}' for i in range(3, 33, 3)],
    #                         'Advection, frequencies (5, 20), no preconditioning',
    #                         'advection'))
    # args_list.append(([f'results/condn_vs_betas/advection/advection_2d_params/5_40_wo_lmbda/betas_{i}' for i in range(5, 55, 5)],
    #                         'Advection, frequencies (5, 40), parameter preconditioning',
    #                         'advection'))
    # args_list.append(([f'results/condn_vs_betas/advection/none_params/5_40_wo_lmbda/betas_{i}' for i in range(5, 55, 5)],
    #                         'Advection, frequencies (5, 40), no preconditioning',
    #                         'advection'))
    # args_list.append(([f'results/condn_vs_betas/advection/advection_2d_params/5_20_256_100/betas_{i}' for i in range(3, 33, 3)],
    #                         'Advection, frequencies (5, 20), parameter preconditioning',
    #                         'advection'))
    # args_list.append(([f'results/condn_vs_betas/advection/none_params/5_20_256_100/betas_{i}' for i in range(3, 33, 3)],
    #                         'Advection, frequencies (5, 20), no preconditioning',
    #                         'advection'))
    args_list.append(([f'results/condn_vs_betas/advection/none_params/mlp_256_100/betas_{i}' for i in range(3, 33, 3)],
                            'Advection, MLP, no preconditioning',
                            'advection'))
    
    for args in args_list:
        load_and_plot(*args)
