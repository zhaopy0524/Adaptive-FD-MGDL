import os
import subprocess
from concurrent.futures import ThreadPoolExecutor


def run_script(pde, betas, precond, precond_on, output_folder):
    python_path = os.path.expanduser("~/miniconda3/envs/pytorch/bin/python")
    command = [
        python_path,
        "main.py",  # Replace with the name of your existing script
        # f"--nfreqs 5 30",
        # f"--lmbda use_golden_search",
        f"--lmbda 1",
        f"--nepochs 10000",
        f"--pde {pde}",
        f"--res 256 100",
        f"--bcs advection_2d",
        f"--precond_on {precond_on}",
        f"--precond {precond}",
        f"--net mlp",
        f"--pde_param {betas}",
        f"--output {output_folder}",
        f"--device cuda:1",
        f"--lr .0001",
        f"--optimizer adam"
    ]

    print('running command', ' '.join(command))
    subprocess.run(' '.join(command), shell = True)

if __name__ == '__main__':
    # pde = 'helmholtz'
    pde = 'advection'
    root = f'results/condn_vs_betas/{pde}'
    betass = range(3, 33, 3)

    # grad preconditioning
    # precond = f'{pde}_fourier_basis_1d'
    # precond_on = 'params'

    # # Param preconditioning
    # precond = 'helmholtz_fourier_basis_1d'
    # precond = f'{pde}_fourier_basis_1d'
    # precond_on = 'grads'
    #
    # # No preconditioning
    precond = 'none'
    precond_on = 'params' # dummy

    output_folders = [f'{root}/{precond}_{precond_on}/mlp_256_100/betas_{betas}' for betas in betass]
    # print(output_folders); exit()
    # pde_params = range(1, 30, 5)
    # Using ThreadPoolExecutor to run scripts in parallel
    l = len(betass)
    with ThreadPoolExecutor(max_workers = 10) as executor:
        # executor.map(run_script, pde_params, output_folders)
        executor.map(run_script, [pde]*l, betass, [precond]*l, [precond_on]*l, output_folders)
