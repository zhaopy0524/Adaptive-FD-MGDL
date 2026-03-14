import os
import subprocess
from concurrent.futures import ThreadPoolExecutor


def run_script(pde, nfreqs, precond, precond_on, output_folder, ):
    python_path = os.path.expanduser("~/miniconda3/envs/lie3.9/bin/python")
    command = [
        python_path,
        "main.py",  # Replace with the name of your existing script
        f"--nfreqs={nfreqs}",
        f"--lmbda=use_golden_search",
        f"--nepochs=200",
        f"--pde={pde}",
        f"--res=1000",
        f"--bcs={'mixed_left' if pde=='helmholtz' else 'zero_bc'}",
        f"--precond_on={precond_on}",
        f"--precond={precond}",
        f"--net=fourier_basis_1d",
        f"--pde_param=4",
        f"--output={output_folder}",
    ]
    print('running command', command)
    subprocess.run(command)

if __name__ == '__main__':
    # pde = 'helmholtz'
    pde = 'poisson'
    root = f'results/condn_vs_nmodes/{pde}'
    nfreqss = range(5, 100, 10)

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

    output_folders = [f'{root}/{precond}_{precond_on}/nfreqs_{nfreqs}' for nfreqs in nfreqss]
    # print(output_folders); exit()
    # pde_params = range(1, 30, 5)
    # Using ThreadPoolExecutor to run scripts in parallel
    l = len(nfreqss)
    with ThreadPoolExecutor() as executor:
        # executor.map(run_script, pde_params, output_folders)
        executor.map(run_script, [pde]*l, nfreqss, [precond]*l, [precond_on]*l, output_folders)
