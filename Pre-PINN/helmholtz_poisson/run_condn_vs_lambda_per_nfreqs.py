import concurrent
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
import itertools

import numpy as np

from run_grid_search import execute_grid_search


def train_model(pde, precond, lmbda, nfreqs, ):
    python_path = os.path.expanduser("~/miniconda3/envs/lie3.9/bin/python")
    command = [
        python_path,
        "main.py",  # Replace with the name of your existing script
        f"--nfreqs={nfreqs}",
        f"--lmbda={lmbda}",
        f"--nepochs=0",
        f"--pde={pde}",
        f"--res=1000",
        f"--optimizer=sgd",
        f"--bcs={'mixed_left' if pde=='helmholtz' else 'zero_bc'}",
        f"--precond_on=params",
        f"--precond={precond}",
        f"--net=fourier_basis_1d",
        f"--lr=0.01", # dummy
        f"--pde_param=4",
        f"--output={f'{root}/{precond}_{precond_on}/nfreqs_{nfreqs}_lambda_{lmbda}'}",
    ]
    print('running command', command)
    subprocess.run(command)


# pde = 'helmholtz'
pde = 'poisson'
root = f'results/condn_vs_lambda_per_nfreqs/{pde}'
nfreqss = range(1, 11)
preconds = ['none', f'{pde}_fourier_basis_1d']
precond_on = 'params' # dummy
lambda_values = np.logspace(-4, 6, num=20)  # The `num` parameter specifies the number of points
param_grid = list(itertools.product([pde], preconds, lambda_values, nfreqss))


# Function to execute grid search
def execute_grid_search(param_grid):
    results = {}
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(train_model, *args): args
                   for args in param_grid}

        for future in concurrent.futures.as_completed(futures):
            args = futures[future]
            try:
                result = future.result()
            except Exception as e:
                print(f"Exception occurred during training with {args}: {e}")
            else:
                results[args] = result
                print(f"Completed training with {args}, result={result}")

    return results


if __name__ == "__main__":
    print(f'running {len(param_grid)} experiments..')
    results = execute_grid_search(param_grid)
    print("Grid search results:", results)
