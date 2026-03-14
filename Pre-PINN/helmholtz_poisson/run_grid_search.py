import concurrent
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
import itertools


def train_model(net, precond, learning_rate, lambda_val):
    python_path = os.path.expanduser("~/miniconda3/envs/lie3.9/bin/python")
    # precond = 'poisson_fourier_basis_1d' if net != 'mlp' else 'none'
    command = [
        python_path,
        "main.py",  # Replace with the name of your existing script
        f"--pde=poisson",
        f"--pde_param=4",
        f"--net={net}",
        f"--lr={learning_rate}",
        f"--lmbda={lambda_val}",
        f"--precond={precond}",
        f"--precond_on=net",
        # f"--optimizer={'adam' if net == 'mlp' else 'sgd'}",
        f"--optimizer=sgd",
        f"--output=results/grid_search/net:{net}_precond:{precond}_lr:{learning_rate}_lambda:{lambda_val}",
    ]
    print('running command', ' '.join(command))
    result = subprocess.run(command, capture_output=True, text=True)
    # Check that the script executed successfully
    if result.returncode == 0:
        # Extract the output (you may need to adjust this based on how your script returns its result)
        some_metric = result.stdout.strip()
        return some_metric
    else:
        print(f"Script failed with error: {result.stderr}")
        return None



# Define your grid
learning_rates = [0.00001, 0.0001, 0.001, 0.01, 0.1]
lambda_values = [0.001, 0.01, 0.1, 1, 10, 100]
nets = ['deep_fourier_basis_1d', 'mlp']
preconds = ['none']


# Generate all combinations
param_grid = list(itertools.product(nets, preconds, learning_rates, lambda_values))
param_grid_precond = list(itertools.product(['deep_fourier_basis_1d'], ['poisson_fourier_basis_1d'], learning_rates, lambda_values))

param_grid = param_grid + param_grid_precond

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
